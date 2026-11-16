"""Paired cold GMPI loss controls over immutable generated candidate files."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np

from pathInference.experiment.contracts import freeze, forbid_formal_references, require, verify_files
from pathInference.research import candidate_scorer as previous
from pathInference.research.candidate_confidence import open_journal, commit_batch
from pathInference.research.gmpi_confidence_loss import replace_training_loss
from pathInference.utils import project_root, read_json, sha256, utc_now, write_json

ROOT = project_root()
CONFIG = ROOT / 'research/gmpi_confidence_v1/CONFIG.json'
RUN = ROOT / 'runs/20260201/gmpi_confidence_v1'
_RUNTIME = None


def config():
    return read_json(CONFIG)


def directory(source, variant):
    require(source in config()['sources'] and variant in config()['variants'], 'Unknown GMPI control')
    path = RUN / ('as' + source) / variant
    path.mkdir(parents=True, exist_ok=True)
    return path


def emit(source, variant, stage, **values):
    record = dict(source=source, variant=variant, stage=stage, updated_utc=utc_now(), **values)
    write_json(RUN / 'ACTIVE.json', record)
    if source and variant:
        write_json(directory(source, variant) / 'STATUS.json', record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def tensor_hash(state):
    value = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        value.update(json.dumps([name, str(array.dtype), array.shape]).encode())
        value.update(array.tobytes())
    return value.hexdigest()


def snapshot():
    cfg = config()
    snap = read_json(ROOT / cfg['source_snapshot'])
    require(set(snap['sources']) == set(cfg['sources']) and not snap['candidates_regenerated'],
            'Candidate snapshot scope differs')
    verify_files(snap['files_sha256'])
    files = dict(snap['files_sha256'])
    files.update({str(ROOT / cfg['source_snapshot']): sha256(ROOT / cfg['source_snapshot']),
                  str(CONFIG): sha256(CONFIG), str(Path(__file__)): sha256(__file__),
                  str(ROOT / 'src/pathInference/research/gmpi_confidence_loss.py'):
                      sha256(ROOT / 'src/pathInference/research/gmpi_confidence_loss.py'),
                  str(ROOT / 'research/gmpi_confidence_v1/run.py'):
                      sha256(ROOT / 'research/gmpi_confidence_v1/run.py')})
    sources = {}
    for source in cfg['sources']:
        worker = ROOT / cfg['baseline'] / 'T-PG-B/gmpi/p0' / ('as'+source) / 'control/gmpi' / ('as'+source) / ('seed'+str(cfg['seed']))
        path = worker / 'configuration.json'
        historical = read_json(path)
        arguments = historical['arguments']
        require(arguments['date'] == cfg['date'] and arguments['seed'] == cfg['seed'], 'Source seed/date differs')
        files[str(path)] = sha256(path)
        sources[source] = dict(snap['sources'][source], arguments=arguments)
    manifest_file = previous.RUN / 'data/data_manifest.json'
    manifest = read_json(manifest_file)
    graph = Path(manifest['graph_dir'])
    data_root = Path(manifest['source_data_root'])
    for file in (graph/'edge_frequency.json', graph/'AS_Feature.json', graph/'ASEmbedding_collectedPath_256.txt',
                 data_root/'data/aspath/path_by_vantage'/cfg['date']/'AS_frequency.json',
                 data_root/'data/aspath'/(cfg['date']+'.as-rel.txt')):
        require(file.is_file(), 'Missing frozen GMPI feature input: '+str(file))
        files[str(file)] = sha256(file)
    freeze(RUN / 'SNAPSHOT.json', dict(configuration=cfg, files_sha256=files, sources=sources,
        data_manifest=str(manifest_file), candidates_regenerated=False, test_labels_used_for_tuning=False,
        baseline='fresh paired controls; historical GMPI retained only as context'))


def runtime(source):
    global _RUNTIME
    if _RUNTIME is not None:
        _RUNTIME['env'].cur_vas = source
        return _RUNTIME
    import torch
    from pathInference.core import activate
    snap = read_json(RUN/'SNAPSHOT.json')
    activate(data_manifest=snap['data_manifest'])
    import environment
    import GMPI_pathLikelihoodEst as architecture
    import myDataset
    import PathInference_trainer as trainer
    from pathInference.downstream.device_runtime import configure_device
    device, info = configure_device(torch, 'cuda', config()['cpu_threads'])
    require(device.type == 'cuda', 'GMPI control requires CUDA')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    emit(source, None, 'loading_frozen_gmpi_environment')
    env = environment.Env(config()['date'], config()['date'])
    env.walksType = '_collectedPath'
    env.cur_vas = source
    env.epoch_test_batches = 0
    env.training_print_num = 100
    vectors = np.asarray(env.get_node_vectors(), np.float32)
    require(list(env.itos) == read_json(previous.RUN/('as'+source)/'VOCABULARY.json'), 'GMPI vocabulary differs')
    freeze(RUN/'RUNTIME.json', dict(python=__import__('sys').version,
        executable=__import__('sys').executable, torch=torch.__version__, numpy=np.__version__,
        gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda, device_runtime=info,
        deterministic_algorithms=True, tf32=False, vscode_hardware_acceleration_changed=False))
    _RUNTIME = dict(torch=torch, env=env, vectors=vectors, architecture=architecture, dataset=myDataset,
                    trainer=trainer, device=device)
    return _RUNTIME


def seed(torch):
    value = config()['seed']
    random.seed(value); np.random.seed(value); torch.manual_seed(value); torch.cuda.manual_seed_all(value)


def make_model(rt, arguments):
    torch = rt['torch']
    embedding = rt['env'].node2vecOutputDim + rt['env'].asrankFeatureDim
    model = rt['architecture'].attentionBasedScoreNetwork(torch.from_numpy(rt['vectors']),
        embedding*3, embedding, embedding*4, arguments['hidden_size'], 20).to(rt['device'])
    model.asEmbedding.weight.requires_grad = True
    require(model.input_semantics_version == 2, 'Frozen GMPI input semantics differ')
    return model


def atomic_save(torch, path, value):
    temporary = path.with_name(path.name+'.tmp')
    torch.save(value, str(temporary)); temporary.replace(path)


def train(source, variant):
    out, cfg = directory(source, variant), config()
    snap = read_json(RUN/'SNAPSHOT.json')
    bound = snap['sources'][source]
    arguments = bound['arguments']
    identity = dict(snapshot_sha256=sha256(RUN/'SNAPSHOT.json'), source=source, variant=variant)
    if (out/'MODEL_FROZEN.json').exists():
        record = read_json(out/'MODEL_FROZEN.json')
        require(record['identity'] == identity, 'Model identity differs')
        verify_files({record['model']:record['model_sha256']}); return
    with forbid_formal_references():
        rt = runtime(source)
        torch, env = rt['torch'], rt['env']
        seed(torch)
        model = make_model(rt, arguments)
        initial = tensor_hash(model.state_dict())
        freeze(RUN/('as'+source)/'INITIALIZATION.json', dict(state_dict_sha256=initial,
            seed=cfg['seed'], snapshot_sha256=identity['snapshot_sha256']))
        optimizer = torch.optim.Adam(model.parameters(), lr=arguments['lr'])
        env.k = arguments['k']
        env.train_loader_params = dict(batch_size=arguments['batch_size'], shuffle=False,
                                      num_workers=0, drop_last=False, pin_memory=True)
        env.test_loader_params = dict(batch_size=arguments['test_batch_size'], shuffle=False,
                                     num_workers=0, drop_last=False, pin_memory=True)
        split = read_json(Path(bound['binding']['split'])/'split_manifest.json')
        datasets = {name:rt['dataset'].IterableDatasetShuffle(str(Path(bound['binding']['split'])/(name+'.csv')),
            env,20,arguments['k'],60000,arguments['seed'],False) for name in ('train','validation')}
        training = rt['trainer'].train if variant == 'smooth_l1' else replace_training_loss(
            rt['trainer'].train, cfg['label_temperature'])
        best, stale, history, best_state = -float('inf'), 0, [], None
        checkpoint = out/'epoch_checkpoint.pt'
        if checkpoint.exists():
            resume = torch.load(str(checkpoint), map_location='cpu')
            require(resume['identity'] == identity and resume['initial_sha256'] == initial, 'Resume identity differs')
            model.load_state_dict(resume['model']); optimizer.load_state_dict(resume['optimizer'])
            best, stale, history, best_state = resume['best'], resume['stale'], resume['history'], resume['best_state']
            random.setstate(resume['python_rng']); np.random.set_state(resume['numpy_rng'])
            torch.set_rng_state(resume['torch_rng'].cpu())
            torch.cuda.set_rng_state_all([value.cpu() for value in resume['cuda_rng']])
            del resume
        emit(source,variant,'training',completed_epochs=len(history),initial_sha256=initial)
        for epoch in range(len(history), arguments['epochs']):
            if stale >= arguments['patience']:
                break
            start, cpu = time.monotonic(),time.process_time()
            loss,reward,count,_ = training(model,datasets['train'],optimizer,env,epoch,'train')
            vl,vr,vc,_,_,_ = rt['trainer'].test(model,datasets['validation'],optimizer,env,epoch,'surrogate_validation')
            require(count == split['row_counts']['train'] and vc == split['row_counts']['validation'], 'Epoch query counts differ')
            score = vr/vc
            require(np.isfinite(score) and np.isfinite(loss), 'Nonfinite GMPI training state')
            improved = score > best
            if improved:
                best,stale=score,0
                best_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
            else:
                stale+=1
            history.append(dict(epoch=epoch,training_jaccard=reward/count,validation_jaccard=score,
                training_loss=loss,validation_original_smooth_l1_loss=vl,checkpoint_improved=improved,
                wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu))
            atomic_save(torch,checkpoint,dict(identity=identity,initial_sha256=initial,
                model=model.state_dict(),optimizer=optimizer.state_dict(),best_state=best_state,
                best=best,stale=stale,history=history,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
            write_json(out/'TRAINING_HISTORY.json',history)
            emit(source,variant,'training',completed_epochs=len(history),best_validation_jaccard=best,
                 last_epoch=history[-1])
        require(best_state is not None and history, 'No trained GMPI state')
        model_file=out/'model.pt'
        atomic_save(torch,model_file,best_state)
        record=dict(identity=identity,model=str(model_file),model_sha256=sha256(model_file),
            initial_sha256=initial,configuration=arguments,label_temperature=cfg['label_temperature'],
            selected_epoch=max(history,key=lambda r:r['validation_jaccard'])['epoch'],
            validation_jaccard=best,epochs=len(history),final_labels_read=False,
            admitted_library_path_set_sha256=bound['binding']['admitted_library_path_set_sha256'])
        del model,optimizer,datasets,best_state
        torch.cuda.empty_cache()
    freeze(out/'MODEL_FROZEN.json',record)
    # The resumable optimizer state is redundant after the selected model is frozen.
    checkpoint.unlink(missing_ok=True)
    emit(source,variant,'model_frozen',epochs=record['epochs'],validation_jaccard=best)


def predict(source, variant):
    from torch.utils.data._utils.collate import default_collate
    require((RUN/'ALL_MODELS_FROZEN.json').is_file(), 'All models must freeze before prediction')
    out,cfg=directory(source,variant),config()
    snap=read_json(RUN/'SNAPSHOT.json'); bound=snap['sources'][source]
    frozen=read_json(out/'MODEL_FROZEN.json')
    verify_files({frozen['model']:frozen['model_sha256'],bound['candidate']:bound['candidate_sha256']})
    if (out/'PREDICTIONS_FROZEN.json').exists():
        p=read_json(out/'PREDICTIONS_FROZEN.json'); verify_files({p['predictions']:p['predictions_sha256']}); return
    database=open_journal(out/'prediction.sqlite',dict(model_sha256=frozen['model_sha256'],
        candidate_sha256=bound['candidate_sha256'],snapshot_sha256=sha256(RUN/'SNAPSHOT.json')))
    try:
        with forbid_formal_references():
            rt=runtime(source); torch=rt['torch']; args=bound['arguments']
            model=make_model(rt,args)
            model.load_state_dict(torch.load(frozen['model'],map_location=rt['device'])); model.eval()
            dataset=rt['dataset'].IterableDatasetShuffle(bound['candidate'],rt['env'],20,args['k'],60000,args['seed'],False)
            count=database.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
            offset=int(database.execute("SELECT value FROM metadata WHERE key='offset'").fetchone()[0])
            def flush(batch,position):
                nonlocal count
                item=default_collate(batch)
                tensors=[t.to(rt['device']) for t in item[:7]]
                with torch.no_grad():
                    scores=model(kpathsEmbedding=tensors[0],k_lastSecHops=tensors[1],kcurASPair=tensors[2],
                        hcurASPair=tensors[3],hispathEmbedding=tensors[4],his_lastSecHops=tensors[5],hisASPairsEmbedding=tensors[6]).squeeze(-1)
                    require(bool(torch.isfinite(scores).all()), 'Nonfinite GMPI prediction scores')
                    slots=torch.arange(args['k'],device=rt['device'])[None,:]
                    sizes=item[9].to(rt['device']).clamp(max=args['k'])[:,None]
                    choices=scores.masked_fill(slots>=sizes,-float('inf')).argmax(dim=1).cpu().tolist()
                rows=[]
                for i,slot in enumerate(choices):
                    pair=item[8][i].replace('-','->')
                    nodes=[str(rt['env'].itos[n]) for n in item[0][i,slot].tolist() if n != rt['env'].stoi['<pad>']]
                    rows.append((pair,'->'.join(nodes),'predicted'))
                commit_batch(database,rows,position); count+=len(batch)
                emit(source,variant,'predicting',committed_queries=count,candidate_byte_offset=position,
                     candidate_bytes=Path(bound['candidate']).stat().st_size)
            with Path(bound['candidate']).open('rb') as stream:
                stream.seek(offset); batch=[]
                for raw in iter(stream.readline,b''):
                    row=next(csv.reader([raw.decode('utf-8')],strict=True))
                    require(set(row[3].split('|')) <= {'0.0'}, 'Test labels must remain zero')
                    batch.append(dataset.convertToTensor(row))
                    if len(batch)>=args['test_batch_size']:
                        flush(batch,stream.tell()); batch=[]
                if batch: flush(batch,stream.tell())
            inventory=read_json(bound['inventory'])['pairs']
            with Path(bound['dispatch']).open() as stream:
                states={r['pair']:r['status'] for r in map(json.loads,stream)}
            require(set(states)==set(inventory),'Query inventory differs')
            with database:
                for pair in inventory:
                    if states[pair]=='model_candidates':
                        require(database.execute('SELECT 1 FROM predictions WHERE pair=?',(pair,)).fetchone(), 'Missing prediction')
                    else:
                        require(states[pair] in ('direct_graph_edge_candidate','same_as_query','no_usable_candidate'), 'Unknown dispatch status')
                        direct=states[pair]=='direct_graph_edge_candidate'
                        database.execute('INSERT OR IGNORE INTO predictions VALUES(?,?,?)',
                                         (pair,pair if direct else '', 'direct_graph_edge' if direct else states[pair]))
            require(database.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]==len(inventory),'Prediction denominator differs')
            output=out/'ranked_predictions.csv'; temporary=out/'ranked_predictions.tmp.csv'
            with temporary.open('w',newline='') as stream:
                writer=csv.writer(stream); writer.writerow(['pair','predicted_path','status'])
                for pair in inventory:
                    writer.writerow([pair,*database.execute('SELECT path,status FROM predictions WHERE pair=?',(pair,)).fetchone()])
            temporary.replace(output)
            del model; torch.cuda.empty_cache()
        freeze(out/'PREDICTIONS_FROZEN.json',dict(model_sha256=frozen['model_sha256'],
            predictions=str(output),predictions_sha256=sha256(output),candidate_sha256=bound['candidate_sha256'],
            queries=len(inventory),final_labels_read=False))
        emit(source,variant,'predictions_frozen',queries=len(inventory))
    finally:
        database.close()


def evaluate(source, variant):
    from pathInference.downstream.evaluation import score_source
    require((RUN/'ALL_PREDICTIONS_FROZEN.json').is_file(), 'Freeze all paired predictions before evaluation')
    out=directory(source,variant)
    if (out/'RESULTS.json').exists(): return
    frozen=read_json(out/'PREDICTIONS_FROZEN.json'); model=read_json(out/'MODEL_FROZEN.json')
    verify_files({frozen['predictions']:frozen['predictions_sha256'],model['model']:model['model_sha256']})
    baseline=read_json(previous.RUN/('as'+source)/'RESULTS.json')
    result=score_source(baseline['reference_file'],frozen['predictions'],source,baseline['fixed_identity'])
    result.update(source=source,variant=variant,rho=baseline['rho'],oracle_jaccard=baseline['oracle_jaccard'],
        candidate_exact_available=baseline['candidate_exact_available'],candidate_sha256=frozen['candidate_sha256'],
        model_sha256=frozen['model_sha256'],initial_sha256=model['initial_sha256'],
        historical_gmpi_jaccard=baseline['baseline_gmpi_jaccard'],final_labels_used_for_tuning=False)
    freeze(out/'RESULTS.json',result)
    emit(source,variant,'evaluated',jaccard=result['jaccard_accuracy'])
