"""Frozen-candidate representation pretraining with resumable audited stages."""
from __future__ import annotations

import csv
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import resource
import threading
import time

import numpy as np

from pathInference.experiment.contracts import freeze, forbid_formal_references, require, verify_files
from pathInference.utils import project_root, read_json, sha256, utc_now, write_json

ROOT = project_root()
CONFIG = ROOT / 'research/representation_pretraining_v1/CONFIG.json'
RUN = ROOT / 'runs/20260201/representation_pretraining_v1'


def code_identity():
    paths = [ROOT / 'src/pathInference/research' / name for name in
             ('representation_pretraining.py', 'representation_experiment.py',
              'path_sequence_confidence.py', 'gmpi_confidence.py', 'gmpi_confidence_loss.py')]
    paths += [ROOT / 'research/representation_pretraining_v1/run.py']
    return {str(path): sha256(path) for path in paths}


def make_engine():
    spec = importlib.util.spec_from_file_location('pathInference.research._pretrained_engine',
                                                 ROOT / 'src/pathInference/research/gmpi_confidence.py')
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    engine.CONFIG, engine.RUN = CONFIG, RUN

    def make_model(runtime, arguments):
        from pathInference.research.path_sequence_confidence import OrderedPathConfidence
        return OrderedPathConfidence(runtime['torch'].from_numpy(runtime['vectors']),
                                     **read_json(CONFIG)['architecture']).to(runtime['device'])
    engine.make_model = make_model
    return engine


def receipt(source, stage, variant=None):
    return RUN / 'tasks' / ('_'.join(p for p in (source, variant, stage) if p) + '.json')


def checked(source, stage, variant=None):
    path = receipt(source, stage, variant)
    if not path.exists():
        return False
    record = read_json(path)
    require(record['code_sha256'] == code_identity() and record['config_sha256'] == sha256(CONFIG),
            'Pretraining code/config changed; do not silently invalidate verified caches')
    verify_files(record['outputs_sha256'])
    return True


def record_task(source, stage, variant, files):
    freeze(receipt(source, stage, variant), dict(source=source, variant=variant, stage=stage,
        completed_utc=utc_now(), code_sha256=code_identity(), config_sha256=sha256(CONFIG),
        outputs_sha256={str(path): sha256(path) for path in files}))


def snapshot():
    cfg = read_json(CONFIG)
    original = read_json(ROOT / cfg['source_snapshot'])
    cold = ROOT / cfg['cold_control']
    audit = read_json(cold / 'COMPLETION_AUDIT.json')
    require(audit['status'] == 'PASS', 'Cold architecture control is not independently verified')
    require(cfg['sources'] == original['configuration']['sources']
            and cfg['architecture'] == original['configuration']['architecture']
            and cfg['seed'] == original['configuration']['seed']
            and cfg['label_temperature'] == original['configuration']['label_temperature'],
            'Pretraining comparison changed architecture, source scope, objective or seed')
    files = dict(original['files_sha256'])
    files.update(code_identity())
    for path in (CONFIG, ROOT / cfg['source_snapshot'], cold / 'COMPLETION_AUDIT.json'):
        files[str(path)] = sha256(path)
    for source in cfg['sources']:
        for path in [cold / ('as' + source) / 'INITIALIZATION.json',
                     *[cold / ('as' + source) / 'ordered_path' / name for name in
                       ('MODEL_FROZEN.json', 'PREDICTIONS_FROZEN.json', 'RESULTS.json', 'model.pt', 'ranked_predictions.csv')],
                     ROOT / cfg['frozen_candidate_run'] / 'augmentation' / ('as' + source) / 'ADMISSION.json']:
            files[str(path)] = sha256(path)
    verify_files(files)
    freeze(RUN / 'SNAPSHOT.json', dict(configuration=cfg, sources=original['sources'],
        data_manifest=original['data_manifest'], files_sha256=files,
        candidates_regenerated=False, test_labels_used_for_tuning=False,
        historical_test_results_already_known=True, extra_path_evidence=False))


def prepare_corpus(source):
    from pathInference.research.representation_pretraining import training_corpus
    cfg = read_json(CONFIG)
    binding = read_json(RUN / 'SNAPSHOT.json')['sources'][source]
    base = ROOT / cfg['frozen_candidate_run']
    admission = read_json(base / 'augmentation' / ('as' + source) / 'ADMISSION.json')
    vocabulary = read_json(base / ('as' + source) / 'VOCABULARY.json')
    with forbid_formal_references():
        paths, manifest = training_corpus(admission['path'], Path(binding['binding']['split']) / 'split_manifest.json',
                                            vocabulary, source, cfg['architecture']['max_length'])
    require(manifest['admitted_library_path_set_sha256'] == admission['path_set_sha256']
            == binding['binding']['admitted_library_path_set_sha256'], 'Admitted library changed')
    out = RUN / ('as' + source)
    out.mkdir(exist_ok=True)
    path = out / 'pretraining_paths.npy'
    temporary = out / 'pretraining_paths.tmp.npy'
    np.save(str(temporary), paths, allow_pickle=False)
    if path.exists():
        require(sha256(path) == sha256(temporary), 'Derived training corpus differs on resume')
        temporary.unlink()
    else:
        temporary.replace(path)
    manifest.update(corpus=str(path), corpus_sha256=sha256(path), library=admission['path'],
                    library_sha256=admission['sha256'], vocabulary_sha256=sha256(base / ('as' + source) / 'VOCABULARY.json'),
                    split_sha256=sha256(Path(binding['binding']['split']) / 'split_manifest.json'))
    freeze(out / 'CORPUS.json', manifest)
    return [path, out / 'CORPUS.json']


def capture_rng(torch):
    return dict(python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())


def restore_rng(torch, state):
    random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda_rng']])


def cold_model(engine, source):
    cfg = read_json(CONFIG)
    rt = engine.runtime(source)
    engine.seed(rt['torch'])
    args = read_json(RUN / 'SNAPSHOT.json')['sources'][source]['arguments']
    model = engine.make_model(rt, args)
    cold = read_json(ROOT / cfg['cold_control'] / ('as' + source) / 'INITIALIZATION.json')
    require(engine.tensor_hash(model.state_dict()) == cold['state_dict_sha256'],
            'Cold initialization differs from audited architecture control')
    if not (RUN / 'BASE_VECTORS.npy').exists():
        temporary = RUN / 'BASE_VECTORS.tmp.npy'
        np.save(str(temporary), rt['vectors'], allow_pickle=False)
        temporary.replace(RUN / 'BASE_VECTORS.npy')
    else:
        require(np.array_equal(np.load(str(RUN / 'BASE_VECTORS.npy')), rt['vectors']),
                'Shared AS vectors/vocabulary differ across sources')
    freeze(RUN / 'BASE_VECTORS.json', dict(path=str(RUN / 'BASE_VECTORS.npy'),
                                         sha256=sha256(RUN / 'BASE_VECTORS.npy')))
    return model, rt, args


def changed_parameters(before, after):
    return [name for name in before if not before[name].equal(after[name].detach().cpu())]


def verify_stage_changes(stage, changed):
    from pathInference.research.representation_pretraining import PATH_MODULES
    require(changed, 'Self supervision did not change any transferred parameter')
    allowed = ('asEmbedding',) if stage == 'as_context' else PATH_MODULES
    require(all(name.split('.')[0] in allowed for name in changed), 'Pretraining changed unrelated scoring parameters')


def pretrain(engine, source, stage):
    from pathInference.research.representation_pretraining import configure_stage, stage_epoch
    cfg = read_json(CONFIG)
    out = RUN / ('as' + source) / ('pretrain_' + stage)
    out.mkdir(exist_ok=True)
    identity = dict(snapshot_sha256=sha256(RUN / 'SNAPSHOT.json'), source=source, stage=stage,
                    corpus_sha256=sha256(RUN / ('as' + source) / 'pretraining_paths.npy'))
    parent = None
    if stage == 'path_context':
        parent = read_json(RUN / ('as' + source) / 'pretrain_as_context/PRETRAIN_FROZEN.json')
        verify_files({parent['model']: parent['model_sha256']})
        identity['parent_sha256'] = parent['model_sha256']
    frozen_path = out / 'PRETRAIN_FROZEN.json'
    if frozen_path.exists():
        frozen = read_json(frozen_path)
        require(frozen['identity'] == identity, 'Pretraining resume identity differs')
        verify_files({frozen['model']: frozen['model_sha256']})
        return [frozen_path, Path(frozen['model']), out / 'HISTORY.json']
    with forbid_formal_references():
        model, rt, _ = cold_model(engine, source)
        torch = rt['torch']
        if stage == 'path_context':
            model.load_state_dict(torch.load(parent['model'], map_location=rt['device']))
        before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        configure_stage(model, stage)
        dimension = model.asEmbedding.embedding_dim
        head = torch.nn.Linear(dimension if stage == 'as_context' else model.width, dimension).to(rt['device'])
        options = cfg['pretraining'][stage]
        optimizer = torch.optim.Adam([p for p in list(model.parameters()) + list(head.parameters()) if p.requires_grad],
                                     lr=options['lr'])
        paths = torch.from_numpy(np.load(str(RUN / ('as' + source) / 'pretraining_paths.npy'))).to(rt['device'])
        checkpoint, history = out / 'checkpoint.pt', []
        if checkpoint.exists():
            state = torch.load(str(checkpoint), map_location='cpu')
            require(state['identity'] == identity, 'Auxiliary checkpoint identity differs')
            model.load_state_dict(state['model']); head.load_state_dict(state['head'])
            optimizer.load_state_dict(state['optimizer']); history = state['history']
            restore_rng(torch, state); del state
        for epoch in range(len(history), options['epochs']):
            engine.emit(source, None, 'pretraining_' + stage, completed_epochs=epoch, total_epochs=options['epochs'])
            start, cpu = time.monotonic(), time.process_time()
            loss, count = stage_epoch(model, paths, stage, head, optimizer, options)
            torch.cuda.synchronize()
            history.append(dict(epoch=epoch, loss=loss, examples=count,
                                wall_seconds=time.monotonic() - start, cpu_seconds=time.process_time() - cpu))
            engine.atomic_save(torch, checkpoint, dict(identity=identity, model=model.state_dict(),
                head=head.state_dict(), optimizer=optimizer.state_dict(), history=history, **capture_rng(torch)))
            write_json(out / 'HISTORY.json', history)
            engine.emit(source, None, 'pretraining_' + stage, completed_epochs=len(history), last_epoch=history[-1])
        changed = changed_parameters(before, model.state_dict())
        verify_stage_changes(stage, changed)
        require(bool(model.asEmbedding.weight[0].eq(0).all()), 'Padding AS embedding changed')
        path = out / 'initialization.pt'
        engine.atomic_save(torch, path, {name: value.detach().cpu() for name, value in model.state_dict().items()})
        freeze(frozen_path, dict(identity=identity, model=str(path), model_sha256=sha256(path),
            initial_state_sha256=engine.tensor_hash(before), final_state_sha256=engine.tensor_hash(model.state_dict()),
            changed_parameters=changed, epochs=len(history), configuration=options, final_labels_read=False,
            validation_routes_read=False, auxiliary_heads_discarded=True,
            base_vectors_sha256=sha256(RUN / 'BASE_VECTORS.npy')))
        checkpoint.unlink(missing_ok=True)
        del model, head, optimizer, paths, before
        torch.cuda.empty_cache()
    return [frozen_path, path, out / 'HISTORY.json']


def train(engine, source, variant):
    from pathInference.research.gmpi_confidence_loss import replace_training_loss
    cfg, snap = read_json(CONFIG), read_json(RUN / 'SNAPSHOT.json')
    bound = snap['sources'][source]
    out = engine.directory(source, variant)
    stage = 'as_context' if variant == 'as_context' else 'path_context'
    initialization = read_json(RUN / ('as' + source) / ('pretrain_' + stage) / 'PRETRAIN_FROZEN.json')
    identity = dict(snapshot_sha256=sha256(RUN / 'SNAPSHOT.json'), source=source, variant=variant,
                    pretraining_sha256=initialization['model_sha256'])
    if (out / 'MODEL_FROZEN.json').exists():
        frozen = read_json(out / 'MODEL_FROZEN.json')
        require(frozen['identity'] == identity, 'Fine-tuning identity differs')
        verify_files({frozen['model']: frozen['model_sha256']}); return
    with forbid_formal_references():
        model, rt, arguments = cold_model(engine, source)
        torch, env = rt['torch'], rt['env']
        rng = capture_rng(torch)
        verify_files({initialization['model']: initialization['model_sha256']})
        model.load_state_dict(torch.load(initialization['model'], map_location=rt['device']))
        for parameter in model.parameters():
            parameter.requires_grad = True
        restore_rng(torch, rng)
        initial = engine.tensor_hash(model.state_dict())
        freeze(out / 'INITIALIZATION.json', dict(state_dict_sha256=initial, seed=cfg['seed'], identity=identity))
        optimizer = torch.optim.Adam(model.parameters(), lr=arguments['lr'])
        env.k = arguments['k']
        env.train_loader_params = dict(batch_size=arguments['batch_size'], shuffle=False,
                                      num_workers=0, drop_last=False, pin_memory=True)
        env.test_loader_params = dict(batch_size=arguments['test_batch_size'], shuffle=False,
                                     num_workers=0, drop_last=False, pin_memory=True)
        split = read_json(Path(bound['binding']['split']) / 'split_manifest.json')
        datasets = {name: rt['dataset'].IterableDatasetShuffle(str(Path(bound['binding']['split']) / (name + '.csv')),
            env, 20, arguments['k'], 60000, arguments['seed'], False) for name in ('train', 'validation')}
        training = replace_training_loss(rt['trainer'].train, cfg['label_temperature'])
        best, stale, history, best_state = -float('inf'), 0, [], None
        checkpoint = out / 'epoch_checkpoint.pt'
        if checkpoint.exists():
            resume = torch.load(str(checkpoint), map_location='cpu')
            require(resume['identity'] == identity and resume['initial_sha256'] == initial, 'Fine-tuning resume differs')
            model.load_state_dict(resume['model']); optimizer.load_state_dict(resume['optimizer'])
            best, stale, history, best_state = resume['best'], resume['stale'], resume['history'], resume['best_state']
            restore_rng(torch, resume); del resume
        for epoch in range(len(history), arguments['epochs']):
            if stale >= arguments['patience']:
                break
            start, cpu = time.monotonic(), time.process_time()
            loss, reward, count, _ = training(model, datasets['train'], optimizer, env, epoch, 'train')
            vl, vr, vc, _, _, _ = rt['trainer'].test(model, datasets['validation'], optimizer, env, epoch, 'surrogate_validation')
            require(count == split['row_counts']['train'] and vc == split['row_counts']['validation'], 'Epoch counts differ')
            score = vr / vc
            require(np.isfinite(score) and np.isfinite(loss), 'Nonfinite fine-tuning state')
            improved = score > best
            if improved:
                best, stale = score, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
            history.append(dict(epoch=epoch, training_jaccard=reward / count, validation_jaccard=score,
                training_loss=loss, validation_original_smooth_l1_loss=vl, checkpoint_improved=improved,
                wall_seconds=time.monotonic() - start, cpu_seconds=time.process_time() - cpu))
            engine.atomic_save(torch, checkpoint, dict(identity=identity, initial_sha256=initial,
                model=model.state_dict(), optimizer=optimizer.state_dict(), best_state=best_state,
                best=best, stale=stale, history=history, **capture_rng(torch)))
            write_json(out / 'TRAINING_HISTORY.json', history)
            engine.emit(source, variant, 'training', completed_epochs=len(history), best_validation_jaccard=best,
                        last_epoch=history[-1])
        require(history and best_state is not None, 'No fine-tuned state')
        path = out / 'model.pt'
        engine.atomic_save(torch, path, best_state)
        freeze(out / 'MODEL_FROZEN.json', dict(identity=identity, model=str(path), model_sha256=sha256(path),
            initial_sha256=initial, configuration=arguments, label_temperature=cfg['label_temperature'],
            selected_epoch=max(history, key=lambda row: row['validation_jaccard'])['epoch'],
            validation_jaccard=best, epochs=len(history), final_labels_read=False,
            admitted_library_path_set_sha256=bound['binding']['admitted_library_path_set_sha256']))
        checkpoint.unlink(missing_ok=True)
        del model, optimizer, datasets, best_state
        torch.cuda.empty_cache()
        engine.emit(source, variant, 'model_frozen', epochs=len(history), validation_jaccard=best)


def progress(error=None):
    cfg = read_json(CONFIG)
    active = read_json(RUN / 'ACTIVE.json') if (RUN / 'ACTIVE.json').exists() else {}
    complete = (RUN / 'COMPLETION_AUDIT.json').is_file() and read_json(RUN / 'COMPLETION_AUDIT.json')['status'] == 'PASS'
    lines = ['# 20260201：冻结候选后的 AS／路径表征预训练', '', '更新时间：' + utc_now(), '',
             '进程 PID：{}；单 GPU、2 CPU。'.format(os.getpid()),
             '冷启动结果复用已审计对照；候选路径、增强库、划分、网络结构和排序损失不变。',
             '仅在 admitted library 的训练查询完整路径上做自监督；排除验证查询。', '',
             '| AS | 语料 | AS 预训练 | 路径预训练 | 微调 | 预测 | 评测 |',
             '|---|---|---|---|---|---|---|']
    counts = {stage: 0 for stage in ('train', 'predict', 'evaluate')}
    for source in cfg['sources']:
        row = [source, '完成' if receipt(source, 'corpus').exists() else '等待']
        row += ['完成' if receipt(source, 'pretrain', stage).exists() else '等待'
                for stage in ('as_context', 'path_context')]
        for stage in counts:
            number = sum(receipt(source, stage, variant).exists() for variant in cfg['variants'])
            counts[stage] += number
            row.append(str(number) + '/2')
        lines.append('| ' + ' | '.join(row) + ' |')
    lines += ['', '当前活动：' + json.dumps(active, ensure_ascii=False), '',
              '全部完成并审计通过。' if complete else '全部 20 个模型冻结后预测；全部 20 份预测冻结后评测。',
              '该表每 15 秒刷新；时间戳是监控心跳，实际工作量见活动中的 epoch／已落盘查询数。']
    if error:
        lines += ['', '错误：' + error]
    temporary = RUN / 'PROGRESS.md.tmp'
    temporary.write_text('\n'.join(lines) + '\n'); temporary.replace(RUN / 'PROGRESS.md')
    write_json(RUN / 'STATUS.json', dict(updated_utc=utc_now(), pid=os.getpid(), counts=counts,
                                       active=active, complete=complete, error=error))


def finish():
    cfg = read_json(CONFIG)
    rows = []
    for source in cfg['sources']:
        cold = read_json(ROOT / cfg['cold_control'] / ('as' + source) / 'ordered_path/RESULTS.json')
        gmpi = read_json(ROOT / cfg['gmpi_control'] / ('as' + source) / 'query_cross_entropy/RESULTS.json')
        for variant in cfg['variants']:
            out = RUN / ('as' + source) / variant
            row = read_json(out / 'RESULTS.json')
            require(row['candidate_sha256'] == cold['candidate_sha256'] == gmpi['candidate_sha256']
                    and row['query_count'] == cold['query_count'] == gmpi['query_count']
                    and row['fixed_identity'] == cold['fixed_identity'] == gmpi['fixed_identity'],
                    'Comparison changed frozen candidate bytes/queries')
            row.update(cold_jaccard=cold['jaccard_accuracy'], cold_exact=cold['ordered_exact_accuracy'],
                gmpi_ce_jaccard=gmpi['jaccard_accuracy'], gmpi_ce_exact=gmpi['ordered_exact_accuracy'],
                cold_delta=row['jaccard_accuracy'] - cold['jaccard_accuracy'],
                gmpi_ce_delta=row['jaccard_accuracy'] - gmpi['jaccard_accuracy'],
                oracle_gap=row['oracle_jaccard'] - row['jaccard_accuracy'],
                timing={stage: read_json(out / ('TIMING_' + stage + '.json')) for stage in ('train', 'predict', 'evaluate')})
            rows.append(row)

    def aggregate(subset):
        count = sum(row['query_count'] for row in subset)
        if not count:
            return dict(queries=0, sources=0)
        fields = ('jaccard_accuracy', 'ordered_exact_accuracy', 'cold_jaccard', 'cold_exact',
                  'gmpi_ce_jaccard', 'gmpi_ce_exact', 'cold_delta', 'gmpi_ce_delta', 'oracle_jaccard', 'oracle_gap')
        return dict(queries=count, sources=len(subset),
                    **{key: math.fsum(row[key] * row['query_count'] for row in subset) / count for key in fields})

    summary = dict(variants={}, candidates_regenerated=False, final_labels_used_for_tuning=False,
                   historical_test_results_already_known=True)
    for variant in cfg['variants']:
        subset = [row for row in rows if row['variant'] == variant]
        summary['variants'][variant] = dict(micro=aggregate(subset),
            macro={key: math.fsum(row[key] for row in subset) / len(subset)
                   for key in ('jaccard_accuracy', 'ordered_exact_accuracy')},
            by_rho={name: aggregate([row for row in subset if (row['rho'] > 1) == high])
                    for name, high in [('high', True), ('low', False)]})
    write_json(RUN / 'RESULTS.json', rows); write_json(RUN / 'SUMMARY.json', summary)
    with (RUN / 'RESULTS.csv').open('w', newline='') as stream:
        fields = ['source', 'variant', 'rho', 'query_count', 'jaccard_accuracy', 'ordered_exact_accuracy',
                  'cold_jaccard', 'cold_exact', 'cold_delta', 'gmpi_ce_jaccard', 'gmpi_ce_delta', 'oracle_jaccard', 'oracle_gap']
        writer = csv.DictWriter(stream, fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    lines = ['# 20260201：表征自监督初始化实验', '',
             '| AS | 冷启动 | AS 预训练 | AS＋路径预训练 | GMPI CE |', '|---|---:|---:|---:|---:|']
    for source in cfg['sources']:
        a, b = [next(row for row in rows if row['source'] == source and row['variant'] == variant)
                for variant in cfg['variants']]
        lines.append('| {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |'.format(
            source, a['cold_jaccard'], a['jaccard_accuracy'], b['jaccard_accuracy'], a['gmpi_ce_jaccard']))
    lines += ['', '| 配置 | 查询数 | Jaccard | 完整匹配率 | Oracle 差距 |', '|---|---:|---:|---:|---:|']
    for variant in cfg['variants']:
        row = summary['variants'][variant]['micro']
        lines.append('| {} | {} | {:.6f} | {:.4%} | {:.6f} |'.format(variant, row['queries'],
                    row['jaccard_accuracy'], row['ordered_exact_accuracy'], row['oracle_gap']))
    lines += ['', '冷启动→AS：只增加 AS 自监督初始化；AS→AS＋路径：只增加路径编码器预训练，AS 嵌入在该阶段冻结。',
              '微调时两组均允许全部参数训练。候选、历史、网络容量、排序损失、优化器、种子、划分与检查点准则不变。',
              '辅助训练只用每源 admitted library 的训练查询部分，规模有限，不等同大规模跨源预训练。',
              '有向相邻跳识别不代表已知真实转发策略；表示的分布对齐不预测邻居路径份额。',
              '已有测试结果已知，本轮是探索性研究，不按测试分源选模或调参。',
              '逐源训练/预测实测时间见 RESULTS.json；预训练分阶段时间见 pretrain_*/TIMING_pretrain.json。',
              '最终结论以独立 COMPLETION_AUDIT.json 为准。']
    (RUN / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def dispatch():
    cfg = read_json(CONFIG)
    RUN.mkdir(parents=True, exist_ok=True); (RUN / 'tasks').mkdir(exist_ok=True)
    lock = (RUN / '.dispatch.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.sched_setaffinity(0, set(cfg['cpu_affinity']))
    engine = make_engine()
    stop = threading.Event()
    def watch():
        while not stop.wait(15):
            progress()
    watcher = threading.Thread(target=watch, daemon=True); watcher.start()
    error = None
    progress()
    try:
        if not checked(None, 'snapshot'):
            snapshot(); record_task(None, 'snapshot', None, [RUN / 'SNAPSHOT.json'])
        verify_files(read_json(RUN / 'SNAPSHOT.json')['files_sha256'])
        for source in cfg['sources']:
            if not checked(source, 'corpus'):
                record_task(source, 'corpus', None, prepare_corpus(source))
        for stage in ('as_context', 'path_context'):
            for source in cfg['sources']:
                if checked(source, 'pretrain', stage):
                    continue
                start, cpu = time.monotonic(), time.process_time()
                outputs = pretrain(engine, source, stage)
                timing = outputs[0].parent / 'TIMING_pretrain.json'
                write_json(timing, dict(wall_seconds=time.monotonic() - start, cpu_seconds=time.process_time() - cpu,
                    peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    measurement='stage wall/CPU; first source includes shared environment load; epoch history separate'))
                record_task(source, 'pretrain', stage, outputs + [timing])
        for stage in ('train', 'predict', 'evaluate'):
            for source in cfg['sources']:
                for variant in cfg['variants']:
                    if checked(source, stage, variant):
                        continue
                    engine.emit(source, variant, 'starting_' + stage)
                    if engine._RUNTIME is not None:
                        engine._RUNTIME['torch'].cuda.synchronize()
                    start, cpu = time.monotonic(), time.process_time()
                    if stage == 'train':
                        train(engine, source, variant)
                    else:
                        getattr(engine, stage)(source, variant)
                    if engine._RUNTIME is not None:
                        engine._RUNTIME['torch'].cuda.synchronize()
                    out = engine.directory(source, variant)
                    timing = out / ('TIMING_' + stage + '.json')
                    write_json(timing, dict(wall_seconds=time.monotonic() - start, cpu_seconds=time.process_time() - cpu,
                        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        measurement='stage clocks with CUDA synchronization'))
                    names = {'train': ['MODEL_FROZEN.json', 'model.pt', 'TRAINING_HISTORY.json', 'INITIALIZATION.json'],
                             'predict': ['PREDICTIONS_FROZEN.json', 'ranked_predictions.csv'],
                             'evaluate': ['RESULTS.json']}[stage]
                    record_task(source, stage, variant, [out / name for name in names] + [timing])
            if stage in ('train', 'predict'):
                name = 'ALL_MODELS_FROZEN.json' if stage == 'train' else 'ALL_PREDICTIONS_FROZEN.json'
                freeze(RUN / name, dict(snapshot_sha256=sha256(RUN / 'SNAPSHOT.json'),
                    receipts_sha256={str(receipt(source, stage, variant)): sha256(receipt(source, stage, variant))
                                    for source in cfg['sources'] for variant in cfg['variants']}))
        finish()
        import subprocess
        subprocess.run([__import__('sys').executable, '-B', str(ROOT / 'tools/validation/audit_representation_pretraining.py')],
                       cwd=str(ROOT), check=True)
    except BaseException as exc:
        error = str(exc); raise
    finally:
        stop.set(); watcher.join(); progress(error); lock.close()
