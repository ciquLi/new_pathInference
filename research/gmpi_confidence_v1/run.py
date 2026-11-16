#!/usr/bin/env python3
"""One GPU process reuses the frozen GMPI environment for all paired controls."""
from __future__ import annotations

from datetime import datetime
import csv
import fcntl
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable]='2'
from pathInference.research import gmpi_confidence as experiment
from pathInference.experiment.contracts import freeze, require, verify_files
from pathInference.utils import read_json,sha256,write_json,utc_now


def code_identity():
    files=[Path(__file__).resolve(),Path(experiment.__file__).resolve(),
           ROOT/'src/pathInference/research/gmpi_confidence_loss.py']
    return {str(file):sha256(file) for file in files}


def receipt(source,variant,stage):
    return experiment.RUN/'tasks'/('_'.join(x for x in (source,variant,stage) if x)+'.json')


def checked(source,variant,stage):
    path=receipt(source,variant,stage)
    if not path.exists(): return False
    task=read_json(path)
    require(task['code_sha256']==code_identity() and task['config_sha256']==sha256(experiment.CONFIG),
            'GMPI task code/config changed')
    verify_files(task['outputs_sha256'])
    return True


def report(error=None):
    run,cfg=experiment.RUN,experiment.config()
    active=read_json(run/'ACTIVE.json') if (run/'ACTIVE.json').exists() else {}
    lines=['# 20260201：GMPI 仅改变训练损失的 GPU 对照','',
        '更新时间：'+datetime.now().astimezone().isoformat(timespec='seconds'),'',
        '全部十源；一个 GPU 工作进程，2 CPU 线程。原 Smooth L1 与查询内交叉熵均从冷初始化开始。',
        '共用同一冻结候选、历史路径、嵌入、训练/验证划分；配对初始权重必须一致。',
        '全部模型冻结后预测；全部预测冻结后才打开最终测试标签。','',
        '| AS | 损失 | 训练 | 预测 | 评测 | 实际状态 |',
        '|---|---|---|---|---|---|']
    rows=[]
    for source in cfg['sources']:
        for variant in cfg['variants']:
            directory=run/('as'+source)/variant
            state=read_json(directory/'STATUS.json') if (directory/'STATUS.json').exists() else {}
            cells=[]
            for stage in ('train','predict','evaluate'):
                done=receipt(source,variant,stage).exists()
                running=active.get('source')==source and active.get('variant')==variant
                cells.append('完成' if done else '运行中' if running and (
                    (stage=='train' and state.get('stage') in ('training','model_frozen')) or
                    (stage=='predict' and state.get('stage')=='predicting')) else '等待')
            status=state.get('stage','等待')
            if 'completed_epochs' in state: status+='；已完成 '+str(state['completed_epochs'])+' epoch'
            if 'committed_queries' in state: status+='；已保存 '+str(state['committed_queries'])+' 查询'
            lines.append('| '+' | '.join([source,variant,*cells,status])+' |')
            if (directory/'RESULTS.json').exists(): rows.append(read_json(directory/'RESULTS.json'))
    if error: lines+=['','已停止，错误：'+error]
    lines+=['','当前工作：'+json.dumps(active,ensure_ascii=False),
            '','断点：每 epoch 的模型/优化器/RNG；每批预测的 SQLite 事务。前面的路径不重新生成。']
    temporary=run/'PROGRESS.md.tmp'; temporary.write_text('\n'.join(lines)+'\n'); temporary.replace(run/'PROGRESS.md')
    write_json(run/'STATUS.json',dict(updated_utc=utc_now(),error=error,completed_controls=len(rows),
        expected_controls=len(cfg['sources'])*len(cfg['variants']),active=active))


def finish():
    run,cfg=experiment.RUN,experiment.config()
    rows=[read_json(run/('as'+s)/v/'RESULTS.json') for s in cfg['sources'] for v in cfg['variants']]
    write_json(run/'RESULTS.json',rows)
    fields=['source','variant','rho','query_count','jaccard_accuracy','ordered_exact_accuracy',
            'oracle_jaccard','historical_gmpi_jaccard']
    with (run/'RESULTS.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    summaries={}
    for variant in cfg['variants']:
        subset=[r for r in rows if r['variant']==variant]
        total=sum(r['query_count'] for r in subset)
        summaries[variant]=dict(queries=total,**{key:sum(r[key]*r['query_count'] for r in subset)/total
            for key in ('jaccard_accuracy','ordered_exact_accuracy','oracle_jaccard')})
    write_json(run/'SUMMARY.json',dict(variants=summaries,changed_factor='training loss only',
        historical_test_results_already_known=True,final_labels_used_for_tuning=False))
    lines=['# 20260201：配对 GMPI 损失对照','',
           '| AS | Smooth L1 Jaccard | 交叉熵 Jaccard | 差值 |','|---|---|---|---|']
    for source in cfg['sources']:
        a,b=[next(r for r in rows if r['source']==source and r['variant']==v) for v in cfg['variants']]
        require(a['initial_sha256']==b['initial_sha256'] and a['candidate_sha256']==b['candidate_sha256'],
                'Paired control differs in initialization/candidates')
        lines.append(f"| {source} | {a['jaccard_accuracy']:.6f} | {b['jaccard_accuracy']:.6f} | "
                     f"{b['jaccard_accuracy']-a['jaccard_accuracy']:+.6f} |")
    lines+=['','全部查询加权：'+json.dumps(summaries,ensure_ascii=False),
            '','本轮只改变损失。原始测试结果此前已知，所以本轮属于探索性对照，不能视为新盲测。']
    (run/'REPORT.md').write_text('\n'.join(lines)+'\n')


def dispatch():
    run,cfg=experiment.RUN,experiment.config()
    run.mkdir(parents=True,exist_ok=True);(run/'tasks').mkdir(exist_ok=True)
    lock=(run/'.dispatch.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.sched_setaffinity(0,{6,7})
    for key,folder in (('MPLCONFIGDIR','matplotlib'),('CUDA_CACHE_PATH','cuda'),('TORCH_HOME','torch')):
        path=run/'runtime'/folder;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    stop=threading.Event()
    def monitor():
        while not stop.wait(10): report()
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    error=None
    try:
        if not checked(None,None,'snapshot'):
            experiment.snapshot()
            freeze(receipt(None,None,'snapshot'),dict(code_sha256=code_identity(),
                config_sha256=sha256(experiment.CONFIG),outputs_sha256={str(run/'SNAPSHOT.json'):sha256(run/'SNAPSHOT.json')}))
        verify_files(read_json(run/'SNAPSHOT.json')['files_sha256'])
        for stage in ('train','predict','evaluate'):
            for source in cfg['sources']:
                for variant in cfg['variants']:
                    if checked(source,variant,stage): continue
                    experiment.emit(source,variant,'starting_'+stage)
                    start,cpu=time.monotonic(),time.process_time()
                    getattr(experiment,stage)(source,variant)
                    directory=experiment.directory(source,variant)
                    write_json(directory/('TIMING_'+stage+'.json'),dict(wall_seconds=time.monotonic()-start,
                        cpu_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        measurement='stage clocks; GPU operations synchronized by existing trainer scalar reads',
                        runtime_reused=experiment._RUNTIME is not None))
                    names={'train':['MODEL_FROZEN.json','model.pt','TRAINING_HISTORY.json'],
                           'predict':['PREDICTIONS_FROZEN.json','ranked_predictions.csv'],
                           'evaluate':['RESULTS.json']}[stage]
                    freeze(receipt(source,variant,stage),dict(code_sha256=code_identity(),
                        config_sha256=sha256(experiment.CONFIG),source=source,variant=variant,stage=stage,
                        completed_utc=utc_now(),outputs_sha256={str(directory/name):sha256(directory/name) for name in names}))
            if stage in ('train','predict'):
                name='ALL_MODELS_FROZEN.json' if stage=='train' else 'ALL_PREDICTIONS_FROZEN.json'
                freeze(run/name,dict(snapshot_sha256=sha256(run/'SNAPSHOT.json'),
                    receipts_sha256={str(receipt(s,v,stage)):sha256(receipt(s,v,stage))
                                     for s in cfg['sources'] for v in cfg['variants']}))
        finish()
    except BaseException as exc:
        error=str(exc); raise
    finally:
        stop.set();watcher.join();report(error);lock.close()


if __name__=='__main__': dispatch()
