"""Frozen experiment registration and shared-resource streaming execution."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from pathveil.experiment.contracts import freeze,require,verify_files
from pathveil.downstream import common
from pathveil.experiment import preparation
from pathveil.experiment.schedule import Scheduler,Task
from pathveil.utils import read_json,sha256,utc_now,validate_date,write_json

from .policy import POLICY, runtime_policy


def write_progress(run, stage, **details):
    run = Path(run)
    date = read_json(run / "PROTOCOL.json")["date"]
    lines = [
        f"# {date}：PATS 源 rho 前缀树",
        "",
        f"当前阶段：{stage}",
        f"更新时间（UTC）：{utc_now()}",
    ]
    for key, value in details.items():
        lines.append(f"{key}：{value}")
    (run / "PROGRESS.md").write_text("\n".join(lines) + "\n")


def pats_training_tasks(sources):
    """Tasks after the one-process batch path-generation barrier."""
    tasks = []
    for source in map(str, sources):
        train_key = "train:" + source
        annotate_key = "annotate:" + source
        evaluate_key = "evaluate:" + source
        tasks.extend(
            (
                Task("train", source, ()),
                Task("annotate", source, (train_key,)),
                Task("evaluate", source, (annotate_key,)),
                Task("rescore", source, (evaluate_key,)),
            )
        )
    return tasks


def load_method(root,registry,override=None):
    path=Path(override or registry["workflow_config"])
    path=path if path.is_absolute() else root/path
    method=read_json(path)
    require(method["workflow_id"]==POLICY["version"],"Wrong registered PATS workflow")
    require(method["training_generators"]=={"low":"Tree B + PATS","high":"Tree B + PATS"},"PATS requires one generator for both rho groups")
    return method,path


def validate_assets(date,registry,proto,data,upstream,bundle,model):
    require(registry["mode"]=="bgp" and registry["date"]==registry["routing_date"]==date,
            "Adaptive workflow needs an explicitly registered BGP date")
    for name, day in registry["auxiliary_dates"].items():
        require(validate_date(day) <= date, "Future public snapshot: " + name)
    require(proto["date"]==data["date"]==data["routing_date"]==upstream["date"]==bundle["date"]==bundle["model_fit_date"]==model["date"]==model["model_fit_date"]==date,
            "Cross-date baseline/data/model cache rejected")
    require(not bundle["formal_labels_read"] and not model["formal_labels_read"],"Dated models fitted final labels")
    formal=set(map(str,registry["test_sources"]))
    require(not formal & set(map(str,bundle["training_sources"])) and
            not formal & set(map(str,model["training_sources"])),"Formal source fitted upstream model")
    require(list(map(str,registry["test_sources"]))==proto["sources"],"Registered cohort differs from baseline")


def prepare(root,date,requested=None,override=None):
    registry_file=root/"configs/dates"/(date+".json");registry=read_json(registry_file)
    method,method_file=load_method(root,registry,override)
    policy=runtime_policy(method)
    baseline_workflow=registry.get("baseline_workflow","rho_prefix_v1")
    baseline=Path(read_json(root/"runs"/date/baseline_workflow/"current.json")["run_dir"])
    baseline_proto=read_json(baseline/"PROTOCOL.json")
    data=read_json(baseline/"data/data_manifest.json")
    upstream=read_json(baseline/"UPSTREAM_BOUND.json")
    require(Path(upstream["run"]).resolve()==(root/registry["upstream_cache"]).resolve(),"Upstream cache is not the explicitly registered date cache")
    verify_files({upstream["model"]:upstream["model_sha256"],upstream["bundle"]:upstream["bundle_sha256"]})
    bundle=read_json(upstream["bundle"])
    model=read_json(Path(upstream["model"]).with_name("MODEL.json"))
    validate_assets(date,registry,baseline_proto,data,upstream,bundle,model)
    sources=list(map(str,registry["test_sources"]))
    code_paths=list(Path(__file__).parent.glob("*.py")) + [
        root/"src/pathveil/prefix_tree/stage.py",
        root/"src/pathveil/prefix_tree/public_tree_pathgen.py",
        root/"src/pathveil/prefix_tree/public_tree_observed_pathgen.py",
        root/"src/pathveil/experiment/compile_queries.py",
        root/"src/pathveil/downstream/common.py",
        root/"src/pathveil/core/mainPathGen.py",
        root/"src/pathveil/core/getData.py",
        root/"src/pathveil/experiment/worker.py",
    ]
    code={str(p.resolve()):sha256(p) for p in code_paths}
    statistic=root/"runs/analysis/neighbor_path_mass_20261028/REPORT.md"
    require(statistic.is_file(),"User-designated fixed TRAIN-statistic provenance missing")
    identity=dict(version=POLICY["version"],date=date,policy=policy,code_sha256=code,
        workflow_config_sha256=sha256(method_file),
        registry_sha256=sha256(registry_file),baseline_protocol_sha256=sha256(baseline/"PROTOCOL.json"),
        baseline_upstream_sha256=sha256(baseline/"UPSTREAM_BOUND.json"),
        user_designated_statistic=dict(path=str(statistic),sha256=sha256(statistic),
            value=21,scope="0201 original TRAIN aggregate K80 P90; user-specified hyperparameter",
            final_source_specific_counts_used=False))
    tag=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    parent=root/"runs"/date/POLICY["version"]
    run=Path(requested).resolve() if requested else parent/("run_"+tag)
    require(root/"runs" in run.parents,"Run must stay inside project runs/")
    run.mkdir(parents=True,exist_ok=True);freeze(run/"RUN_IDENTITY.json",identity)
    proto=copy.deepcopy(baseline_proto)
    proto.update(version=POLICY["version"],baseline_run=str(baseline),research_identity=identity,sources=sources)
    proto["method"]["workflow_id"]=POLICY["version"]
    proto["method"]["query_generator"]="prefix-anchored tail stitching (PATS) for training and queries"
    proto["method"]["training_generators"]={"low":"Tree B + PATS","high":"Tree B + PATS"}
    proto["method"]["selection_policy"]=policy
    proto["method"]["augmentation"].update(
        mode="direct_source_tree_for_both_groups",source_threshold=policy["quality"]["source_threshold"],
        minimum_paths_for_both_groups=policy["quality"]["minimum_paths"],neighbor_threshold=None,
        neighbor_first_splicing=False,maximum_spliced_prefix_AS_count=3)
    proto["method"]["resources"].update(cpu_workers=6)
    proto["upstream_config"]=copy.deepcopy(proto["upstream_config"])
    proto["upstream_config"]["root_selection"]["max_k"]=policy["low_root_limit"]
    proto["code_sha256"].update(code)
    proto["code_sha256"][str(method_file.resolve())]=sha256(method_file)
    proto["frozen_method_file"]=str(method_file.resolve())
    freeze(run/"PROTOCOL.json",proto)
    for name in ("UPSTREAM_BOUND.json","data/data_manifest.json"):
        freeze(run/name,read_json(baseline/name))
    freeze(run/"T-PG-B/METHOD_FROZEN.json",dict(version=POLICY["version"],date=date,sources=sources,
        generator=proto["method"]["query_generator"],public_features=proto["public_features"],
        normal_history_only=True,training_objective=proto["method"]["training_objective"],protocol_sha256=sha256(run/"PROTOCOL.json")))
    freeze(run/"EVALUATION_POLICY_AMENDMENT.json",dict(selection_scope="single_source_generator",
        training_and_selection_unchanged=True,final_scores_used_for_further_training=False))
    manifest=read_json(run/"data/data_manifest.json")
    from pathveil.prefix_tree.public_prefix_tree import PublicPrefixTopology
    from pathveil.prefix_tree import public_tree_spec
    topology=PublicPrefixTopology.from_manifest(run/"data/data_manifest.json")
    freeze(run/"query_trees/METHOD_FROZEN.json",dict(version=POLICY["version"],date=date,sources=sources,
        configuration=POLICY,public_graph_identity_sha256=topology.graph_identity_sha256,
        public_inputs_sha256=topology.inputs_sha256,inputs_sha256={str(run/"PROTOCOL.json"):sha256(run/"PROTOCOL.json")},
        spec_code_sha256=sha256(public_tree_spec.__file__)))
    summary=[]
    for source in sources:
        rho=proto["public_features"][source]["rho"]
        require(rho is not None,"Unknown source structure")
        summary.append(dict(source=source,rho=rho,root_limit=6 if rho>1 else policy["low_root_limit"],
                            mandatory_recursion=rho>1,secondhop_limit=policy["recursion_topk"],
                            recursion_gate="forced" if rho>1 else "valid_tail_count"))
    freeze(run/"PLAN.json",dict(date=date,sources=summary,policy=policy,upstream_models_retrained=False,
        final_references_used_for_selection=False,preflight_expected="focused tests, full suite, March migration replay"))
    write_json(parent/"current.json",dict(run_dir=str(run),date=date,version=POLICY["version"],updated_utc=utc_now()))
    if not (run/"PROGRESS.md").exists():
        (run/"PROGRESS.md").write_text("# "+date+"：PATS 源 rho 前缀树\n\n已冻结方法及输入；等待启动结构构造和下游任务。\n")
    print(json.dumps(dict(run_dir=str(run),sources=summary)),flush=True)
    return run


def memory_coordinator(root):
    scripts=str(root/"scripts")
    if scripts not in sys.path:sys.path.insert(0,scripts)
    from build_training_cases import WorkerMemoryCoordinator
    return WorkerMemoryCoordinator(root/"runs")


def build_all(run):
    run=Path(run);proto=read_json(run/"PROTOCOL.json");root=Path(os.environ["PATHVEIL_PROJECT_ROOT"])
    memory=memory_coordinator(root);active={};done=set();failures={};sources=proto["sources"]
    code_root=Path(__file__).resolve().parents[2]
    try:
        while len(done)<len(sources):
            for source,(process,stream) in list(active.items()):
                result=process.poll()
                if result is None:continue
                stream.close();memory.release(process);del active[source]
                ready=run/"construction"/source/"SOURCE_READY.json"
                if result or not ready.exists():failures[source]=dict(exit_code=result,log=str(run/"logs"/("construct_"+source+".log")))
                else:
                    value=read_json(ready);verify_files(value["outputs_sha256"]);done.add(source)
            if not failures:
                for source in sources:
                    if source in done or source in active:continue
                    ready=run/"construction"/source/"SOURCE_READY.json"
                    if ready.exists():
                        value=read_json(ready);verify_files(value["outputs_sha256"])
                        require(value["protocol_sha256"]==sha256(run/"PROTOCOL.json"),"Cached source identity differs")
                        done.add(source);continue
                    if len(active)>=4:break
                    log=run/"logs"/("construct_"+source+".log");log.parent.mkdir(exist_ok=True)
                    stream=log.open("ab")
                    command=[sys.executable,"-B","-m","pathveil.pats",proto["date"],"--stage","source","--run-dir",str(run),"--source",source]
                    env=dict(os.environ,PYTHONPATH=str(code_root),OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
                    process=memory.spawn(lambda:subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,
                        stdout=stream,stderr=subprocess.STDOUT,start_new_session=True),6*1024**3,10*1024**3)
                    if process is None:stream.close();break
                    active[source]=(process,stream)
            write_json(run/"CONSTRUCTION_STATUS.json",dict(status="FAILED" if failures else "RUNNING",
                completed=sorted(done,key=int),total=len(sources),failures=failures,
                active={s:p.pid for s,(p,_) in active.items()},updated_utc=utc_now()))
            write_progress(run,"source_tree_construction",
                           completed=f"{len(done)}/{len(sources)}",
                           active=", ".join(sorted(active)) or "none")
            if failures:raise RuntimeError("Source construction failed: "+json.dumps(failures))
            if len(done)<len(sources):time.sleep(3)
        write_json(run/"CONSTRUCTION_STATUS.json",dict(status="COMPLETED",completed=sources,total=len(sources),failures={},active={},updated_utc=utc_now()))
        write_progress(run,"source_tree_completed",completed=f"{len(sources)}/{len(sources)}")
    finally:
        for process,stream in active.values():
            if process.poll() is None:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
            stream.close();memory.release(process)


class StreamingScheduler(Scheduler):
    def start(self,task):
        status=self.run/"CONSTRUCTION_STATUS.json"
        if status.exists() and read_json(status).get("failures"):
            raise RuntimeError("Source construction failed; preserve completed source results")
        if not (self.run/"construction"/task.source/"SOURCE_READY.json").exists():
            return False
        return super().start(task)


def execute(run,device):
    run=Path(run);proto=read_json(run/"PROTOCOL.json")
    resume_root=os.environ.get("PATHVEIL_RESUME_RUN")
    resume_allowed=False
    if resume_root:
        root=Path(resume_root).resolve()
        resume_allowed=common.inside(run,root) and (root/"RESUME_PATCH.json").is_file()
    if not resume_allowed:
        verify_files(proto["code_sha256"])
    lock=(run/".run.lock").open("a+");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    logs=run/"logs";logs.mkdir(exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[2]),OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
    command=[sys.executable,"-B","-m","pathveil.pats",proto["date"],"--stage","build_all","--run-dir",str(run)]
    with (logs/"construction_dispatch.log").open("ab") as stream:
        builder=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        try:
            require(builder.wait(timeout=None)==0,"Construction dispatcher failed")
            write_progress(run,"batch_path_generation_start",sources=len(proto["sources"]),
                           train_workers=proto["method"]["selection_policy"].get("pathgen_train_threads",2))
            batch_command=[sys.executable,"-B","-m","pathveil.pats.batch_pathgen",
                           "--run-dir",str(run)]
            batch_env=dict(os.environ,
                           PATHVEIL_PROJECT_ROOT=str(Path(__file__).resolve().parents[3]),
                           PYTHONPATH=str(Path(__file__).resolve().parents[2]),
                           PATHVEIL_STAGE_GUARD_ROOT=str(run.resolve()))
            with (logs/"pats_batch_pathgen.log").open("ab") as batch_log:
                batch_result=subprocess.run(batch_command,env=batch_env,stdin=subprocess.DEVNULL,
                                            stdout=batch_log,stderr=subprocess.STDOUT)
            require(batch_result.returncode==0,"PATS batch path generation failed")
            write_progress(run,"batch_path_generation_completed",sources=len(proto["sources"]))
            write_progress(run,"gmpi_training_and_evaluation",sources=len(proto["sources"]))
            scheduler=StreamingScheduler(run,pats_training_tasks(proto["sources"]),proto["method"]["resources"],device)
            scheduler.execute()
            from pathveil.experiment.report import publish
            results=publish(run);require(results["status"]=="COMPLETED","Incomplete final source results")
            state=read_json(run/"STATUS.json");state.update(status="COMPLETED",updated_utc=utc_now())
            write_json(run/"STATUS.json",state)
            write_progress(run,"completed",sources=len(proto["sources"]))
        except BaseException:
            if builder.poll() is None:
                os.kill(builder.pid,signal.SIGINT)
                try:builder.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(builder.pid,signal.SIGTERM);builder.wait()
            state=read_json(run/"STATUS.json") if (run/"STATUS.json").exists() else {}
            state.update(status="FAILED",error=traceback.format_exc(),updated_utc=utc_now())
            write_json(run/"STATUS.json",state)
            write_progress(run,"failed",error=type(sys.exc_info()[1]).__name__)
            raise
        finally:lock.close()
