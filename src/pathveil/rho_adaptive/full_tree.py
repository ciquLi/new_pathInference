"""V2: the same full source-prefix tree and PG for training and queries.

Reuse verified source construction/augmentation caches. A completed source is
reused only when its actual generator, full tree, library, model configuration,
dated data and frozen numerical backend match; metrics never select reuse.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
from pathlib import Path

from pathveil.experiment import preparation,train,worker
from pathveil.experiment.contracts import freeze,freeze_bytes,require,verify_files
from pathveil.experiment.report import publish
from pathveil.experiment.schedule import Scheduler,downstream_tasks
from pathveil.prefix_tree.public_prefix_tree import PublicPrefixTopology
from pathveil.prefix_tree.public_tree_spec import content_sha
from pathveil.utils import project_root,read_json,sha256,utc_now,write_json
from .construction import build_spec
from .policy import POLICY as V1_POLICY


VERSION="rho_adaptive_full_tree_v2"
POLICY=copy.deepcopy(V1_POLICY)
POLICY.update(version=VERSION,query_tree="same full selected source tree for both rho groups",
              training_generators={"low":"Tree B + PG","high":"Tree B + PG"})


def validate_same_generator(prepared,tree,prefixes):
    require(prepared["training_generator"]=="Tree B + PG","Training/query generators differ")
    actual={tuple(row["prefix"]) for row in tree["primary_leaves"]}
    require(actual==set(map(tuple,prefixes)),"Query tree lost selected augmentation prefixes")
    require(tree["fallback_leaves"]==[],"Unregistered fallback tree changes the candidate protocol")


def numerical_tree(tree):
    return {name:tree[name] for name in ("primary_leaves","fallback_leaves","prefix_classification")}


def prepare(root,date,requested=None):
    registry_file=root/"configs/dates"/(date+".json");registry=read_json(registry_file)
    method_file=root/registry["workflow_config"];method=read_json(method_file)
    require(registry["workflow_id"]==method["workflow_id"]==VERSION,"Wrong registered full-tree method")
    require(method["selection_policy"]==POLICY,"Frozen full-tree policy changed")
    parent=Path(read_json(root/"runs"/date/registry["full_tree_cache_workflow"]/"current.json")["run_dir"])
    old=read_json(parent/"PROTOCOL.json")
    require(old["date"]==date and old["sources"]==list(map(str,registry["test_sources"])),"Construction cache date/cohort differs")
    verify_files(old["code_sha256"])
    data=read_json(parent/"data/data_manifest.json");upstream=read_json(parent/"UPSTREAM_BOUND.json")
    require(data["date"]==data["routing_date"]==upstream["date"]==date,"Cross-date data/model cache")
    require(Path(upstream["run"]).resolve()==(root/registry["upstream_cache"]).resolve(),"Unregistered upstream cache")
    verify_files({upstream["model"]:upstream["model_sha256"],upstream["bundle"]:upstream["bundle_sha256"]})
    require(old["upstream_config"]["root_selection"]["max_k"]==21,"Wrong cached root budget")
    for name in ("low_root_selection","low_recursion","high_root_selection","high_recursion","high_secondhop_formula","quality"):
        require(old["method"]["selection_policy"][name]==POLICY[name],"Construction cache selection policy differs: "+name)
    source_identities={}
    for source in old["sources"]:
        case_file=parent/"construction"/source/"CASE.json";case=read_json(case_file)
        require(case["source"]==source and case["date"]==date and not case["formal_labels_read"],"Source CASE scope differs")
        verify_files(case["files_sha256"])
        admission_file=parent/"augmentation"/("as"+source)/"ADMISSION.json";admission=read_json(admission_file)
        require(not admission["final_labels_read"],"Augmentation used final routes")
        verify_files(admission["inputs_sha256"])
        verify_files({admission["path"]:admission["sha256"],admission["json_path"]:admission["json_sha256"]})
        source_identities[source]=dict(case_sha256=sha256(case_file),admission_sha256=sha256(admission_file))
    identity=dict(version=VERSION,date=date,policy=POLICY,parent_run=str(parent),
        parent_protocol_sha256=sha256(parent/"PROTOCOL.json"),workflow_config_sha256=sha256(method_file),
        registry_sha256=sha256(registry_file),source_identities=source_identities,
        code_sha256={str(Path(__file__).resolve()):sha256(__file__)})
    tag=content_sha(identity)[:16];directory=root/"runs"/date/VERSION
    run=Path(requested).resolve() if requested else directory/("run_"+tag)
    require(root/"runs" in run.parents,"Run must stay in project outputs")
    run.mkdir(parents=True,exist_ok=True);freeze(run/"RUN_IDENTITY.json",identity)
    proto=copy.deepcopy(old);proto.update(version=VERSION,construction_cache_run=str(parent),research_identity=identity)
    proto["method"].update(workflow_id=VERSION,query_generator="full source-prefix tree + PG for training and queries",
        training_generators=POLICY["training_generators"],selection_policy=POLICY)
    proto["code_sha256"].update(identity["code_sha256"])
    proto["code_sha256"][str(method_file.resolve())]=sha256(method_file)
    freeze(run/"PROTOCOL.json",proto)
    for name in ("UPSTREAM_BOUND.json","data/data_manifest.json"):
        freeze(run/name,read_json(parent/name))
    freeze(run/"T-PG-B/METHOD_FROZEN.json",dict(version=VERSION,date=date,sources=proto["sources"],
        public_features=proto["public_features"],generator=proto["method"]["query_generator"],
        normal_history_only=True,training_objective=proto["method"]["training_objective"],protocol_sha256=sha256(run/"PROTOCOL.json")))
    freeze(run/"EVALUATION_POLICY_AMENDMENT.json",dict(selection_scope="single_source_generator",
        training_and_selection_unchanged=True,final_scores_used_for_further_training=False))
    topology=PublicPrefixTopology.from_manifest(run/"data/data_manifest.json")
    freeze(run/"query_trees/METHOD_FROZEN.json",dict(version=VERSION,date=date,sources=proto["sources"],
        configuration=POLICY,public_graph_identity_sha256=topology.graph_identity_sha256,
        public_inputs_sha256=topology.inputs_sha256,inputs_sha256={str(run/"PROTOCOL.json"):sha256(run/"PROTOCOL.json")},
        spec_code_sha256=sha256(preparation.specs.__file__)))
    trees={};plans={}
    for source in proto["sources"]:
        src=parent/"augmentation"/("as"+source);dst=run/"augmentation"/("as"+source);dst.mkdir(parents=True,exist_ok=True)
        admission=read_json(src/"ADMISSION.json")
        for name in ("paths.csv","paths.json","PATH_RECORDS.json"):
            freeze_bytes(dst/name,(src/name).read_bytes())
        admission.update(path=str(dst/"paths.csv"),json_path=str(dst/"paths.json"),protocol_sha256=sha256(run/"PROTOCOL.json"))
        admission["inputs_sha256"][str(src/"ADMISSION.json")]=sha256(src/"ADMISSION.json")
        freeze(dst/"ADMISSION.json",admission)
        plan_file=parent/"construction"/source/"plan.json";plan=read_json(plan_file);plans[source]=plan
        tree=build_spec(topology,source,plan["prefixes"],{str(plan_file):sha256(plan_file),str(dst/"ADMISSION.json"):sha256(dst/"ADMISSION.json"),str(Path(__file__).resolve()):sha256(__file__)})
        tree.update(version=VERSION,configuration=POLICY,configuration_sha256=content_sha(POLICY))
        trees[source]=tree
        # The original preparer verifies this source-only TRAIN provenance binding
        # before accepting the audit; identical corpora need not be rescanned.
        audit=parent/"T-PG-B/pg_inputs"/("as"+source)/"control"/("derived_source_"+source+"_audit.json")
        if audit.exists():
            destination=run/"T-PG-B/pg_inputs"/("as"+source)/"control"/audit.name
            destination.parent.mkdir(parents=True,exist_ok=True);freeze_bytes(destination,audit.read_bytes())
    original=preparation.specs.build_spec
    preparation.specs.build_spec=lambda graph,source,prefixes,input_bindings=None:copy.deepcopy(trees[str(source)])
    release=read_json(Path(upstream["run"])/"upstream/RELEASE.json")
    graph={key:release[key] for key in ("selected_neighbors_csv","recursive_anchors_csv")}
    try:
        for source in proto["sources"]:
            preparation.prepare_source(run,source,Path(proto["baseline_run"])/"data/downstream_view/data_manifest.json",topology,graph)
            ready=read_json(run/"T-PG-B/pg_inputs"/("as"+source)/"PREPARED.json")
            tree=read_json(run/"query_trees"/(source+".json"));validate_same_generator(ready,tree,plans[source]["prefixes"])
            admission=read_json(run/"augmentation"/("as"+source)/"ADMISSION.json")
            require(ready["admitted_library_path_set_sha256"][source]==admission["path_set_sha256"],"Training/query library mismatch")
            print(json.dumps(dict(date=date,stage="full_tree_preparation",source=source,tree_prefixes=len(set(map(tuple,plans[source]["prefixes"]))))),flush=True)
    finally:preparation.specs.build_spec=original
    freeze(run/"PLAN.json",dict(version=VERSION,date=date,policy=POLICY,parent_cache=str(parent),
        augmentation_rebuilt=False,new_mass_prediction=False,training_query_generator_equal=True))
    write_json(run/"CONSTRUCTION_STATUS.json",dict(status="COMPLETED",completed=proto["sources"],
        total=len(proto["sources"]),active={},failures={},verified_augmentation_cache_reused=True,updated_utc=utc_now()))
    write_json(directory/"current.json",dict(date=date,run_dir=str(run),version=VERSION,updated_utc=utc_now()))
    print(json.dumps(dict(date=date,run_dir=str(run),status="PREPARED")),flush=True)
    return run


def reuse_completed_sources(run):
    proto=read_json(run/"PROTOCOL.json");parent=Path(proto["construction_cache_run"]);old=read_json(parent/"PROTOCOL.json")
    require(proto["method"]["gmpi"]==old["method"]["gmpi"] and proto["method"]["training_objective"]==old["method"]["training_objective"],"GMPI protocol changed")
    require(sha256(run/"UPSTREAM_BOUND.json")==sha256(parent/"UPSTREAM_BOUND.json"),"Upstream model identity changed")
    imported=[]
    for source in proto["sources"]:
        result_file=parent/"T-PG-B/independent_evaluation"/("source_"+source+".json")
        if not result_file.exists():continue
        before=read_json(parent/"T-PG-B/pg_inputs"/("as"+source)/"PREPARED.json")
        if before["training_generator"]!="Tree B + PG":continue
        after=read_json(run/"T-PG-B/pg_inputs"/("as"+source)/"PREPARED.json")
        old_tree=read_json(parent/"query_trees"/(source+".json"));new_tree=read_json(run/"query_trees"/(source+".json"))
        if numerical_tree(old_tree)!=numerical_tree(new_tree):continue
        require(before["configuration"]==after["configuration"] and
                before["admitted_library_path_set_sha256"]==after["admitted_library_path_set_sha256"] and
                before["test_identity"]==after["test_identity"],"Completed source inputs differ")
        require(all(after["code_sha256"].get(path)==digest for path,digest in before["code_sha256"].items()),"Numerical backend changed")
        worker.verify(parent,"rescore",source)
        selected=train.verify_selection(source,parent/"T-PG-B")
        row=read_json(result_file)
        require(row["checkpoint_sha256"]==selected["chosen_model_sha256"],"Frozen checkpoint identity differs")
        row["equivalent_completed_source_reuse"]=dict(parent_run=str(parent),result_sha256=sha256(result_file),
            verified_same_full_tree=True,verified_same_library=True,verified_same_model_protocol=True,
            metric_used_to_select_reuse=False)
        freeze(run/"T-PG-B/independent_evaluation"/result_file.name,row);imported.append(source)
    freeze(run/"REUSED_COMPLETED_SOURCES.json",dict(sources=imported,parent_run=str(parent),selection="exact protocol equivalence only; never metric value"))
    return imported


def run_native(run,device):
    proto=read_json(run/"PROTOCOL.json");verify_files(proto["code_sha256"])
    with (run/".run.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        imported=reuse_completed_sources(run);publish(run)
        remaining=[s for s in proto["sources"] if s not in imported]
        tasks=downstream_tasks(remaining)
        if tasks:Scheduler(run,tasks,proto["method"]["resources"],device).execute()
        result=publish(run);require(result["status"]=="COMPLETED","Not all sources completed")
        write_json(run/"STATUS.json",dict(date=proto["date"],status="COMPLETED",complete=6*len(proto["sources"]),
            total=6*len(proto["sources"]),numerical_tasks=len(tasks),reused_sources=imported,active=[],failures={},updated_utc=utc_now()))


def execute(args,registry):
    root=project_root()
    if args.migration_replay:raise ValueError("Use the explicit original March replay")
    if args.check_only:
        from pathveil.data.preflight import check_date
        report=check_date(args.date,deep=args.deep_check);print(json.dumps(report,indent=2));return 0 if report["raw_ready"] else 2
    if args.plan_only:
        print(json.dumps(dict(version=VERSION,date=args.date,policy=POLICY,new_training_started=False),indent=2));return 0
    require(not args.augmentation_only and not args.upstream_only,"V2 reuses verified construction; use --prepare-only or full run")
    if args.method_config:
        path=Path(args.method_config);path=path if path.is_absolute() else root/path
        require(path.resolve()==(root/registry["workflow_config"]).resolve(),"Use the registered frozen full-tree configuration")
    requested=args.run_dir
    if args.fresh_run:
        require(requested is None,"Use either --fresh-run or --run-dir")
        requested=root/"runs"/args.date/VERSION/("fresh_"+utc_now().replace(":","").replace("-","").replace(".","_"))
    run=prepare(root,args.date,requested)
    if args.prepare_only:return 0
    run_native(run,args.device);return 0
