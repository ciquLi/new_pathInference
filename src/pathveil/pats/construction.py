"""Build one source tree and feed it to both augmentation and query generation."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import os
from pathlib import Path
import pickle

import lightgbm as lgb
import numpy as np

from pathveil.experiment.contracts import (freeze,freeze_bytes,forbid_formal_references,
    normalize_pool,path_bytes,require,signature,verify_files)
from pathveil.prefix_tree import public_tree_spec as specs
from pathveil.prefix_tree.public_prefix_tree import commercial_phases
from pathveil.prefix_tree.registered_topology import RegisteredPublicPrefixTopology as PublicPrefixTopology
from pathveil.upstream.destination import DestinationFeatureStore
from pathveil.upstream.neighbors import NeighborRuntime
from pathveil.upstream import path_pipeline
from pathveil.upstream.implementation.frontier_adapter import SafeFrontierAdapter
from pathveil.upstream.implementation.frontier_policy import FrontierPolicy
from pathveil.utils import read_json,sha256,utc_now,write_json

from .policy import POLICY,RECURSION_TOPK,LOW_TAIL_MIN,select_root,mandatory_plan
from .admission import quality_selection


class Runtime(NeighborRuntime):
    def __init__(self,bundle,data,config,topology):
        self.topology=topology
        super().__init__(bundle,data,config,threads=1)

    def _infer_root(self,source):
        original=super()._infer_root(source)
        shape=self.topology.source_features(str(source))
        b1=shape["public_b1"]
        rho=shape["public_l2_distinct_asns"]/(b1*b1) if b1 else None
        return select_root(original,rho)


class Adapter(SafeFrontierAdapter):
    def __init__(self,source,prediction,index,relations,predictor,policy):
        runtime=predictor.predictor
        shape=runtime.topology.source_features(str(source));b1=shape["public_b1"]
        require(b1>0,"No public source shape")
        self.rho=shape["public_l2_distinct_asns"]/(b1*b1)
        self.topology=runtime.topology
        policy=FrontierPolicy(valid_tail_count_min=LOW_TAIL_MIN,max_prefix_AS_count=3,
                              next_hop_topk=RECURSION_TOPK)
        super().__init__(source,prediction,index,relations,predictor,policy)

    def plan(self):
        if self.rho<=1:
            return super().plan()
        # High-rho sources force recursion; low-rho sources retain the original
        # tail-capacity gate. Both branches use the same fixed recursion width.
        result=mandatory_plan(self.source,self.prediction["selected_neighbors"],
                              self.capacity,self.predict_next)
        result["prefix_edge_provenance"]={"->".join(prefix):proof for prefix,proof in self.prefix_proofs.items()}
        result["frontier_prediction_queries"]={"->".join(prefix):query for prefix,query in self.queries.items()}
        return result


@contextmanager
def construction_adapter():
    original=path_pipeline.SafeFrontierAdapter
    path_pipeline.SafeFrontierAdapter=Adapter
    try:
        yield
    finally:
        path_pipeline.SafeFrontierAdapter=original


def build_spec(topology,source,prefixes,bindings):
    """Reuse the frozen prefix/state record schema; selected prefixes stay primary."""
    physical=sorted(set(tuple(map(str,p)) for p in prefixes),key=lambda p:tuple(map(int,p)))
    require(physical and all(2<=len(p)<=3 and p[0]==str(source) for p in physical),"Invalid unified tree")
    leaves=[];classifications=[]
    for prefix in physical:
        records=specs._leaf_records(topology,prefix,"public_primary")
        leaves.extend(records);first=records[0]
        names=("prefix","tier","fully_public_observed","commercial_compatibility","public_roles",
               "public_role_conflict","unknown_is_not_absence","commercial_incompatible_is_not_route_nonexistence")
        row={name:first[name] for name in names}
        row.update(layer="public_primary",phase_states=sorted({r["phase_state"] for r in records}),
            phase_traces=[r["phase_trace"] for r in records],forced_unknown_fallback=False,structural_weight=1.)
        classifications.append(row)
    files=dict(topology.inputs_sha256,**bindings)
    files[str(Path(specs.__file__).resolve())]=sha256(specs.__file__)
    verify_files(files)
    shape=topology.source_features(str(source))
    return dict(version=POLICY["version"],date=topology.date,source=str(source),maximum_prefix_AS_count=3,
        status="FROZEN_FOR_EVALUATION",configuration=POLICY,configuration_sha256=specs.content_sha(POLICY),
        source_shape=shape,firsthop_policy="rho-conditioned dual model selection",
        selected_firsthops=sorted({p[1] for p in physical},key=int),
        all_public_firsthops=list(map(str,topology.neighbors(source))),firsthop_decisions=[],
        primary_leaves=leaves,fallback_leaves=[],prefix_classification=classifications,
        primary_unique_prefix_count=len(physical),primary_phase_leaf_count=len(leaves),
        fallback_unique_prefix_count=0,fallback_phase_leaf_count=0,original_R_s_prefix_count=len(physical),
        original_R_s_prefix_set_sha256=signature(physical),admitted_path_set_sha256=None,
        library_csv_sha256=None,construction_method_sha256=None,
        graph_identity_sha256=topology.graph_identity_sha256,data_code_input_sha256=files,
        phase_state_and_parent_context_are_not_merged=True,unknown_is_not_absence=True,
        ordinal_public_layer_priority_is_not_probability=True,no_formal_reference_paths=True,
        no_model_training_or_PG_started=True,no_augmentation_subset_reconstruction=True)


def construct(run,source):
    run=Path(run);source=str(source);proto=read_json(run/"PROTOCOL.json")
    require(source in proto["sources"],"Unregistered source")
    verify_files(proto["code_sha256"])
    ready=run/"construction"/source/"SOURCE_READY.json"
    if ready.exists():
        value=read_json(ready);verify_files(value["outputs_sha256"])
        require(value["protocol_sha256"]==sha256(run/"PROTOCOL.json"),"Ready source protocol differs")
        return value
    upstream=read_json(run/"UPSTREAM_BOUND.json");bundle=read_json(upstream["bundle"])
    verify_files({upstream["model"]:upstream["model_sha256"],upstream["bundle"]:upstream["bundle_sha256"]})
    data=read_json(run/"data/data_manifest.json");topology=PublicPrefixTopology.from_manifest(run/"data/data_manifest.json")
    config=copy.deepcopy(proto["upstream_config"]);config["root_selection"]["max_k"]=POLICY["low_root_limit"]
    output=run/"construction"/source
    # Source labels, final predictions and mixed test descriptive tables remain closed.
    with forbid_formal_references():
        runtime=Runtime(bundle,data,config,topology);store=DestinationFeatureStore(bundle,runtime.public)
        with construction_adapter():
            case=path_pipeline.construct_case(source,runtime,config,store,output)
        plan=read_json(output/"plan.json")
        require(plan["prefixes"],"No evidence-admissible source prefix; source remains unresolved")
        with Path(case["candidates"]).open("rb") as stream:
            candidates=pickle.load(stream)["candidates"]
        with np.load(case["features"],allow_pickle=False) as archive:
            matrix=archive["X"]
        model=lgb.Booster(model_file=upstream["model"])
        require(matrix.shape==(len(candidates),255) and model.feature_name()==case["feature_names"],"Original scorer feature schema differs")
        raw=model.predict(matrix,num_threads=1)
        paths,scores,diagnostics=normalize_pool(source,[c["path"] for c in candidates],raw)
        shape=proto["public_features"][source];high=shape["rho"]>1
        threshold=POLICY["quality"]["source_threshold"]
        minimum=POLICY["quality"]["minimum_paths"]
        selected,selection=quality_selection(paths,scores,threshold,minimum)
        admitted=set(selected);best={}
        for index,c in enumerate(candidates):
            path=tuple(map(str,c["path"]))
            if path not in best or raw[index]>raw[best[path]]:
                best[path]=index
        maximum=float(scores.max());records=[]
        for path,score in zip(paths,scores):
            if path in admitted:
                records.append(dict(path=list(path),prefix=list(map(str,candidates[best[path]]["prefix"])),
                    raw_score=float(score),relative_score=float(score)/maximum))
        directory=run/"augmentation"/("as"+source);directory.mkdir(parents=True,exist_ok=True)
        csv_bytes,json_bytes=path_bytes(sorted(admitted))
        freeze_bytes(directory/"paths.csv",csv_bytes);freeze_bytes(directory/"paths.json",json_bytes)
        freeze(directory/"PATH_RECORDS.json",records)
        inputs={str(output/"CASE.json"):sha256(output/"CASE.json"),**case["files_sha256"]}
        item=dict(status="FROZEN",date=proto["date"],source=source,
            strategy="source_unified_tree_high_q075_min1000" if high else "source_unified_tree_low_q075_min1000",
            rho=shape["rho"],public_features=shape,selection=selection,rows=len(admitted),
            destinations=len({p[-1] for p in admitted}),path=str(directory/"paths.csv"),json_path=str(directory/"paths.json"),
            sha256=sha256(directory/"paths.csv"),json_sha256=sha256(directory/"paths.json"),
            path_set_sha256=signature(admitted),inputs_sha256=inputs,
            protocol_sha256=sha256(run/"PROTOCOL.json"),final_labels_read=False,
            raw_candidate_paths=len(paths),score_diagnostics=diagnostics)
        freeze(directory/"ADMISSION.json",item)
        tree=build_spec(topology,source,plan["prefixes"],dict(inputs,**{str(directory/"ADMISSION.json"):sha256(directory/"ADMISSION.json")}))
        tree.update(admitted_path_set_sha256=item["path_set_sha256"],library_csv=item["path"],
                    library_csv_sha256=item["sha256"],construction_method_sha256=sha256(run/"query_trees/METHOD_FROZEN.json"))
        freeze(output/"augmentation_tree.json",tree)
        freeze(run/"query_trees"/(source+".json"),tree)
        freeze(output/"CONSTRUCTION_COMPLETED.json",dict(date=proto["date"],source=source,
            tree_prefixes_sha256=signature(plan["prefixes"]),tree_prefixes=len(set(map(tuple,plan["prefixes"]))),
            original_scorer_sha256=upstream["model_sha256"],final_reference_paths_opened=False))
    # Frozen downstream registration reads endpoint metadata and binds reference hashes.
    # Route sequences remain closed until the original checkpoint/evaluation gate.
    if data["mode"] == "traceroute":
        from .trace import prepare_source
        return prepare_source(run, source)
    from pathveil.experiment import preparation
    original=preparation.specs.build_spec
    def reuse_tree(graph,s,original_prefixes,input_bindings=None):
        require(str(s)==source,"Wrong source during unified-tree preparation")
        return read_json(run/"query_trees"/(source+".json"))
    preparation.specs.build_spec=reuse_tree
    release=read_json(Path(upstream["run"])/"upstream/RELEASE.json")
    graph={key:release[key] for key in ("selected_neighbors_csv","recursive_anchors_csv")}
    try:
        preparation.prepare_source(run,source,Path(proto["baseline_run"])/"data/downstream_view/data_manifest.json",topology,graph)
    finally:
        preparation.specs.build_spec=original
    prepared_value=read_json(run/"T-PG-B/pg_inputs"/("as"+source)/"PREPARED.json")
    actual_tree=read_json(run/"query_trees"/(source+".json"))
    require(prepared_value["training_generator"]=="Tree B + PATS",
            "Frozen rho-group training generator differs")
    require({tuple(p["prefix"]) for p in actual_tree["primary_leaves"]}==set(map(tuple,plan["prefixes"])),
            "PATS augmentation/query trees differ")
    prepared=run/"T-PG-B/pg_inputs"/("as"+source)/"PREPARED.json"
    outputs=[prepared,run/"query_trees"/(source+".json"),directory/"ADMISSION.json",output/"CONSTRUCTION_COMPLETED.json"]
    value=dict(status="PASS",source=source,date=proto["date"],protocol_sha256=sha256(run/"PROTOCOL.json"),
               outputs_sha256={str(f):sha256(f) for f in outputs},completed_utc=utc_now())
    freeze(ready,value)
    return value
