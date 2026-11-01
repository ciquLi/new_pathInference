"""Read completed metrics and publish progress for explicitly selected dated runs."""
import argparse
import csv
from datetime import datetime,timezone
import io
import json
from pathlib import Path
import time


def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+".watch.tmp");temporary.write_text(text);temporary.replace(path)


def publish(run):
    proto=read(run/"PROTOCOL.json")
    state=read(run/"STATUS.json",{});construction=read(run/"CONSTRUCTION_STATUS.json",{})
    outcome=read(run/"RESULTS.json",{})
    results={r["source_asn"]:r for r in outcome.get("per_source",[])}
    baseline={r["source_asn"]:r for r in read(Path(proto["baseline_run"])/"RESULTS.json",{}).get("per_source",[])}
    full_tree=proto["version"]=="rho_adaptive_full_tree_v2"
    description=("两类AS训练和查询均用同一完整前缀树+PG；首跳选择与递归规则保持。已验证等价的完成源直接复用并记录来源。" if full_tree else
        "低ρ保留原PG训练及原S-N查询树，仅首跳上限改为21；高ρ双Top3并集，各首跳强制递归，二跳数量由固定ρ公式决定。")
    lines=["# "+proto["date"]+"："+proto["version"], "",
        "状态：{}；结构准备{}/{}；下游{}/{}。".format(state.get("status","PREPARING"),len(construction.get("completed",[])),len(proto["sources"]),state.get("complete",0),state.get("total",6*len(proto["sources"]))),
        "", description, "",
        "| AS | ρ | 当前阶段 | 增强路径 | 有效阈值 | 原Jaccard | 新Jaccard | 变化/百分点 |",
        "|---|---:|---|---:|---:|---:|---:|---:|"]
    rows=[]
    for source in proto["sources"]:
        admission=read(run/"augmentation"/("as"+source)/"ADMISSION.json",{})
        active=[r["stage"] for r in state.get("active",[]) if r["source"]==source]
        phase="completed" if source in results else ",".join(active) if active else "constructing" if source in construction.get("active",{}) else "queued"
        if source in construction.get("failures",{}):phase="CONSTRUCTION_FAILED"
        if any(key.endswith(":"+source) for key in state.get("failures",{})):phase="DOWNSTREAM_FAILED"
        row=dict(date=proto["date"],source=source,rho=proto["public_features"][source]["rho"],phase=phase,
            augmentation_paths=admission.get("rows"),effective_threshold=admission.get("selection",{}).get("effective_threshold"),
            baseline_jaccard=baseline.get(source,{}).get("jaccard_accuracy"),jaccard=results.get(source,{}).get("jaccard_accuracy"),delta_percentage_points=None)
        if source in results and source in baseline:
            a,b=results[source],baseline[source]
            if a["fixed_identity"]!=b["fixed_identity"] or a["query_count"]!=b["query_count"]:row["phase"]="INCOMPARABLE_REFERENCE"
            else:row["delta_percentage_points"]=100*(a["jaccard_accuracy"]-b["jaccard_accuracy"])
        rows.append(row)
        number=lambda v:"—" if v is None else str(v)
        pct=lambda v:"—" if v is None else "{:.4%}".format(v)
        lines.append("| {} | {:.4f} | {} | {} | {} | {} | {} | {} |".format(source,row["rho"],row["phase"],
            number(row["augmentation_paths"]),number(row["effective_threshold"]),pct(row["baseline_jaccard"]),pct(row["jaccard"]),number(row["delta_percentage_points"])))
    lines += ["", "[逐AS完整评测]({}/REPORT.md) · [调度详情]({}/PROGRESS.md) · [冻结方法]({}/PLAN.json)".format(run,run,run),
        "", "每30秒更新。未完成不计平均；最终参考只用于冻结后的评测。", "",
        "更新UTC："+datetime.now(timezone.utc).isoformat(), ""]
    content="\n".join(lines)
    write(run/"LIVE_PROGRESS.md",content)
    pointer=read(run.parent/"current.json",{})
    if Path(pointer.get("run_dir","/nonexistent")).resolve()==run:
        write(run.parent/"PROGRESS.md",content)
    stream=io.StringIO(newline="");writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(run/"BASELINE_COMPARISON.csv",stream.getvalue())
    return content,state.get("status") in ("COMPLETED","FAILED","SUPERSEDED")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs",nargs="+",type=Path,required=True)
    parser.add_argument("--overview",type=Path);parser.add_argument("--watch",action="store_true")
    args=parser.parse_args()
    runs=[p.resolve() for p in args.run_dirs]
    while True:
        reports=[publish(run) for run in runs]
        if args.overview:write(args.overview,"\n\n---\n\n".join(text for text,done in reports))
        if not args.watch or all(done for text,done in reports):break
        time.sleep(30)
