"""Bridge the single formal date entry to the frozen adaptive numerical APIs."""
import json
from pathlib import Path

from pathveil.data.preflight import check_date
from pathveil.utils import project_root,read_json,utc_now
from .workflow import prepare,execute as run


def execute(args,registry):
    root=project_root()
    if args.check_only:
        report=check_date(args.date,deep=args.deep_check)
        print(json.dumps(report,indent=2,ensure_ascii=False))
        return 0 if report["raw_ready"] else 2
    if args.plan_only:
        from .policy import high_secondhop_count,POLICY
        from .workflow import load_method
        method,path=load_method(root,registry,getattr(args,"method_config",None))
        baseline=Path(read_json(root/"runs"/args.date/"rho_prefix_v1/current.json")["run_dir"])
        proto=read_json(baseline/"PROTOCOL.json")
        print(json.dumps(dict(date=args.date,workflow=POLICY["version"],config=str(path),
            sources=[dict(source=s,rho=proto["public_features"][s]["rho"],
                secondhop_topk=high_secondhop_count(proto["public_features"][s]["rho"]) if proto["public_features"][s]["rho"]>1 else 10)
                for s in registry["test_sources"]],selection_policy=POLICY,
            new_training_started=False),indent=2))
        return 0
    requested=args.run_dir
    if getattr(args,"fresh_run",False):
        if requested:raise ValueError("Use either --fresh-run or --run-dir")
        requested=root/"runs"/args.date/"rho_adaptive_v1"/("fresh_"+utc_now().replace(":","").replace("-","").replace(".","_"))
    native=prepare(root,args.date,requested,getattr(args,"method_config",None))
    if args.prepare_only or args.upstream_only:return 0
    if args.augmentation_only:
        from .workflow import build_all
        build_all(native)
        return 0
    run(native,args.device)
    return 0
