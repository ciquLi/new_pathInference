import argparse
import os
from pathlib import Path

from .workflow import prepare,execute,build_all


def main():
    parser=argparse.ArgumentParser(description="Registered-date frozen rho-adaptive workflow")
    parser.add_argument("date")
    parser.add_argument("--project-root",type=Path,default=Path(os.environ.get("PATHVEIL_PROJECT_ROOT",Path(__file__).resolve().parents[3])))
    parser.add_argument("--stage",choices=["prepare","run","build_all","source"],default="run")
    parser.add_argument("--run-dir",type=Path)
    parser.add_argument("--source")
    parser.add_argument("--device",choices=["cpu","auto","cuda"],default="cuda")
    args=parser.parse_args();os.environ["PATHVEIL_PROJECT_ROOT"]=str(args.project_root.resolve())
    if args.stage=="prepare":return prepare(args.project_root.resolve(),args.date)
    run=args.run_dir
    if run is None:run=prepare(args.project_root.resolve(),args.date)
    if args.stage=="source":
        from .construction import construct
        if not args.source:parser.error("--source is required")
        construct(run,args.source)
    elif args.stage=="build_all":build_all(run)
    else:execute(run,args.device)


if __name__=="__main__":main()
