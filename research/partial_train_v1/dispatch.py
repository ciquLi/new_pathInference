#!/usr/bin/env python3
"""Resource dispatch only; numerical research and production APIs stay frozen."""
import fcntl
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pathInference.research import partial_train as experiment
from pathInference.experiment.contracts import freeze
from pathInference.utils import read_json, sha256, utc_now, write_json


def main():
    root, batch = experiment.ROOT, experiment.BATCH
    batch.mkdir(parents=True, exist_ok=True)
    with (batch / ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = read_json(experiment.CONFIG)
        freeze(batch / "CONFIG.json", config)
        freeze(batch / "DISPATCH_IDENTITY.json", dict(
            script=str(Path(__file__).resolve()), sha256=sha256(__file__),
            numerical_script_sha256=sha256(experiment.__file__),
            config_sha256=sha256(experiment.CONFIG)))
        try:
            for date in config["dates"]:
                logs = batch / "logs"
                run = root / "runs" / date / experiment.NAME / "fixed25"
                experiment.run_command([sys.executable, "-B", "-m", "pathInference.research.partial_train",
                                        "--stage", "prepare", "--date", date],
                                       logs / (date + "_prepare.log"), date + " 准备混合训练库")
                experiment.run_command(["env", "PATHVEIL_STAGE_GUARD_ROOT=" + str(run),
                                        sys.executable, "-B", "-m", "pathInference.pats.batch_pathgen", "--run-dir", str(run)],
                                       logs / (date + "_pathgen.log"), date + " 两AS候选生成")
                for source in experiment.SOURCES:
                    for stage in ("train", "annotate", "evaluate", "rescore"):
                        experiment.run_command([sys.executable, "-B", "-m", "pathInference.experiment.worker",
                                                "--run", str(run), "--stage", stage, "--source", source,
                                                "--device", "cuda", "--cpu-threads", "4"],
                                               logs / (date + "_" + source + "_" + stage + ".log"),
                                               date + " AS" + source + " " + stage)
                experiment.report()
            write_json(batch / "STATUS.json", dict(status="COMPLETED", completed=10, updated_utc=utc_now()))
            (batch / "PROGRESS.md").write_text("# 两AS部分吸纳实验\n\n已完成五日期共10/10组。结果见 REPORT.md、RESULTS.csv。\n")
        except BaseException as error:
            write_json(batch / "STATUS.json", dict(status="FAILED", error=repr(error), updated_utc=utc_now()))
            with (batch / "PROGRESS.md").open("a") as stream:
                stream.write("\n任务失败：" + repr(error) + "\n")
            raise


if __name__ == "__main__":
    main()
