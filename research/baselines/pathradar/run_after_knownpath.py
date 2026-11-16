"""Run five PathRadar BGP dates after KnownPath has finished."""

import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).resolve().parent
DATES = ("20260201", "20260301", "20260401", "20260501", "20260601")


def main():
    knownpath_root = ROOT / "runs/baselines/knownpath"
    batch_log = knownpath_root / "BATCH.log"
    while True:
        if all((knownpath_root / date / "COMPLETE.json").is_file() for date in DATES):
            break
        if batch_log.is_file() and "Traceback (most recent call last)" in batch_log.read_text():
            raise RuntimeError("KnownPath batch failed; PathRadar was not started")
        time.sleep(60)

    for date in DATES:
        output = ROOT / "runs/baselines/pathradar" / date / "COMPLETE.json"
        if output.is_file():
            result = json.loads(output.read_text())
            if (result.get("date") == date and result.get("round") == 2
                    and result.get("requested_pairs", 0) > 0):
                print(date, "already complete", flush=True)
                continue
            if result.get("date") != date:
                raise ValueError("Invalid completion record: " + str(output))
        print(date, "starting", flush=True)
        subprocess.run([sys.executable, "-u", str(SOURCE / "run_baseline.py"), date], check=True)
    subprocess.run([sys.executable, "-u", str(SOURCE.parent / "report.py")], check=True)
    print("All five PathRadar BGP dates complete", flush=True)


if __name__ == "__main__":
    main()
