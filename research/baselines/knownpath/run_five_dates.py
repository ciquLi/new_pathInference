"""Sequential, resumable KnownPath BGP baseline dispatcher."""

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).resolve().parent
DATES = ("20260201", "20260301", "20260401", "20260501", "20260601")


def main():
    root = ROOT / "runs/baselines/knownpath"
    root.mkdir(parents=True, exist_ok=True)
    for date in DATES:
        complete = root / date / "COMPLETE.json"
        if complete.is_file():
            report = json.loads(complete.read_text())
            if report.get("date") != date or report.get("scored_pairs") != report.get("expected_pairs"):
                raise ValueError("Invalid completion record: " + str(complete))
            print(date, "already complete", flush=True)
            continue
        command = [sys.executable, str(SOURCE / "run_baseline.py"), date,
                   "--workers", "8"]
        if date == "20260201":
            command.append("--resume-four-to-eight")
        print(date, "starting", flush=True)
        subprocess.run(command, check=True)
    print("All five KnownPath BGP dates complete", flush=True)


if __name__ == "__main__":
    main()
