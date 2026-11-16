"""Record a completed local train-only KnownPath index."""

import hashlib
import json
from pathlib import Path
import sqlite3
import sys


ROOT = Path(__file__).resolve().parents[3]
FILES = ROOT.parent / "files"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(date):
    output = ROOT / "runs/baselines/knownpath/indexes" / date
    db = output / "knownpath.db"
    train = FILES / "data/aspath/path_by_vantage" / date / "path_without_prefix/train"
    connection = sqlite3.connect(db.as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        rows = connection.execute("SELECT COUNT(*) FROM sure_paths").fetchone()[0]
    finally:
        connection.close()
    if rows <= 0:
        raise ValueError("Empty KnownPath train index")
    digest = hashlib.sha256()
    files = sorted(train.glob("*.csv"))
    for path in files:
        digest.update(path.name.encode())
        digest.update(bytes.fromhex(sha256(path)))
    record = {"date": date, "train_csv_files": len(files),
              "train_csv_sha256": digest.hexdigest(), "db_rows": rows,
              "db_sha256": sha256(db)}
    (output / "INDEX_COMPLETE.json").write_text(json.dumps(record, indent=2) + "\n")
    print(date, "train index complete", rows, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
