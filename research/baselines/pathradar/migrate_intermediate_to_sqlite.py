#!/usr/bin/env python3

import argparse
from pathlib import Path

from data_store import DB_PATH, INTERMEDIATE_DIR, PathRadarStore


def dataset_path(dataset: str) -> Path:
    return INTERMEDIATE_DIR / dataset


def asn_from_name(dataset: str, path: Path) -> str:
    if dataset == "shortest":
        return path.stem
    return path.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate grouped intermediate files to sqlite.")
    parser.add_argument(
        "--db-path",
        default=str(DB_PATH),
        help="SQLite database path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Commit batch size.",
    )
    args = parser.parse_args()

    datasets = ["data", "train", "validation", "test", "rib", "origintest", "shortest"]
    store = PathRadarStore(args.db_path)
    store.init_schema()

    for dataset in datasets:
        root = dataset_path(dataset)
        if not root.is_dir():
            print(f"[skip] dataset={dataset} missing_dir={root}", flush=True)
            continue

        files = sorted(p for p in root.iterdir() if p.is_file())
        if dataset == "shortest":
            invalid = [path.name for path in files if not path.name.endswith(".txt")]
            if invalid:
                raise SystemExit(
                    f"invalid shortest artifact(s): {invalid[:10]}; expected only <asn>.txt files"
                )
            rib_root = dataset_path("rib")
            if rib_root.is_dir():
                expected = {path.name for path in rib_root.iterdir() if path.is_file()}
                actual = {path.stem for path in files}
                if actual != expected:
                    missing = sorted(expected - actual)
                    extra = sorted(actual - expected)
                    raise SystemExit(
                        f"shortest ASN set mismatch: missing={missing[:10]} extra={extra[:10]}"
                    )
            files = [path for path in files if path.name.endswith(".txt")]
        total = len(files)
        print(f"[start] dataset={dataset} files={total}", flush=True)
        batch: list[tuple[str, str]] = []
        for index, path in enumerate(files, start=1):
            payload = path.read_text(encoding="utf-8", errors="ignore")
            batch.append((asn_from_name(dataset, path), payload))
            if len(batch) >= args.batch_size:
                store.upsert_many(dataset, batch)
                store.commit()
                batch.clear()
            if index % 5000 == 0 or index == total:
                print(f"[progress] dataset={dataset} processed={index}/{total}", flush=True)
        if batch:
            store.upsert_many(dataset, batch)
            store.commit()
        print(f"[done] dataset={dataset} rows={store.count_dataset(dataset)}", flush=True)

    store.close()
    print(f"[complete] sqlite={args.db_path}", flush=True)


if __name__ == "__main__":
    main()
