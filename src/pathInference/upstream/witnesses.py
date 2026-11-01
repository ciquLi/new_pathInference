"""Origin-preserving terminal suffix witnesses, without formal-origin routes."""
from __future__ import annotations

from pathlib import Path
import sqlite3

from .contracts import read, require, sha, verify, write


def prepare_witnesses(data, split_manifest, targets, output):
    split = split_manifest.get("split", split_manifest)
    formal = set(map(int, split["formal_test"]))
    targets = set(map(int, targets))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    database, marker = out / "terminal_suffixes.sqlite", out / "WITNESSES.json"
    canonical = Path(data["merged_train"])
    binding = dict(
        date=data["date"],
        canonical=str(canonical),
        canonical_sha256=sha(canonical),
        targets=sorted(targets),
        excluded_raw_formal_origins=sorted(formal),
    )
    if marker.exists():
        previous = read(marker)
        require(previous["binding"] == binding, "Suffix witness cache differs from dated input")
        verify({str(database): previous["database_sha256"]})
        return database
    require(not database.exists(), "Incomplete witness database exists; use a fresh run directory")
    db = sqlite3.connect(str(database))
    db.execute(
        "CREATE TABLE witnessed(source INTEGER,text TEXT,donor INTEGER,original_path TEXT,PRIMARY KEY(source,text,donor)) WITHOUT ROWID"
    )
    batch = []
    malformed = excluded = retained = total = 0
    try:
        with canonical.open("rb") as stream:
            for raw in stream:
                total += 1
                try:
                    origin = int(raw.split(b"->", 1)[0])
                except ValueError:
                    malformed += 1
                    continue
                if origin in formal:
                    excluded += 1
                    continue
                try:
                    original = tuple(int(v) for v in raw.strip().split(b"->"))
                except ValueError:
                    malformed += 1
                    continue
                route = tuple(v for i, v in enumerate(original) if not i or v != original[i - 1])
                if (
                    len(route) < 2
                    or len(route) != len(set(route))
                    or any(not 0 < v < 2**32 for v in route)
                ):
                    malformed += 1
                    continue
                for i in range(1, len(route) - 1):
                    if route[i] in targets:
                        batch.append(
                            (
                                route[i],
                                "->".join(map(str, route[i:])),
                                origin,
                                "->".join(map(str, route)),
                            )
                        )
                        retained += 1
                if len(batch) >= 10000:
                    db.executemany("INSERT OR IGNORE INTO witnessed VALUES(?,?,?,?)", batch)
                    db.commit()
                    batch.clear()
        if batch:
            db.executemany("INSERT OR IGNORE INTO witnessed VALUES(?,?,?,?)", batch)
        db.commit()
    finally:
        db.close()
    write(
        marker,
        dict(
            status="PASS",
            binding=binding,
            database=str(database),
            database_sha256=sha(database),
            canonical_rows=total,
            malformed_rows=malformed,
            excluded_formal_origin_rows=excluded,
            retained_raw_witnesses=retained,
            complete_terminal_preserved=True,
            reverse_or_interior_terminal_synthesis=False,
            formal_reference_reads=False,
        ),
    )
    return database


class WitnessStore:
    def __init__(self, database, training_sources, folds):
        self.connection = sqlite3.connect(
            "file:" + str(Path(database).resolve()) + "?mode=ro", uri=True
        )
        self.training = set(map(int, training_sources))
        self.folds = {int(s): int(f) for s, f in folds.items()}

    def source(self, source, training_only=False):
        source = int(source)
        held = self.folds.get(source)
        result = set()
        for text, donor in self.connection.execute(
            "SELECT text,donor FROM witnessed WHERE source=?", (source,)
        ):
            if training_only and (
                donor not in self.training or held is not None and self.folds[donor] == held
            ):
                continue
            result.add(tuple(text.split("->")))
        return result

    def close(self):
        self.connection.close()
