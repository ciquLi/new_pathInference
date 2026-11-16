from __future__ import annotations

import os
import sqlite3
from pathlib import Path

REPO_DIR = Path(os.environ.get("PATHRADAR_RUN_DIR", Path(__file__).resolve().parent.parent))
INTERMEDIATE_DIR = Path(os.environ.get("PATHRADAR_INTERMEDIATE_DIR", REPO_DIR / "intermediate"))
DB_PATH = INTERMEDIATE_DIR / "pathradar_store.sqlite"


class PathRadarStore:
    def __init__(self, db_path: Path | str = DB_PATH):
        self.db_path = Path(db_path)
        if os.environ.get("PATHRADAR_READONLY_DB") == "1":
            self.conn = sqlite3.connect(
                self.db_path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True
            )
        else:
            self.conn = sqlite3.connect(str(self.db_path))
            self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA temp_store=MEMORY")

    def close(self) -> None:
        self.conn.close()

    def init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS grouped_paths (
                dataset TEXT NOT NULL,
                asn TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY (dataset, asn)
            );
            CREATE INDEX IF NOT EXISTS idx_grouped_paths_dataset
            ON grouped_paths(dataset);
            """
        )
        self.conn.commit()

    def upsert_many(self, dataset: str, items: list[tuple[str, str]]) -> None:
        self.conn.executemany(
            """
            INSERT INTO grouped_paths(dataset, asn, payload)
            VALUES (?, ?, ?)
            ON CONFLICT(dataset, asn) DO UPDATE SET payload=excluded.payload
            """,
            ((dataset, asn, payload) for asn, payload in items),
        )

    def commit(self) -> None:
        self.conn.commit()

    def has_db(self) -> bool:
        return self.db_path.exists()

    def list_asns(self, dataset: str) -> list[str]:
        cur = self.conn.execute(
            "SELECT asn FROM grouped_paths WHERE dataset = ? ORDER BY asn",
            (dataset,),
        )
        return [row[0] for row in cur]

    def list_asn_set(self, dataset: str) -> set[str]:
        return set(self.list_asns(dataset))

    def get_payload(self, dataset: str, asn: str) -> str | None:
        cur = self.conn.execute(
            "SELECT payload FROM grouped_paths WHERE dataset = ? AND asn = ?",
            (dataset, asn),
        )
        row = cur.fetchone()
        return row[0] if row else None

    def iter_dataset(self, dataset: str):
        cur = self.conn.execute(
            "SELECT asn, payload FROM grouped_paths WHERE dataset = ? ORDER BY asn",
            (dataset,),
        )
        yield from cur

    def count_dataset(self, dataset: str) -> int:
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM grouped_paths WHERE dataset = ?",
            (dataset,),
        )
        return int(cur.fetchone()[0])
