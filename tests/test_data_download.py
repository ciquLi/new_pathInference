"""Acquisition must reject plausible-looking incomplete or shifted snapshots."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathveil.data.downloader import download_asrank, query


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def fetch(self, kind, date, offset, size):
        rows = [
            {"date": "2026-05-01", "asn": str(asn)}
            for asn in range(offset + 1, min(offset + size, 3) + 1)
        ]
        return {
            "totalCount": 3,
            "pageInfo": {"hasNextPage": offset + len(rows) < 3},
            "edges": [{"node": row} for row in rows],
        }

    def test_complete_pages_and_final_manifest(self):
        result = download_asrank(
            "20260501", "asns", self.root, expected_count=3, page_size=2, fetcher=self.fetch
        )
        self.assertEqual(result["rows"], 3)
        self.assertEqual(result["unique_rows"], 3)
        self.assertEqual([page["count"] for page in result["pages"]], [2, 1])
        self.assertTrue(result["all_dates_verified"])
        self.assertEqual(len(result["sha256"]), 64)
        self.assertEqual(
            json.loads((self.root / "asns20260501.manifest.json").read_text())["status"], "complete"
        )

    def test_duplicate_across_pages_does_not_publish(self):
        def duplicate(kind, date, offset, size):
            result = self.fetch(kind, date, offset, size)
            if offset:
                result["edges"][0]["node"]["asn"] = "1"
            return result

        with self.assertRaisesRegex(ValueError, "Duplicate"):
            download_asrank("20260501", "asns", self.root, page_size=2, fetcher=duplicate)
        self.assertFalse((self.root / "asns20260501.jsonl").exists())
        self.assertEqual(
            json.loads((self.root / "asns20260501.manifest.json").read_text())["status"], "failed"
        )

    def test_wrong_month_rejected_even_with_correct_row_count(self):
        def shifted(*args):
            result = self.fetch(*args)
            result["edges"][0]["node"]["date"] = "2026-04-01"
            return result

        with self.assertRaisesRegex(ValueError, "different date"):
            download_asrank("20260501", "asns", self.root, page_size=2, fetcher=shifted)
        self.assertFalse((self.root / "asns20260501.jsonl").exists())

    def test_incomplete_page_and_changing_count_rejected(self):
        for variation in ["truncate", "count"]:
            root = self.root / variation

            def bad(kind, date, offset, size):
                result = self.fetch(kind, date, offset, size)
                if offset:
                    if variation == "truncate":
                        result["edges"] = []
                    else:
                        result["totalCount"] = 4
                return result

            with self.assertRaises(ValueError):
                download_asrank("20260501", "asns", root, page_size=2, fetcher=bad)
            self.assertFalse((root / "asns20260501.jsonl").exists())

    def test_date_filtered_count_and_expected_count(self):
        text = query("asns", "20260501", 0, 5000)
        self.assertIn('dateStart:"2026-05-01",dateEnd:"2026-05-01"', text)
        self.assertIn("totalCount", text)
        with self.assertRaisesRegex(ValueError, "Registered ASRank count"):
            download_asrank("20260501", "asns", self.root, expected_count=4, fetcher=self.fetch)

    def test_checked_pages_resume_but_corrupt_cache_rejected(self):
        calls = []

        def interrupted(kind, date, offset, size):
            if offset:
                raise RuntimeError("offline")
            return self.fetch(kind, date, offset, size)

        with self.assertRaises(RuntimeError):
            download_asrank(
                "20260501", "asns", self.root, page_size=2, workers=1, fetcher=interrupted
            )
        page = self.root / ".pages/20260501/asns/2/000000000.json"
        self.assertTrue(page.exists())
        data = json.loads(page.read_text())
        data["connection"]["edges"][0]["node"]["asn"] = "99"
        page.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            download_asrank(
                "20260501", "asns", self.root, page_size=2, workers=1, fetcher=self.fetch
            )


if __name__ == "__main__":
    unittest.main()
