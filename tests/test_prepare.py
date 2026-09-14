"""Tests for pinned archive inventory and content-addressed corpus manifests."""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare import parse_7z_slt, write_corpus_manifest
from audit_pack import summarize_reanalysis


class ArchiveListingTests(unittest.TestCase):
    def test_parse_7z_slt_records(self):
        records = parse_7z_slt(
            "Path = archive.7z\nType = 7z\n\n"
            "Path = 1000000a/a.pack\nSize = 123\nFolder = -\n\n"
        )
        self.assertEqual(records[1]["Path"], "1000000a/a.pack")
        self.assertEqual(records[1]["Size"], "123")


class CorpusManifestTests(unittest.TestCase):
    def test_duplicate_pack_is_one_unique_file_with_two_sources(self):
        corpus = {
            "teacher_id": "teacher",
            "requested_nodes": 1_000_000,
            "format": "gensfen-pack",
        }
        first = {
            "archive_sha256": "a" * 64,
            "files": [{
                "member": "1000000a/same.pack", "bytes": 10,
                "sha256": "c" * 64, "path": "packs/c.pack",
            }],
        }
        second = {
            "archive_sha256": "b" * 64,
            "files": [{
                "member": "1000000b/same.pack", "bytes": 10,
                "sha256": "c" * 64, "path": "packs/c.pack",
            }],
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, manifest = write_corpus_manifest(root, "corpus", corpus, first)
            _, manifest = write_corpus_manifest(root, "corpus", corpus, second)
        self.assertEqual(len(manifest["unique_files"]), 1)
        self.assertEqual(len(manifest["unique_files"][0]["sources"]), 2)


class AuditSummaryTests(unittest.TestCase):
    def test_bounds_are_excluded_from_point_error(self):
        samples = [
            {"reanalysis": {
                "label_delta_cp_stm": -10, "selected_move_matches": True,
                "wall_seconds": 2.0, "node_limit_fraction": 1.0,
            }},
            {"reanalysis": {
                "label_delta_cp_stm": None, "selected_move_matches": False,
                "wall_seconds": 3.0, "node_limit_fraction": 0.9,
            }},
        ]
        result = summarize_reanalysis(samples)
        self.assertEqual(result["exact_scores"], 1)
        self.assertEqual(result["bounded_scores"], 1)
        self.assertEqual(result["exact_score_mae_cp"], 10.0)
        self.assertEqual(result["selected_move_matches"], 1)


if __name__ == "__main__":
    unittest.main()
