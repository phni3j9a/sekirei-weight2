"""Public source/projection checks for the dedicated weekly input route."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import prepare_weekly_nonlinear as prepare


class PrepareTests(unittest.TestCase):
    def test_base_snapshot_remains_complete_and_hash_exact(self):
        prepare.verify_base_snapshot()

    def test_weekly_reader_preserves_join_and_numeric_export_bodies(self):
        old = (prepare.BASE / "source/crates/sekirei-train/src/paired_nonlinear_actual.rs").read_text()
        new = (prepare.REPAIR / "paired_nonlinear_actual.rs").read_text()
        for begin, end in (("pub fn strict_train_join(", "pub struct ReaderExporter"),
                           ("    fn commit_completed_native03(", "pub fn actual_entry"),
                           ("pub fn verify_inputs_and_initialized_io(", None)):
            old_part = old[old.index(begin):old.index(end)] if end else old[old.index(begin):]
            new_part = new[new.index(begin):new.index(end)] if end else new[new.index(begin):new.index("#[cfg(test)]", new.index(begin))]
            self.assertEqual(old_part.rstrip(), new_part.rstrip())

    def test_profile_codegen_freezes_short_pack_counts_with_distinct_mode(self):
        identity = {"bytes": 1, "sha256": "a" * 64}
        packs = [{"sha256": f"{i:064x}", "bytes": 1000,
                  "games": 392 if i == 1 else 10000,
                  "selected_games": 392 if i == 1 else 500} for i in range(1, 14)]
        profile = {"kind": "whole-pack-hash-ranked-frozen-holdout-profile-v2",
                   "sampling": "whole-pack SHA256(seed:pack-sha:index) rank; explicit per-pack game counts; ply>=16/every4/max32",
                   "frozen_dataset_manifest_sha256": prepare.FROZEN_MANIFEST_SHA,
                   "train_budget": 112681, "selection_seed": "public-fixture", "games_per_pack": 500,
                   "packs": packs, "corpus_manifest_sha256": prepare.TEACHER.split(":")[-1]}
        manifest = {"teacher_identity": prepare.TEACHER, "positions": {"train": 112681, "holdout": 5895},
                    "derivation": {"kind": "whole-pack-hash-ranked-frozen-holdout-v2",
                                   "profile_sha256": identity["sha256"], "profile": profile}}
        plan = {"schema": "sekirei.weekly-nonlinear-selected-plan.v1", "status": "frozen-before-build",
                "mode": "weekly-public-fixture-e3-v1", "teacher_identity": prepare.TEACHER,
                "dataset_inputs": {name: dict(identity) for name in prepare.DATASET_FILES},
                "selection_profile": identity}
        source = prepare.profile_source(plan, identity, manifest, identity, profile, identity)
        self.assertIn(b"pub const TOTAL_SELECTED_GAMES:u64=6392;", source)
        self.assertIn(b"pub const GAMES_PER_PACK:u64=500;", source)
        self.assertIn(b"weekly-public-fixture-e3-v1", source)
        for change in ("old_mode", "wrong_profile", "wrong_selected"):
            p, m, f = deepcopy(plan), deepcopy(manifest), deepcopy(profile)
            if change == "old_mode":
                p["mode"] = "white-view-paired-nonlinear-adam-e3-v1"
            elif change == "wrong_profile":
                m["derivation"]["profile_sha256"] = "b" * 64
            else:
                f["packs"][0]["selected_games"] = 500
                m["derivation"]["profile"] = f
            with self.subTest(change=change), self.assertRaises(ValueError):
                prepare.profile_source(p, identity, m, identity, f, identity)

    def test_duplicate_nonfinite_json_is_rejected_before_profile_codegen(self):
        for raw in (b'{"x": 1, "x": 1}', b'{"x": NaN}', b'{"x": Infinity}'):
            with self.assertRaises(ValueError):
                prepare.raw_json(raw)


if __name__ == "__main__":
    unittest.main()
