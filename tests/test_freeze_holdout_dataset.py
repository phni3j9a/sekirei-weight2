"""Synthetic fixtures for immutable holdout and expanded-label provenance."""
from argparse import Namespace
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import freeze_holdout_dataset as freeze
from diagnose_weights import verified_split


SFENS = [
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 16",
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 17",
    "4k4/9/9/9/9/9/9/9/4K4 b - 16",
    "4k4/9/9/9/9/9/9/9/4K4 w - 17",
    "4k4/9/9/9/9/9/9/9/4K4 b P 18",
    "4k4/9/9/9/9/9/9/9/4K4 w P 19",
]
SEED = "first-weight-v1"
CORPUS = "a" * 64
PACK = "b" * 64
TEACHER = f"external:suisho11beta-1m-pack:{CORPUS}"


def game_ids(split, count):
    found = []
    for n in range(10000):
        gid = hashlib.sha256(f"public-fixture-{n}".encode()).hexdigest()
        if freeze.split_for_game(gid, SEED) == split:
            found.append(gid)
            if len(found) == count:
                return found
    raise AssertionError("cannot make deterministic fixture game IDs")


TRAIN = game_ids("train", 2)
HOLD = game_ids("holdout", 2)


def game(gid, index, split):
    return {"game_id": gid, "game_index": index, "pack_sha256": PACK, "split": split}


def position(sfen, gid):
    return {"schema_version": 1, "sfen": sfen,
            "source": {"kind": "gensfen-pack", "path": gid, "ply": int(sfen.split()[3])},
            "tags": {"side_to_move": "black" if sfen.split()[1] == "b" else "white", "phase": "middlegame"}}


def write_dataset(root, expanded):
    root.mkdir()
    games = [game(TRAIN[0], 0, "train"), game(HOLD[0], 1, "holdout")]
    if expanded:
        games += [game(TRAIN[1], 200, "train"), game(HOLD[1], 201, "holdout")]
        # Old holdout board appears at a different ply, and another row carries
        # the old holdout game ID. Neither is present in the expanded holdout.
        rows = {"train": [position(SFENS[0], TRAIN[0]), position(SFENS[2].rsplit(" ", 1)[0] + " 99", TRAIN[1]),
                          position(SFENS[3], HOLD[0]), position(SFENS[4], TRAIN[1]),
                          position(SFENS[5], HOLD[1]), position(SFENS[1], TRAIN[1])],
                "holdout": [position(SFENS[1], HOLD[1])]}
    else:
        rows = {"train": [position(SFENS[0], TRAIN[0])], "holdout": [position(SFENS[2], HOLD[0])]}
    for split, positions in rows.items():
        labels = [{"sfen": row["sfen"], "score_cp": i * 30, "teacher_identity": TEACHER,
                   "label_depth": 0} for i, row in enumerate(positions)]
        for kind, values in (("positions", positions), ("labels", labels)):
            (root / f"{split}.{kind}.jsonl").write_text("".join(json.dumps(v) + "\n" for v in values))
    manifest = {"schema_version": 1, "teacher_identity": TEACHER,
        "source_corpus_manifest_sha256": CORPUS, "dependencies": {"cshogi": "fixture", "numpy": "fixture"},
        "seed": SEED, "split": freeze.SPLIT_RECIPE, "sampling": freeze.SAMPLING,
        "games_per_pack": 400 if expanded else 200, "games": games,
        "independent_exclusions": {"games": 1000, "unique_positions": 88187,
            "corpus_canonical_sha256": "c" * 64, "source_manifest_sha256": "d" * 64,
            "policy": freeze.EXCLUSION_POLICY},
        "positions": {split: len(values) for split, values in rows.items()}, "files": {}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    refresh_hashes(root)


def refresh_hashes(root):
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"] = {name: {"sha256": freeze.sha256(root / name), "bytes": (root / name).stat().st_size}
                         for name in sorted(freeze.FILES)}
    path.write_text(json.dumps(manifest))


class FrozenDatasetTests(unittest.TestCase):
    def setup_inputs(self, root):
        expanded, frozen = root / "expanded", root / "frozen"
        write_dataset(expanded, True); write_dataset(frozen, False)
        return Namespace(expanded_dataset=expanded, frozen_holdout_dataset=frozen, output=root / "derived")

    def test_freezes_bytes_and_excludes_old_games_and_ply_independent_boards(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.setup_inputs(Path(directory))
            with redirect_stdout(io.StringIO()):
                manifest = freeze.prepare(args)
            self.assertEqual(manifest["positions"], {"train": 2, "holdout": 1})
            self.assertEqual(set(manifest["files"]), freeze.FILES)
            for kind in ("positions", "labels"):
                name = f"holdout.{kind}.jsonl"
                self.assertEqual((args.output / name).read_bytes(), (args.frozen_holdout_dataset / name).read_bytes())
            self.assertEqual(manifest["derivation"]["excluded_training_rows"], {
                "frozen_holdout_game_rows": 1, "frozen_holdout_board_rows": 1,
                "new_reserved_holdout_game_rows": 1, "expanded_reserved_holdout_board_rows": 1})
            # The current diagnostic wrapper accepts both resulting splits.
            self.assertEqual(len(verified_split(args.output, "holdout")[1]), 1)
            self.assertEqual(len(verified_split(args.output, "train")[1]), 2)
            self.assertEqual(manifest["derivation"]["frozen_holdout_game_count"], 1)
            self.assertEqual(manifest["derivation"]["reserved_new_holdout_game_count"], 1)

    def test_tampered_file_and_incompatible_exclusion_identity_are_rejected(self):
        for mode in ("file", "exclusion"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                args = self.setup_inputs(Path(directory))
                if mode == "file":
                    with (args.expanded_dataset / "train.labels.jsonl").open("a") as stream:
                        stream.write("{}\n")
                else:
                    path = args.expanded_dataset / "manifest.json"
                    manifest = json.loads(path.read_text())
                    manifest["independent_exclusions"]["corpus_canonical_sha256"] = "e" * 64
                    path.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    freeze.prepare(args)
                self.assertFalse(args.output.exists())

    def test_missing_label_and_original_game_identity_changes_are_rejected(self):
        for mode in ("label", "game"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                args = self.setup_inputs(Path(directory))
                if mode == "label":
                    path = args.expanded_dataset / "train.labels.jsonl"
                    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
                    refresh_hashes(args.expanded_dataset)
                else:
                    path = args.expanded_dataset / "manifest.json"
                    manifest = json.loads(path.read_text()); manifest["games"][0]["game_index"] = 7
                    path.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    freeze.prepare(args)
                self.assertFalse(args.output.exists())

    def test_input_mutation_after_loading_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.setup_inputs(Path(directory))
            original = freeze.verify_sources
            def change_source(expanded, frozen):
                original(expanded, frozen)
                with (args.expanded_dataset / "train.labels.jsonl").open("a") as stream:
                    stream.write(" ")
            with patch.object(freeze, "verify_sources", side_effect=change_source):
                with self.assertRaisesRegex(ValueError, "changed"):
                    freeze.prepare(args)
            self.assertFalse(args.output.exists())

    def test_existing_or_git_output_is_rejected_and_side_hand_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.setup_inputs(root)
            args.output.mkdir()
            with self.assertRaises(ValueError):
                freeze.prepare(args)
            repo = root / "public-repository"
            repo.mkdir()
            subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
            args.output = repo / "private-data"
            with self.assertRaisesRegex(ValueError, "private"):
                freeze.prepare(args)
            self.assertNotEqual(freeze.board_key(SFENS[2]), freeze.board_key(SFENS[3]))
            self.assertNotEqual(freeze.board_key(SFENS[2]), freeze.board_key(SFENS[4]))
            for sfen in ("board b - 0", "board z - 1", "board b - +2", "board b - 1 extra"):
                with self.assertRaises(ValueError):
                    freeze.board_key(sfen)


if __name__ == "__main__":
    unittest.main()
