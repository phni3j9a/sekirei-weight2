"""Public synthetic source-selection, leakage, and immutable-input fixtures."""
from argparse import Namespace
from contextlib import ExitStack, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import diverse_pack_dataset as diverse
import freeze_holdout_dataset as freeze
from diagnose_weights import verified_split


def pack_bytes(lengths):
    result = bytearray()
    for count in lengths:
        result.append(1)
        for _ in range(count):
            result.extend(struct.pack("<Hh", (2 << 7) | 1, 50))
        result.extend(b"\0\0\0")
    return bytes(result)


def row(board, ply=16, cp=50, index=0, pack="a" * 64):
    return {"sfen": f"{board} b - {ply}", "selected_move": "7g7f", "game_ply": ply,
            "ply_in_game": index, "teacher_eval_cp_stm": cp, "pack_sha256": pack,
            "side_to_move": "black"}


def split_rows(split, prefix, pack="a" * 64):
    for n in range(1000):
        rows = [row(f"{prefix}-{n}", pack=pack)]
        if diverse.split_for_game(diverse.game_identity(rows), "first-weight-v1") == split:
            return rows
    raise AssertionError("cannot find synthetic split")


def simple_frozen():
    held = split_rows("holdout", "public-held")
    gid = diverse.game_identity(held)
    return {"manifest": {"seed": "first-weight-v1"},
            "games": {gid: {"split": "holdout"}},
            "splits": {"holdout": {"keys": {diverse.board_key(held[0]["sfen"])}}}}, held


def span(rows, index=0):
    return ({"game_index": index, "rank": "b" * 64}, rows)


class SelectionTests(unittest.TestCase):
    def test_index_handles_both_headers_and_validates_truncation_and_result(self):
        regular = pack_bytes([2, 1])
        indexed = list(diverse.iter_game_spans(regular))
        self.assertEqual([s["positions"] for s in indexed], [2, 1])
        self.assertEqual(indexed[-1]["end"], len(regular))
        hcp = b"\0" + bytes(32) + struct.pack("<H", 16) + regular[1:12]
        self.assertEqual(list(diverse.iter_game_spans(hcp))[0]["positions"], 2)
        for bad in (b"\2", b"\0", regular[:-1], b"\1\x83\x01\0"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                list(diverse.iter_game_spans(bad))

    def test_rank_is_reproducible_not_a_prefix_and_is_independent_of_labels(self):
        data = pack_bytes([1] * 2000)
        selected, census = diverse.select_spans(data, "a" * 64, "fixed-seed", 500)
        changed = bytearray(data)
        for game in diverse.iter_game_spans(changed):
            struct.pack_into("<h", changed, game["start"] + 3, -1234)
        self.assertEqual(diverse.select_spans(changed, "a" * 64, "fixed-seed", 500), (selected, census))
        self.assertEqual(selected, diverse.select_spans(data, "a" * 64, "fixed-seed", 500)[0])
        self.assertEqual(len({s["game_index"] for s in selected}), 500)
        self.assertTrue(any(s["game_index"] >= 400 for s in selected))
        self.assertEqual(census["games"], 2000)
        self.assertEqual(census["prefix_400_positions"], 400)
        self.assertNotEqual(selected, diverse.select_spans(data, "a" * 64, "different-seed", 500)[0])
        for count in (True, 0, 1001, 2001):
            with self.assertRaises(ValueError):
                diverse.select_spans(data, "a" * 64, "seed", count)

    def test_row_filter_preserves_the_original_fixed_rule(self):
        rows = [row("too-early", 15), row("not-every-four", 16, index=1),
                row("mate", 20, cp=30000, index=4), row("independent", 24, index=8)]
        rows += [row(f"kept-{i}", 28 + 4 * i, cp=-100, index=12 + 4 * i) for i in range(40)]
        result = diverse.eligible_rows(rows, {diverse.board_key(rows[3]["sfen"])})
        self.assertEqual(len(result), 32)
        self.assertEqual(result[0]["sfen"], "kept-0 b - 28")

    def test_short_pack_uses_only_the_explicit_count_and_census_has_all_five_keys(self):
        data = pack_bytes([0] * 392)
        selected, census = diverse.select_spans(data, "a" * 64, "fixed-seed", 392)
        self.assertEqual(len(selected), 392)
        self.assertEqual(census, {"games": 392, "positions": 0, "prefix_400_positions": 0,
                                  "after_prefix_positions": 0, "selected_prefix_400_games": 392})
        with self.assertRaisesRegex(ValueError, "fewer games"):
            diverse.select_spans(data, "a" * 64, "fixed-seed", 500)

    def test_complete_game_and_board_separation_include_new_reserved_games(self):
        frozen, held = simple_frozen()
        training = split_rows("train", "first")
        another = split_rows("train", "second", pack="b" * 64)
        reserved = split_rows("holdout", "reserved", pack="b" * 64)
        # A training game encounters the frozen holdout board at another ply.
        training += [dict(held[0], sfen=held[0]["sfen"].rsplit(" ", 1)[0] + " 20", game_ply=20, ply_in_game=4)]
        # Complete-game split can change when moves are added: choose a suffix
        # until this synthetic full sequence retains its training split.
        for i in range(1000):
            training[0]["selected_move"] = f"fixture-{i}"
            if diverse.split_for_game(diverse.game_identity(training), "first-weight-v1") == "train":
                break
        games = [[span(training), span(held, 1), span(reserved, 2)],
                 [span(another), span(training, 3)]]
        kept, inventory, counts = diverse.select_train(games, frozen, set(), 2)
        self.assertEqual([r["sfen"] for r in kept], [training[0]["sfen"], another[0]["sfen"]])
        self.assertEqual(counts["duplicate_selected_games"], 1)
        self.assertEqual(counts["reserved_selected_games"], 2)
        self.assertEqual(counts["reserved_board_rows"], 1)
        train_ids = {r["game_id"] for r in kept}
        self.assertTrue(all(diverse.split_for_game(gid, "first-weight-v1") == "train" for gid in train_ids))
        self.assertFalse(train_ids & {g["game_id"] for g in inventory if g["split"] == "holdout"})

    def test_fixed_budget_truncates_last_game_without_changing_its_split(self):
        frozen, _ = simple_frozen()
        rows = [row(f"budget-{n}", 16 + 4 * n, index=4 * n) for n in range(4)]
        for i in range(1000):
            rows[0]["selected_move"] = f"fixture-{i}"
            if diverse.split_for_game(diverse.game_identity(rows), "first-weight-v1") == "train":
                break
        kept, _, _ = diverse.select_train([[span(rows)]], frozen, set(), 3)
        self.assertEqual(len(kept), 3)
        self.assertEqual(len({r["game_id"] for r in kept}), 1)
        with self.assertRaisesRegex(ValueError, "only 4 usable rows"):
            diverse.select_train([[span(rows)]], frozen, set(), 5)

    def test_ply_independent_duplicate_boards_do_not_consume_extra_budget(self):
        frozen, _ = simple_frozen()
        rows = [row("same", 16), row("same", 20, index=4), row("different", 24, index=8)]
        for i in range(1000):
            rows[0]["selected_move"] = f"fixture-{i}"
            if diverse.split_for_game(diverse.game_identity(rows), "first-weight-v1") == "train":
                break
        kept, _, counts = diverse.select_train([[span(rows)]], frozen, set(), 2)
        self.assertEqual([r["game_ply"] for r in kept], [16, 24])
        self.assertEqual(counts["duplicate_training_board_rows"], 1)

    def test_profile_rejects_extra_fields_boolean_aliases_duplicate_keys_and_wrong_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            expected = {"schema_version": 1, "packs": ["a" * 64]}
            for actual in (dict(expected, schema_version=True), dict(expected, extra="x")):
                path.write_text(json.dumps(actual))
                with self.assertRaises(ValueError):
                    diverse.verify_profile(path, diverse.sha256(path), expected)
            path.write_text('{"schema_version":1,"schema_version":1,"packs":[]}')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                diverse.verify_profile(path, diverse.sha256(path), expected)
            path.write_text(json.dumps(expected))
            self.assertEqual(diverse.verify_profile(path, diverse.sha256(path), expected), expected)
            with self.assertRaisesRegex(ValueError, "externally pinned"):
                diverse.verify_profile(path, "b" * 64, expected)

    def test_profile_binds_the_decoder_separation_helpers_and_configuration_bytes(self):
        inventory = diverse.producer_sources()
        required = {"scripts/diverse_pack_dataset.py", "scripts/audit_pack.py", "scripts/pack_dataset.py",
                    "scripts/freeze_holdout_dataset.py", "scripts/prepare.py", "scripts/benchmark.py",
                    "scripts/acquire_quest.py", "scripts/smoke.py", "config/toolchain.lock.json",
                    "config/audit-requirements.txt", "config/quest-corpus.json"}
        self.assertEqual(set(inventory), required)
        root = Path(diverse.__file__).resolve().parent.parent
        for name, info in inventory.items():
            self.assertEqual(info, {"sha256": diverse.sha256(root / name), "bytes": (root / name).stat().st_size})


class ProducerTests(unittest.TestCase):
    def setup_inputs(self, root):
        corpus, quest, frozen = root / "corpus", root / "quest", root / "frozen"
        corpus.mkdir(); quest.mkdir(); (quest / "games").mkdir(); frozen.mkdir()
        (quest / "manifest.json").write_text("{}")
        (quest / "games" / "public.csa").write_text("synthetic fixture; replay is mocked")
        paths = []
        for counts in ([1, 1, 1], [1, 2, 1]):
            data = pack_bytes(counts)
            path = corpus / (hashlib.sha256(data).hexdigest() + ".pack")
            path.write_bytes(data); paths.append(path)
        corpus_manifest = {"unique_files": [{"sha256": p.stem, "bytes": p.stat().st_size,
                                              "path": p.name} for p in paths]}
        source_manifest = corpus / "manifest.json"
        source_manifest.write_text(json.dumps(corpus_manifest))
        source_hash = diverse.sha256(source_manifest)
        teacher = f"external:suisho11beta-1m-pack:{source_hash}"
        old_train = split_rows("train", "original", pack=paths[0].stem)
        held = split_rows("holdout", "original-held", pack=paths[1].stem)
        games = [{"game_id": diverse.game_identity(rows), "game_index": i,
                  "pack_sha256": rows[0]["pack_sha256"], "split": split}
                 for i, (split, rows) in enumerate((('train', old_train), ('holdout', held)))]
        for split, rows in (("train", old_train), ("holdout", held)):
            positions = [{"schema_version": 1, "sfen": rows[0]["sfen"],
                          "source": {"kind": "gensfen-pack", "path": diverse.game_identity(rows), "ply": 16}}]
            labels = [{"sfen": rows[0]["sfen"], "score_cp": 50, "teacher_identity": teacher, "label_depth": 0}]
            for kind, values in (("positions", positions), ("labels", labels)):
                (frozen / f"{split}.{kind}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in values))
        dependencies = {"cshogi": "fixture", "numpy": "fixture"}
        exclusion = {"games": 1000, "unique_positions": 88187, "policy": freeze.EXCLUSION_POLICY,
                     "corpus_canonical_sha256": "c" * 64, "source_manifest_sha256": "d" * 64}
        manifest = {"schema_version": 1, "teacher_identity": teacher,
                    "source_corpus_manifest_sha256": source_hash, "dependencies": dependencies,
                    "seed": "first-weight-v1", "split": freeze.SPLIT_RECIPE, "sampling": freeze.SAMPLING,
                    "games_per_pack": 400, "games": games, "independent_exclusions": exclusion,
                    "positions": {"train": 1, "holdout": 1},
                    "files": {name: {"sha256": diverse.sha256(frozen / name), "bytes": (frozen / name).stat().st_size}
                              for name in sorted(freeze.FILES)}}
        (frozen / "manifest.json").write_text(json.dumps(manifest))
        args = Namespace(corpus_runtime=corpus, quest_runtime=quest, frozen_holdout_dataset=frozen,
                         expected_frozen_manifest_sha256=diverse.sha256(frozen / "manifest.json"),
                         expected_corpus_manifest_sha256=source_hash, selection_seed="fixture-selection",
                         games_per_pack=2, train_count=2, output=root / "output",
                         profile=root / "profile.json", expected_profile_sha256="")
        pack_selections = {p.stem: {"games": 3, "selected_games": 2} for p in paths}
        args.profile.write_text(json.dumps(diverse.selection_profile(args, freeze.load_dataset(frozen),
                                                                    corpus_manifest, source_hash,
                                                                    pack_selections)))
        args.expected_profile_sha256 = diverse.sha256(args.profile)
        return args, source_manifest, corpus_manifest, paths, dependencies, exclusion

    def producer_mocks(self, source_manifest, corpus_manifest, paths, dependencies, exclusion):
        stack = ExitStack()
        stack.enter_context(patch.object(diverse, "require_decoder", return_value=(None, None, dependencies)))
        stack.enter_context(patch.object(diverse, "corpus_files", return_value=(source_manifest, corpus_manifest, paths)))
        stack.enter_context(patch.object(diverse, "independent_position_exclusions", return_value=(set(), exclusion)))
        def decode(path, pack_hash, spans, *_):
            # This fixture checks producer/source contracts, not cshogi legality.
            return [(s, split_rows("train", f"{pack_hash}-{s['game_index']}", pack_hash)) for s in spans], "e" * 64
        stack.enter_context(patch.object(diverse, "decode_selected", side_effect=decode))
        return stack

    def test_producer_freezes_bytes_and_emits_exact_budget_and_full_input_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            args, *inputs = self.setup_inputs(Path(directory))
            old_hashes = freeze.load_dataset(args.frozen_holdout_dataset)["hashes"]
            with self.producer_mocks(*inputs), redirect_stdout(io.StringIO()):
                manifest = diverse.prepare(args)
            self.assertEqual(manifest["positions"], {"train": 2, "holdout": 1})
            self.assertEqual(manifest["derivation"]["profile_sha256"], args.expected_profile_sha256)
            self.assertEqual(manifest["derivation"]["kind"], "whole-pack-hash-ranked-frozen-holdout-v2")
            self.assertEqual(manifest["derivation"]["profile"]["kind"],
                             "whole-pack-hash-ranked-frozen-holdout-profile-v2")
            self.assertEqual(len(manifest["derivation"]["indexes"]), 2)
            for indexed in manifest["derivation"]["indexes"]:
                self.assertEqual(set(indexed["census"]), {"games", "positions", "prefix_400_positions",
                                                        "after_prefix_positions", "selected_prefix_400_games"})
                self.assertEqual(indexed["census"]["games"], 3)
                self.assertEqual(len(indexed["selected_spans"]), 2)
            self.assertEqual(len(verified_split(args.output, "train")[1]), 2)
            for kind in ("positions", "labels"):
                name = f"holdout.{kind}.jsonl"
                self.assertEqual((args.output / name).read_bytes(), (args.frozen_holdout_dataset / name).read_bytes())
            freeze.unchanged(old_hashes)
            self.assertTrue(all((args.output / name).stat().st_mode & 0o777 == 0o600
                                for name in [*freeze.FILES, "manifest.json"]))
            self.assertEqual(args.output.stat().st_mode & 0o777, 0o700)
            self.assertFalse(list(args.output.parent.glob(".diverse-pack-*")))

    def test_profile_constructor_requires_complete_explicit_typed_census_and_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, manifest, paths, *_ = self.setup_inputs(Path(directory))
            frozen = freeze.load_dataset(args.frozen_holdout_dataset)
            good = {p.stem: {"games": 3, "selected_games": 2} for p in paths}
            for mutation in ("missing", "extra", "boolgames", "boolcount", "cap", "insufficient", "extra-key"):
                choices = json.loads(json.dumps(good))
                if mutation == "missing":
                    choices.pop(paths[0].stem)
                elif mutation == "extra":
                    choices["f" * 64] = {"games": 3, "selected_games": 2}
                elif mutation == "boolgames":
                    choices[paths[0].stem]["games"] = True
                elif mutation == "boolcount":
                    choices[paths[0].stem]["selected_games"] = True
                elif mutation == "cap":
                    choices[paths[0].stem]["selected_games"] = 3
                elif mutation == "insufficient":
                    choices[paths[0].stem]["games"] = 1
                else:
                    choices[paths[0].stem]["extra"] = 0
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    diverse.selection_profile(args, frozen, manifest,
                                              args.expected_corpus_manifest_sha256, choices)

    def test_generation_rechecks_declared_census_count_membership_size_and_types(self):
        for mode in ("games", "count", "missing", "duplicate", "bytes", "bool", "old-kind"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                args, *inputs = self.setup_inputs(Path(directory))
                profile = json.loads(args.profile.read_bytes())
                if mode == "games":
                    profile["packs"][0]["games"] = 4
                elif mode == "count":
                    args.games_per_pack = profile["games_per_pack"] = 5
                    profile["packs"][0]["games"] = 5
                    profile["packs"][0]["selected_games"] = 5
                elif mode == "missing":
                    profile["packs"].pop()
                elif mode == "duplicate":
                    profile["packs"].append(profile["packs"][0])
                elif mode == "bytes":
                    profile["packs"][0]["bytes"] += 1
                elif mode == "bool":
                    profile["packs"][0]["selected_games"] = True
                else:
                    profile["kind"] = "whole-pack-hash-ranked-frozen-holdout-profile-v1"
                args.profile.write_text(json.dumps(profile))
                args.expected_profile_sha256 = diverse.sha256(args.profile)
                with self.producer_mocks(*inputs), redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                    diverse.prepare(args)
                self.assertFalse(args.output.exists())

    def test_generation_accepts_explicit_short_pack_count_without_implicit_minimum(self):
        with tempfile.TemporaryDirectory() as directory:
            args, *inputs = self.setup_inputs(Path(directory))
            profile = json.loads(args.profile.read_bytes())
            args.games_per_pack = profile["games_per_pack"] = 500
            profile["packs"][0]["selected_games"] = 3
            args.profile.write_text(json.dumps(profile))
            args.expected_profile_sha256 = diverse.sha256(args.profile)
            with self.producer_mocks(*inputs), redirect_stdout(io.StringIO()):
                manifest = diverse.prepare(args)
            indexes = {p["pack_sha256"]: p for p in manifest["derivation"]["indexes"]}
            declared = profile["packs"][0]
            self.assertEqual(len(indexes[declared["sha256"]]["selected_spans"]), 3)

    def test_pinned_profile_and_input_mutation_fail_before_output_creation(self):
        for mode in ("profile", "frozen", "corpus", "during"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                args, *inputs = self.setup_inputs(Path(directory))
                if mode == "profile":
                    args.profile.write_text("{}")
                elif mode == "frozen":
                    with (args.frozen_holdout_dataset / "train.labels.jsonl").open("a") as stream:
                        stream.write("{}\n")
                elif mode == "corpus":
                    inputs[0].write_text("{}")
                with self.producer_mocks(*inputs), redirect_stdout(io.StringIO()):
                    if mode == "during":
                        original = diverse.select_train
                        def mutate(*values):
                            result = original(*values)
                            inputs[2][0].write_bytes(b"changed")
                            return result
                        with patch.object(diverse, "select_train", side_effect=mutate), self.assertRaises(ValueError):
                            diverse.prepare(args)
                    else:
                        with self.assertRaises(ValueError):
                            diverse.prepare(args)
                self.assertFalse(args.output.exists())

    def test_git_existing_and_source_output_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, *_ = self.setup_inputs(root)
            for output in (args.frozen_holdout_dataset / "new", args.quest_runtime / "new"):
                args.output = output
                with self.assertRaises(ValueError):
                    diverse.prepare(args)
            repo = root / "repository"
            repo.mkdir(); subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
            args.output = repo / "dataset"
            with self.assertRaisesRegex(ValueError, "private"):
                diverse.prepare(args)

    def test_source_or_pool_mutation_during_manifest_publication_has_no_success_manifest(self):
        for mode in ("input", "pool"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                args, *inputs = self.setup_inputs(Path(directory))
                original = diverse.atomic_write_json
                def mutate(path, value):
                    original(path, value)
                    if mode == "input":
                        inputs[2][0].write_bytes(b"changed during manifest publication")
                    else:
                        (args.quest_runtime / "games" / "extra.csa").write_text("unexpected extra raw")
                with self.producer_mocks(*inputs), patch.object(diverse, "atomic_write_json", side_effect=mutate), \
                        redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                    diverse.prepare(args)
                self.assertFalse((args.output / "manifest.json").exists())
                self.assertTrue((args.output / "manifest.failed.json").is_file())


if __name__ == "__main__":
    unittest.main()
