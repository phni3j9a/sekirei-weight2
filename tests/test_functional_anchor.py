"""Public synthetic fixtures only; no real dataset, core, engine or training."""
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import functional_anchor as anchor
from diagnose_weights import verified_split


CORPUS = "a" * 64
PACK = "b" * 64
ORIGINAL = "external:suisho11beta-1m-pack:" + CORPUS
SFENS = [f"4k4/9/9/9/9/9/9/9/4K4 {side} {hand} 16"
         for hand in ("P", "2P") for side in ("b", "w")]


def game_id(split):
    for index in range(10000):
        value = hashlib.sha256(f"public-functional-anchor-{index}".encode()).hexdigest()
        digest = hashlib.sha256(f"first-weight-v1:{value}".encode()).hexdigest()
        actual = "holdout" if int(digest[:16], 16) % 10 == 0 else "train"
        if actual == split:
            return value
    raise AssertionError("no synthetic game ID")


def jsonl(rows):
    # Noncanonical bytes deliberately exercise untouched-byte preservation.
    return "".join(json.dumps(r, ensure_ascii=False) + " \r\n" for r in rows).encode()


def fixture():
    files, games = {}, []
    for index, split in enumerate(("train", "holdout")):
        gid = game_id(split)
        games.append({"game_id": gid, "game_index": index, "pack_sha256": PACK, "split": split})
        positions = [{"schema_version": 1, "sfen": sfen,
                      "source": {"kind": "gensfen-pack", "path": gid, "ply": 16},
                      "tags": {"side_to_move": "black" if sfen.split()[1] == "b" else "white"}}
                     for sfen in SFENS[index * 2:index * 2 + 2]]
        labels = [{"sfen": row["sfen"], "score_cp": 3 if i == 0 else -3,
                   "teacher_identity": ORIGINAL, "label_depth": 0,
                   "source_metadata": {"fixture": "合成", "row": i}}
                  for i, row in enumerate(positions)]
        # Cache and position orders need not coincide. Preserve both.
        files[f"{split}.positions.jsonl"] = jsonl(positions)
        files[f"{split}.labels.jsonl"] = jsonl(list(reversed(labels)))
    manifest = {"schema_version": 1, "teacher_identity": ORIGINAL,
                "source_corpus_manifest_sha256": CORPUS,
                "dependencies": {"cshogi": "synthetic", "numpy": "synthetic"},
                "games_per_pack": 400, "seed": "first-weight-v1",
                "split": anchor.SPLIT_RECIPE, "sampling": anchor.SAMPLING, "games": games,
                "independent_exclusions": {"games": 1000, "unique_positions": 88187,
                    "policy": anchor.EXCLUSION_POLICY, "corpus_canonical_sha256": "c" * 64,
                    "source_manifest_sha256": "d" * 64},
                "derivation": {"kind": "expanded-train-frozen-holdout-v1",
                    "input_sha256": {"synthetic-source-manifest": "e" * 64},
                    "frozen_holdout_game_count": 246, "reserved_new_holdout_game_count": 283,
                    "input_unchanged": True, "wall_seconds": 0.125},
                "limitations": ["public synthetic fixture"],
                "positions": {"train": 2, "holdout": 2}}
    return refresh(manifest, files)


def refresh(manifest, files):
    manifest = copy.deepcopy(manifest)
    manifest["files"] = {name: {"sha256": anchor.sha256_bytes(data), "bytes": len(data),
                               "count": len(data.splitlines())} for name, data in files.items()}
    data = json.dumps(manifest, ensure_ascii=False, indent=2).encode() + b"\n"
    return data, files, anchor.sha256_bytes(data)


class FunctionalAnchorTests(unittest.TestCase):
    def make_view(self):
        data, files, digest = fixture()
        spec = anchor.build_spec(data, files, expected_manifest_sha256=digest)
        view = anchor.derive_train_view(data, files, spec, expected_manifest_sha256=digest)
        return data, files, digest, view

    def validate(self, view, data, files, digest, spec_digest=None):
        return anchor.validate_train_view(view, data, files, expected_manifest_sha256=digest,
            expected_spec_sha256=spec_digest or anchor.spec_sha256(view["spec"]))

    def test_signed_ties_use_exact_nearest_even_oracle(self):
        for teacher in (-29999, -100, -5, -3, -1, 0, 1, 3, 5, 100, 29999):
            for material in (-101, -100, -3, -1, 0, 1, 3, 100, 101):
                with self.subTest(teacher=teacher, material=material):
                    self.assertEqual(anchor.blend_target(teacher, material),
                                     round(Fraction(teacher + material, 2)))
        self.assertEqual([anchor.blend_target(t, 0) for t in (3, 5, -3, -5)], [2, 2, -2, -2])

    def test_teacher_is_validated_before_blend_and_target_cannot_overflow(self):
        for teacher, material in ((30000, -30000), (-30000, 30000), (0, 60000), (0, -59999)):
            with self.subTest(teacher=teacher, material=material), self.assertRaises(ValueError):
                anchor.blend_target(teacher, material)
        for invalid in (True, False, 3.0, float("nan"), float("inf"), "3", None):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    anchor.blend_target(invalid, 0)
                with self.assertRaises(ValueError):
                    anchor.blend_target(0, invalid)

    def test_material_is_fixed_and_changes_sign_with_side_to_move(self):
        self.assertEqual([anchor.fixed_material_cp(sfen) for sfen in SFENS], [100, -100, 200, -200])
        for bad in (None, True, "not an SFEN", "9/9/9/9/9/9/9/9/9 b - 16"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                anchor.fixed_material_cp(bad)

    def test_canonical_spec_is_order_independent_and_has_no_output_hash_cycle(self):
        data, files, digest, view = self.make_view()
        spec = view["spec"]
        reordered = dict(reversed(list(spec.items())))
        self.assertEqual(anchor.spec_sha256(spec), anchor.spec_sha256(reordered))
        self.assertEqual(anchor.derived_identity(spec), anchor.IDENTITY_PREFIX + anchor.spec_sha256(spec))
        self.assertEqual(spec["original_manifest_sha256"], digest)
        self.assertEqual(spec["original_files"], json.loads(data)["files"])
        self.assertNotIn("output_files", spec)
        self.assertNotIn("output_manifest_sha256", spec)
        self.assertEqual(view["recipe"]["output_files"], view["manifest"]["files"])
        self.assertEqual(view["recipe"]["spec_sha256"], anchor.spec_sha256(spec))
        self.assertFalse(self.validate(view, data, files, digest)["real_generation_ready"])

    def test_original_bytes_orders_and_source_provenance_are_preserved(self):
        data, files, digest, view = self.make_view()
        for name in ("train.positions.jsonl", "holdout.positions.jsonl", "holdout.labels.jsonl"):
            self.assertEqual(view["files"][name], files[name])
        old = [json.loads(row) for row in files["train.labels.jsonl"].splitlines()]
        new = [json.loads(row) for row in view["files"]["train.labels.jsonl"].splitlines()]
        self.assertEqual([r["sfen"] for r in old], [r["sfen"] for r in new])
        self.assertEqual([r["score_cp"] for r in new], [-52, 52])
        for a, b in zip(old, new):
            self.assertEqual({k: v for k, v in a.items() if k not in ("score_cp", "teacher_identity")},
                             {k: v for k, v in b.items() if k not in ("score_cp", "teacher_identity")})
        original = json.loads(data)
        for name, value in original.items():
            if name not in ("teacher_identity", "files"):
                self.assertEqual(view["manifest"][name], value)
        self.assertEqual(view["spec"]["source_provenance"]["derivation"], original["derivation"])
        self.assertEqual(view["recipe"]["rounding"], {"count": 2, "half_integer_count": 2,
            "max_abs_error_twice_cp": 1, "sum_abs_error_twice_cp": 2, "sum_signed_error_twice_cp": 0})
        self.assertEqual(files, fixture()[1])  # Inputs were not mutated.

    def test_existing_guard_accepts_derived_train_rejects_derived_holdout_accepts_original(self):
        data, files, digest, view = self.make_view()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, derived = root / "original", root / "derived"
            original.mkdir(); derived.mkdir()
            (original / "manifest.json").write_bytes(data)
            (derived / "manifest.json").write_bytes(anchor.canonical_json_bytes(view["manifest"]) + b"\n")
            for name in files:
                (original / name).write_bytes(files[name])
                (derived / name).write_bytes(view["files"][name])
            self.assertEqual(verified_split(derived, "train")[1], [52, -52])
            with self.assertRaisesRegex(ValueError, "external label"):
                verified_split(derived, "holdout")
            self.assertEqual(verified_split(original, "holdout")[1], [3, -3])
        identity = anchor.derived_identity(view["spec"])
        self.assertEqual(view["manifest"]["teacher_identity"], identity)
        self.assertEqual(view["manifest"]["split_teacher_identities"], {"train": identity, "holdout": ORIGINAL})

    def test_cached_teacher_types_identity_depth_duplicates_missing_and_extra_are_rejected(self):
        for mode in ("bool", "float", "nan", "inf", "cap", "identity", "depth-bool", "depth", "duplicate", "missing", "extra"):
            with self.subTest(mode=mode):
                data, files, _ = fixture(); manifest = json.loads(data); files = dict(files)
                rows = [json.loads(row) for row in files["train.labels.jsonl"].splitlines()]
                values = {"bool": True, "float": 3.0, "nan": float("nan"), "inf": float("inf"), "cap": 30000}
                if mode in values: rows[0]["score_cp"] = values[mode]
                elif mode == "identity": rows[0]["teacher_identity"] = "external:another"
                elif mode == "depth-bool": rows[0]["label_depth"] = False
                elif mode == "depth": rows[0]["label_depth"] = 1
                elif mode == "duplicate": rows[1] = copy.deepcopy(rows[0])
                elif mode == "missing": rows.pop()
                else:
                    row = copy.deepcopy(rows[0]); row["sfen"] = SFENS[2]; rows.append(row)
                files["train.labels.jsonl"] = jsonl(rows)
                data, files, digest = refresh(manifest, files)
                with self.assertRaises(ValueError):
                    anchor.build_spec(data, files, expected_manifest_sha256=digest)

    def test_position_duplicates_overlap_wrong_source_and_bad_count_are_rejected(self):
        for mode in ("duplicate", "board-other-ply", "board-other-spelling", "overlap", "overlap-alias", "wrong-game", "count-bool", "file-count", "file-size-bool"):
            with self.subTest(mode=mode):
                data, files, _ = fixture(); manifest = json.loads(data); files = dict(files)
                rows = [json.loads(row) for row in files["train.positions.jsonl"].splitlines()]
                if mode == "duplicate": rows[1] = copy.deepcopy(rows[0])
                elif mode == "board-other-ply": rows[1]["sfen"] = rows[0]["sfen"].rsplit(" ", 1)[0] + " 99"
                elif mode == "board-other-spelling": rows[1]["sfen"] = rows[0]["sfen"].replace("/9/", "/45/", 1)
                elif mode == "overlap": rows[0]["sfen"] = SFENS[2]
                elif mode == "overlap-alias": rows[0]["sfen"] = SFENS[2].replace("/9/", "/45/", 1)
                elif mode == "wrong-game": rows[0]["source"]["path"] = game_id("holdout")
                files["train.positions.jsonl"] = jsonl(rows)
                if mode in ("board-other-ply", "board-other-spelling"):
                    labels = [json.loads(row) for row in files["train.labels.jsonl"].splitlines()]
                    labels[0]["sfen"] = rows[1]["sfen"]; files["train.labels.jsonl"] = jsonl(labels)
                if mode in ("overlap", "overlap-alias"):
                    labels = [json.loads(row) for row in files["train.labels.jsonl"].splitlines()]
                    labels[1]["sfen"] = rows[0]["sfen"]; files["train.labels.jsonl"] = jsonl(labels)
                data, files, digest = refresh(manifest, files); manifest = json.loads(data)
                if mode == "count-bool": manifest["positions"]["train"] = True
                elif mode == "file-count": manifest["files"]["train.labels.jsonl"]["count"] = 3
                elif mode == "file-size-bool": manifest["files"]["train.labels.jsonl"]["bytes"] = True
                data = anchor.canonical_json_bytes(manifest); digest = anchor.sha256_bytes(data)
                with self.assertRaises(ValueError):
                    anchor.build_spec(data, files, expected_manifest_sha256=digest)

    def test_original_hash_and_file_set_size_hash_cannot_be_rebound_silently(self):
        data, files, digest = fixture()
        for mode in ("manifest", "file", "missing", "extra", "size"):
            with self.subTest(mode=mode):
                altered, content = data, dict(files)
                if mode == "manifest": altered += b" "
                elif mode == "file": content["train.labels.jsonl"] += b" "
                elif mode == "missing": del content["holdout.labels.jsonl"]
                elif mode == "extra": content["unexpected.jsonl"] = b"{}\n"
                else:
                    manifest = json.loads(data); manifest["files"]["train.labels.jsonl"]["bytes"] += 1
                    altered = anchor.canonical_json_bytes(manifest)
                with self.assertRaises(ValueError):
                    anchor.build_spec(altered, content, expected_manifest_sha256=digest)

    def test_source_inventory_and_exclusion_derivation_provenance_are_mandatory(self):
        for mode in ("duplicate-game", "duplicate-location", "wrong-split", "missing-exclusions",
                     "bool-exclusion-count", "missing-derivation", "empty-input-hashes", "bad-input-hash",
                     "source-identity", "bad-cap", "schema-bool"):
            with self.subTest(mode=mode):
                data, files, _ = fixture(); manifest = json.loads(data)
                if mode == "duplicate-game":
                    game = copy.deepcopy(manifest["games"][0]); game["game_index"] = 2
                    manifest["games"].append(game)
                elif mode == "duplicate-location": manifest["games"][1]["game_index"] = 0
                elif mode == "wrong-split": manifest["games"][0]["split"] = "holdout"
                elif mode == "missing-exclusions": del manifest["independent_exclusions"]
                elif mode == "bool-exclusion-count": manifest["independent_exclusions"]["games"] = True
                elif mode == "missing-derivation": del manifest["derivation"]
                elif mode == "empty-input-hashes": manifest["derivation"]["input_sha256"] = {}
                elif mode == "bad-input-hash": manifest["derivation"]["input_sha256"]["synthetic-source-manifest"] = "INVALID"
                elif mode == "source-identity": manifest["teacher_identity"] = "external:another"
                elif mode == "bad-cap": manifest["games_per_pack"] = True
                else: manifest["schema_version"] = True
                altered = anchor.canonical_json_bytes(manifest)
                with self.assertRaises(ValueError):
                    anchor.build_spec(altered, files, expected_manifest_sha256=anchor.sha256_bytes(altered))

    def test_spec_changes_formula_material_identity_inputs_or_output_cycle_are_rejected(self):
        data, files, digest, view = self.make_view(); pinned = anchor.spec_sha256(view["spec"])
        for mode in ("ratio", "bool", "float", "ridge", "values", "source", "identity", "input-hash", "count", "output-cycle"):
            with self.subTest(mode=mode):
                changed = copy.deepcopy(view); spec = changed["spec"]
                if mode == "ratio": spec["formula"]["denominator"] = 3
                elif mode == "bool": spec["formula"]["teacher_numerator"] = True
                elif mode == "float": spec["formula"]["teacher_numerator"] = 1.0
                elif mode == "ridge": spec["material"]["kind"] = "ridge=1"
                elif mode == "values": spec["material"]["piece_values"][0] = 101
                elif mode == "source": spec["source_provenance"]["derivation"]["input_sha256"]["synthetic-source-manifest"] = "f" * 64
                elif mode == "identity": spec["original_teacher_identity"] = "external:another"
                elif mode == "input-hash": spec["original_files"]["train.labels.jsonl"]["sha256"] = "f" * 64
                elif mode == "count": spec["positions"]["train"] = True
                else: spec["output_files"] = copy.deepcopy(view["manifest"]["files"])
                with self.assertRaises(ValueError):
                    self.validate(changed, data, files, digest, pinned)
                with self.assertRaises(ValueError):
                    anchor.derive_train_view(data, files, spec, expected_manifest_sha256=digest)

    def test_changed_view_rejects_even_coherently_rehashed_order_and_false_readiness(self):
        data, files, digest, view = self.make_view()
        for mode in ("root", "holdout-id", "label-order", "position-order", "target-bool", "target-float", "target-range", "holdout-byte", "recipe-hash", "ready", "core-proof", "nonfinite"):
            with self.subTest(mode=mode):
                changed = copy.deepcopy(view)
                if mode == "root": changed["manifest"]["teacher_identity"] = ORIGINAL
                elif mode == "holdout-id": changed["manifest"]["split_teacher_identities"]["holdout"] = changed["manifest"]["teacher_identity"]
                elif mode in ("label-order", "position-order"):
                    name = "train.labels.jsonl" if mode == "label-order" else "train.positions.jsonl"
                    changed["files"][name] = b"\n".join(reversed(changed["files"][name].splitlines())) + b"\n"
                    changed["manifest"]["files"][name].update(sha256=anchor.sha256_bytes(changed["files"][name]), bytes=len(changed["files"][name]))
                elif mode in ("target-bool", "target-float", "target-range"):
                    name = "train.labels.jsonl"
                    rows = [json.loads(row) for row in changed["files"][name].splitlines()]
                    rows[0]["score_cp"] = {"target-bool": True, "target-float": -52.0, "target-range": 30000}[mode]
                    changed["files"][name] = jsonl(rows)
                    changed["manifest"]["files"][name].update(sha256=anchor.sha256_bytes(changed["files"][name]), bytes=len(changed["files"][name]))
                elif mode == "holdout-byte": changed["files"]["holdout.labels.jsonl"] += b" "
                elif mode == "recipe-hash": changed["recipe"]["output_manifest_sha256"] = "0" * 64
                elif mode == "ready": changed["recipe"]["real_generation_ready"] = True
                elif mode == "core-proof": changed["recipe"]["mandatory_preflight"]["status"] = "verified"
                else: changed["recipe"]["rounding"]["extra"] = float("nan")
                with self.assertRaises(ValueError):
                    self.validate(changed, data, files, digest)

    def test_duplicate_json_keys_and_nonfinite_source_metadata_are_rejected(self):
        data, files, _ = fixture()
        invalid = [b'{"schema_version":1,"schema_version":1}',
                   data.replace(b'"wall_seconds": 0.125', b'"wall_seconds": NaN'),
                   data.replace(b'"wall_seconds": 0.125', b'"wall_seconds": 1e999')]
        for altered in invalid:
            with self.subTest(altered=altered[:50]), self.assertRaises(ValueError):
                anchor.build_spec(altered, files, expected_manifest_sha256=anchor.sha256_bytes(altered))

    def test_canonical_json_does_not_coerce_key_or_container_types(self):
        for invalid in ({1: "one"}, {"metadata": {True: "true"}}, {"array": (1, 2)}, {"nan": float("nan")}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                anchor.canonical_json_bytes(invalid)


if __name__ == "__main__":
    unittest.main()
