"""Small public synthetic labels/core rows; no real dataset, inference or search."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import weekly_static_residuals as d


GATE, _ = d.proof.load_numeric_gate(d.proof.REPO / "scripts/white_view_build_contract.py")


def jsonl(values):
    return ("\n".join(json.dumps(v, allow_nan=False) for v in values) + "\n").encode()


def core_row(index, sfen, native):
    material = GATE.material(sfen)
    row = {"index": index, "native_core_cp": native, "nearest_core_cp": material,
        "native_quantized_float_cp": float(native), "nearest_quantized_float_cp": float(material),
        "material_cp": material, "ft_prefixes_checked": True, "all_preclamp_intermediates_finite": True}
    for role in ("candidate", "reference"):
        for operation, count in (("l2_products", 16384), ("l2_additions", 16384),
                                 ("output_products", 32), ("output_additions", 32)):
            row[role + "_" + operation] = count
    return row


def fixture(split="train"):
    # Four distinct public technical boards, including both sides to move.
    public = json.loads(d.proof.FIXTURES.read_bytes())
    chosen, keys = [], set()
    for row in public:
        key = " ".join(row["sfen"].split(" ")[:3])
        if key not in keys:
            chosen.append(row["sfen"]); keys.add(key)
        if len(chosen) == 4:
            break
    sfens = []
    for i, sfen in enumerate(chosen):
        fields = sfen.split(" ")
        fields[1] = "b" if i % 2 == 0 else "w"
        fields[3] = str((16, 32, 96, 192)[i])
        sfens.append(" ".join(fields))
    positions, labels, core = [], [], []
    teachers, errors = (-301, 0, 300, 1200), (1, -2, 3, -4)
    games = [{"game_id": "a" * 64, "pack_sha256": "c" * 64, "split": split},
             {"game_id": "b" * 64, "pack_sha256": "d" * 64, "split": split}]
    for i, sfen in enumerate(sfens):
        positions.append({"schema_version": 1, "sfen": sfen,
            "source": {"kind": "gensfen-pack", "path": games[i // 2]["game_id"], "ply": int(sfen.split(" ")[3])},
            "tags": {"side_to_move": "black" if i % 2 == 0 else "white", "phase": "middlegame"}})
        labels.append({"sfen": sfen, "score_cp": teachers[i], "label_depth": 0, "teacher_identity": GATE.TEACHER})
        core.append(core_row(i, sfen, teachers[i] + errors[i]))
    return {"games": games}, positions, labels, core


def analyze(data, old=None, split="train"):
    manifest, positions, labels, core = data
    return d.analyze_split(manifest, jsonl(positions), jsonl(labels), jsonl(core),
                           set() if old is None else old, split, len(positions), GATE)


class StaticResidualTests(unittest.TestCase):
    def test_cli_failure_cannot_write_to_existing_preserved_input_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preserved = root / "preserved-input"
            preserved.mkdir()
            sentinel = preserved / "original.txt"
            sentinel.write_bytes(b"keep original input")
            proof_file = root / "not-a-proof.json"
            proof_file.write_bytes(b"{}")
            result = subprocess.run([sys.executable, "-B", str(Path(d.__file__).resolve()),
                "--proof", str(proof_file), "--expected-proof-sha256", "0" * 64,
                "--numeric-build-contract", str(d.proof.REPO / "scripts/white_view_build_contract.py"),
                "--output", str(preserved)], capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"external technical proof SHA differs", result.stderr)
            self.assertEqual(sorted(p.name for p in preserved.iterdir()), ["original.txt"])
            self.assertEqual(sentinel.read_bytes(), b"keep original input")

    def test_exact_label_join_and_stm_signed_errors_for_both_sides(self):
        data = fixture()
        result = analyze(data)
        overall = result["overall"]
        self.assertEqual((overall["count"], overall["split_denominator"]), (4, 4))
        self.assertEqual(overall["absolute_error_sum_cp"], 10)
        self.assertEqual(overall["signed_error_sum_cp"], -2)
        self.assertEqual(overall["squared_error_sum_cp2"], 30)
        self.assertEqual((overall["mae_cp"], overall["mean_signed_error_cp"], overall["mean_squared_error_cp2"]),
                         (2.5, -0.5, 7.5))
        sides = result["groups"]["side_to_move"]
        self.assertEqual(sides["black"]["signed_error_sum_cp"], 4)
        self.assertEqual(sides["white"]["signed_error_sum_cp"], -6)
        # A cache is joined by exact SFEN, not by label row order.
        reversed_labels = deepcopy(data)
        reversed_labels[2].reverse()
        self.assertEqual(analyze(reversed_labels), result)

    def test_partition_boundaries_counts_and_empty_denominators(self):
        result = analyze(fixture())
        for groups in result["groups"].values():
            self.assertEqual(sum(v["count"] for v in groups.values()), 4)
            self.assertEqual(sum(v["squared_error_sum_cp2"] for v in groups.values()), 30)
            self.assertTrue(all(v["split_denominator"] == 4 for v in groups.values()))
        cp = result["groups"]["absolute_teacher_cp"]
        self.assertEqual(cp["[300,1000)"]["count"], 2)
        self.assertEqual(cp["[0,300)"]["count"], 1)
        self.assertEqual(cp["[10000,30000)"]["count"], 0)
        self.assertIsNone(cp["[10000,30000)"]["mae_cp"])
        self.assertEqual(d.band(31, d.PLY_BANDS), "[16,32)")
        self.assertEqual(d.band(32, d.PLY_BANDS), "[32,64)")
        with self.assertRaises(ValueError):
            d.band(30000, d.TEACHER_BANDS)

    def test_previous_board_overlap_ignores_ply_and_includes_stm(self):
        data = fixture()
        old = deepcopy(data[1][:1])
        old[0]["sfen"] = old[0]["sfen"].rsplit(" ", 1)[0] + " 200"
        keys = d.previous_train_boards(jsonl(old), 1, GATE)
        result = analyze(data, old=keys)
        groups = result["groups"]["previous_train_board_overlap"]
        self.assertEqual((groups["present"]["count"], groups["absent"]["count"]), (1, 3))
        opposite = data[1][0]["sfen"].split(" ")
        opposite[1] = "w"
        self.assertNotIn(d.board_key(" ".join(opposite), GATE), keys)
        holdout = fixture("holdout")
        with self.assertRaisesRegex(ValueError, "holdout overlaps"):
            analyze(holdout, old=keys, split="holdout")
        with self.assertRaises(ValueError):
            d.previous_train_boards(jsonl(old + old), 2, GATE)

    def test_fixed_top_error_concentration_and_zero_total(self):
        values = d.concentration(list(range(1, 101)))
        self.assertEqual([v["count"] for v in values], [1, 5, 10, 20])
        self.assertEqual(values[0]["squared_error_sum_cp2"], 10000)
        self.assertEqual(values[1]["squared_error_sum_cp2"], sum(i * i for i in range(96, 101)))
        self.assertEqual(values[1]["train_squared_error_sum_cp2"], sum(i * i for i in range(1, 101)))
        self.assertEqual([v["count"] for v in d.concentration([1, 1, 1])], [1, 1, 1, 1])
        self.assertTrue(all(v["squared_error_share"] is None for v in d.concentration([0, 0])))

    def test_missing_duplicate_labels_wrong_game_and_typed_teacher_rejected(self):
        for mutation in ("missing_label", "duplicate_label", "bool_teacher", "mate_teacher", "wrong_split", "wrong_tags"):
            data = deepcopy(fixture())
            if mutation == "missing_label": data[2].pop()
            if mutation == "duplicate_label": data[2][1] = deepcopy(data[2][0])
            if mutation == "bool_teacher": data[2][0]["score_cp"] = True
            if mutation == "mate_teacher": data[2][0]["score_cp"] = 30000
            if mutation == "wrong_split": data[0]["games"][0]["split"] = "holdout"
            if mutation == "wrong_tags": data[1][1]["tags"]["side_to_move"] = "black"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                analyze(data)

    def test_bad_raw_core_order_count_domain_operations_and_bridge_rejected(self):
        for mutation in ("index", "missing", "extra", "bool_cp", "domain", "operations", "bridge", "extra_field"):
            data = deepcopy(fixture())
            core = data[3]
            if mutation == "index": core[1]["index"] = 0
            if mutation == "missing": core.pop()
            if mutation == "extra": core.append(deepcopy(core[-1]))
            if mutation == "bool_cp": core[0]["native_core_cp"] = True
            if mutation == "domain": core[0]["native_core_cp"] = 899000
            if mutation == "operations": core[0]["candidate_l2_products"] -= 1
            if mutation == "bridge": core[0]["native_quantized_float_cp"] += 2.0
            if mutation == "extra_field": core[0]["ignored"] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                analyze(data)

    def test_nan_and_duplicate_json_keys_are_rejected(self):
        manifest, positions, labels, core = fixture()
        for raw in (jsonl(core).replace(b'"native_quantized_float_cp": -300.0', b'"native_quantized_float_cp": NaN'),
                    jsonl(core).replace(b'"index": 0', b'"index": 0, "index": 0')):
            with self.assertRaises(ValueError):
                d.analyze_split(manifest, jsonl(positions), jsonl(labels), raw, set(), "train", 4, GATE)

    def test_public_summary_contains_only_aggregates_and_no_positions_or_paths(self):
        summaries = {"train": analyze(fixture()), "holdout": analyze(fixture("holdout"), split="holdout")}
        public = d.public_projection("white-view-diverse-games-seed42-e3-v1", summaries)
        raw = json.dumps(public)
        for secret in ("sfen", "/home/", '"path"', "game_id"):
            self.assertNotIn(secret, raw)
        self.assertFalse(public["adoption_claimed"])
        summaries["train"]["positions"] = ["private"]
        with self.assertRaises(ValueError):
            d.public_projection("white-view-diverse-games-seed42-e3-v1", summaries)

    def test_external_proof_pin_fails_before_any_physical_proof_consumer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "proof.json"
            path.write_text('{"schema":"synthetic"}')
            args = SimpleNamespace(proof=path, expected_proof_sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "external technical proof SHA"):
                d.run(args, {"signal": None})

    def test_malformed_proof_and_untyped_success_flags_cannot_enter_consumer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "proof.json"
            cases = (b'{"status":', json.dumps({"schema": "sekirei.weekly-nonlinear-model-technical-proof.v1",
                "status": "complete", "model_adopted": False, "adoption_claimed": 0, "final_used": False}).encode())
            for raw in cases:
                path.write_bytes(raw)
                args = SimpleNamespace(proof=path, expected_proof_sha256=d.proof.physical_ref(path)["sha256"],
                    numeric_build_contract=d.proof.REPO / "scripts/white_view_build_contract.py", output=root / "output")
                with patch.object(d, "validate_source_binding") as consumer, self.assertRaises(ValueError):
                    d.run(args, {"signal": None})
                consumer.assert_not_called()
                self.assertFalse(args.output.exists())

    def test_dedicated_complete_consumer_failure_never_aggregates_or_publishes(self):
        with tempfile.TemporaryDirectory(dir=Path.home()) as directory:
            root = Path(directory).resolve()
            proofdir, trainer, dataset = root / "proof", root / "trainer", root / "dataset"
            for directory in (proofdir, trainer, dataset): directory.mkdir()
            def file_ref(name, raw=b"{}"):
                path = root / name
                path.write_bytes(raw)
                return d.proof.physical_ref(path)
            recipe = file_ref("recipe.json")
            completed, engine = file_ref("completed.json"), file_ref("engine.json")
            identity = file_ref("identity.json")
            native = file_ref("trainer/native.bin", b"synthetic")
            manifest = file_ref("dataset/manifest.json")
            allow = file_ref("allow.json", json.dumps({"schema": "sekirei.weekly-nonlinear-heavy-process-allowlist.v1",
                "status": "frozen-before-preflight", "services": []}).encode())
            body = {"schema": "sekirei.weekly-nonlinear-model-technical-proof.v1", "status": "complete",
                "model_adopted": False, "adoption_claimed": False, "final_used": False, "inputs_before": {},
                "recipe": recipe, "training_completion": completed, "engine_manifest": engine,
                "engine_identity": identity, "native03": native,
                "lock_records": [{"path": str(root / (str(i) + ".lock")), "exclusive": True,
                    "nonblocking": True, "acquired": True} for i in range(5)]}
            path = proofdir / "technical-proof.json"
            path.write_text(json.dumps(body))
            args = SimpleNamespace(proof=path, expected_proof_sha256=d.proof.physical_ref(path)["sha256"],
                numeric_build_contract=d.proof.REPO / "scripts/white_view_build_contract.py", output=root / "diagnostics")
            bound = {"recipe": {"manifest": manifest}, "build": {"source_root": str(trainer / "source")},
                "parent": {"process_evidence": {"allowlist": allow}}}
            with patch.object(d, "validate_source_binding", return_value=bound), \
                 patch.object(d.proof, "require_clear_processes", return_value=[]), \
                 patch.object(d.proof, "validate_technical_proof", side_effect=ValueError("bad complete proof")) as consumer, \
                 patch.object(d, "analyze_split") as aggregate, patch("builtins.print") as output:
                with self.assertRaisesRegex(ValueError, "bad complete proof"):
                    d.run(args, {"signal": None})
                consumer.assert_called_once()
                aggregate.assert_not_called()
                output.assert_not_called()
            self.assertFalse(args.output.exists())


if __name__ == "__main__":
    unittest.main()
