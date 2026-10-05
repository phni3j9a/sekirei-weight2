"""Public synthetic rows and tiny children; no model, corpus or engine runs."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import weekly_nonlinear_proof as p


GATE, _ = p.load_numeric_gate(p.REPO / "scripts/white_view_build_contract.py")


class Memory:
    def __init__(self, raw):
        self.raw = raw

    def read(self, reference):
        value = self.raw[reference["path"]]
        GATE.fullref(reference)
        if not GATE.exact(GATE.digest(value), p.small(reference)):
            raise ValueError("test input changed")
        return value

    def json(self, reference):
        return GATE.strict_json(self.read(reference))


def ref(path, raw):
    return {"path": path, **GATE.digest(raw)}


def incremental_result():
    result = {key: True for key in (
        "accumulator_refresh_equal", "parent_restoration_equal", "observed_float_intermediates_finite",
        "all_preclamp_intermediates_finite", "normal_score_domain_verified", "native_structure_verified",
        "native_loaded_fullbytes_equal")}
    result.update(positions_checked=8185, fixtures=15, walks=16, search_walks=8,
        captures=1, promotions=1, drops=1, undos=1, null_undos=1, max_material_difference_cp=200,
        incremental_refresh_error_cp=0, mxcsr_control="9fc0", material_bound_enabled=False,
        max_abs_core_cp=100, max_abs_quantized_float_cp=100.5, maximum_stored_float_core_bridge_cp=0.5,
        l2_products_checked=8185*16384, l2_additions_checked=8185*16384,
        output_products_checked=8185*32, output_additions_checked=8185*32,
        native_q_l1_upper=200.0, native_absolute_cp_upper=500.0)
    return result


class ProofNumericalTests(unittest.TestCase):
    def test_engine_public_relocation_binds_same_bytes_and_preserves_historical_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            old, current = root / "retired-public-worktree", root / "current-public-worktree"
            (current / "scripts").mkdir(parents=True)
            public = current / "scripts" / "source.py"
            public.write_bytes(b"frozen public source\n")
            private = root / "runtime-source.rs"
            private.write_bytes(b"frozen private source\n")
            historical = str(old / "scripts" / "source.py")
            original = {historical: p.info(public), str(private): p.info(private)}
            manifest = {"inputs_before": deepcopy(original), "inputs_after": deepcopy(original)}
            manifest_path, identity_path = root / "engine-manifest.json", root / "engine-identity.json"
            manifest_path.write_text(json.dumps(manifest))
            identity_path.write_text("{}")
            manifest_ref, identity_ref = p.physical_ref(manifest_path), p.physical_ref(identity_path)
            with patch.object(p, "REPO", current), patch.object(p, "OLD_PUBLIC_ROOT", old), \
                 patch.object(GATE.build, "immutable_inputs_for_runtime", return_value=original):
                inventory, relocation = p.engine_input_inventory(manifest, {}, manifest_ref, identity_ref, GATE)
                self.assertFalse(old.exists())
                self.assertEqual(inventory, {str(public): p.info(public), str(private): p.info(private)})
                self.assertEqual(relocation["original_inventory"], original)
                self.assertEqual(relocation["relocations"], {historical: p.physical_ref(public)})
                self.assertTrue(relocation["historical_manifest_preserved"])
                self.assertEqual(p.physical_ref(manifest_path), manifest_ref)
                receipt_path = root / "relocation.json"
                receipt_ref = p.write_new(receipt_path, relocation)
                expected = dict(inventory)
                p.collect_refs(expected, [manifest_ref, identity_ref, receipt_ref])
                reader = p.PhysicalBytes(expected, gate=GATE, physical_ref=p.physical_ref)
                self.assertEqual(p.validate_engine_relocation(receipt_ref, reader, manifest,
                    identity_ref, manifest_ref, GATE), inventory)
                changed = deepcopy(relocation)
                changed["current_public_root"] = str(root)
                changed_ref = p.write_new(root / "wrong-relocation.json", changed)
                changed_inputs = dict(expected)
                p.collect_refs(changed_inputs, changed_ref)
                with self.assertRaises(ValueError):
                    p.validate_engine_relocation(changed_ref, p.PhysicalBytes(changed_inputs,
                        gate=GATE, physical_ref=p.physical_ref), manifest, identity_ref, manifest_ref, GATE)
                public.write_bytes(b"modified public src\n")
                with self.assertRaises(ValueError):
                    p.validate_engine_relocation(receipt_ref, reader, manifest, identity_ref, manifest_ref, GATE)
                public.unlink()
                with self.assertRaises((FileNotFoundError, ValueError)):
                    p.validate_engine_relocation(receipt_ref, reader, manifest, identity_ref, manifest_ref, GATE)
                self.assertEqual(p.physical_ref(manifest_path), manifest_ref)

    def test_engine_relocation_rejects_unfrozen_public_or_missing_private_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            old, current = root / "retired", root / "current"
            current.mkdir()
            path = str(old / "unfrozen.rs")
            record = GATE.digest(b"frozen")
            manifest = {"inputs_before": {}, "inputs_after": {path: record}}
            manifest_ref, identity_ref = ref(str(root / "manifest"), b"manifest"), ref(str(root / "identity"), b"identity")
            with patch.object(p, "REPO", current), patch.object(p, "OLD_PUBLIC_ROOT", old), \
                 patch.object(GATE.build, "immutable_inputs_for_runtime", return_value={path: record}):
                with self.assertRaises(ValueError):
                    p.engine_input_inventory(manifest, {}, manifest_ref, identity_ref, GATE)
            private = str(root / "missing-private.rs")
            original = {private: record}
            with patch.object(p, "REPO", current), patch.object(p, "OLD_PUBLIC_ROOT", old), \
                 patch.object(GATE.build, "immutable_inputs_for_runtime", return_value=original):
                inventory, relocation = p.engine_input_inventory(
                    {"inputs_before": original, "inputs_after": original}, {}, manifest_ref, identity_ref, GATE)
                self.assertEqual(inventory, original)
                self.assertEqual(relocation["relocations"], {})
                with self.assertRaises((FileNotFoundError, ValueError)):
                    p.verify_map(inventory)

    def test_public_sources_only_sole_barrier_delta(self):
        self.assertEqual(hashlib.sha256(p.FIXTURES.read_bytes()).hexdigest(), p.FIXTURE_SHA)
        self.assertEqual(p.COUNTS, {"train": 112681, "holdout": 5895, "fixtures": 15})
        self.assertEqual(sum(p.COUNTS.values()), 118591)
        for kind, (name, sha) in p.PROOF_SOURCES.items():
            raw = (p.PROOF_DIR / name).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), sha)
            activated = p.enabled_probe(raw, kind)
            p.validate_activation(activated, kind)
            if kind == "native_contract":
                self.assertEqual(activated, raw)
            else:
                self.assertEqual(activated.replace(b"const PROTOTYPE_ONLY:bool=false;",
                    b"const PROTOTYPE_ONLY:bool=true;", 1), raw)
            with self.assertRaises(ValueError):
                p.enabled_probe(raw + b"\n", kind)
            with self.assertRaises(ValueError):
                p.validate_activation(activated + b"\n", kind)

    def test_incremental_exact_scope_operations_and_native_bridges(self):
        valid = incremental_result()
        bounds = {"q_l1_upper": 200.0, "absolute_cp_upper": 500.0}
        self.assertEqual(p.incremental_summary(json.dumps(valid).encode(), bounds, GATE), valid)
        cases = {"positions_checked": 8184, "fixtures": 14, "walks": 15, "search_walks": 7,
            "l2_products_checked": 8185*16384-1, "output_additions_checked": True,
            "native_loaded_fullbytes_equal": False, "parent_restoration_equal": False,
            "captures": 0, "mxcsr_control": "1fc0", "maximum_stored_float_core_bridge_cp": 1.001,
            "material_bound_enabled": True, "native_absolute_cp_upper": 501.0}
        for key, value in cases.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                changed = deepcopy(valid)
                changed[key] = value
                p.incremental_summary(json.dumps(changed).encode(), bounds, GATE)
        with self.assertRaises(ValueError):
            p.incremental_summary(b'{"fixtures":15,"fixtures":15}', bounds, GATE)

    def test_core_every_row_index_operation_finiteness_and_bound(self):
        fixture = json.loads(p.FIXTURES.read_bytes())[0]
        sfen, material = fixture["sfen"], fixture["expected_cp_stm"]
        row = {"index": 0, "native_core_cp": material, "nearest_core_cp": material,
            "native_quantized_float_cp": float(material), "nearest_quantized_float_cp": float(material),
            "material_cp": material, "ft_prefixes_checked": True, "all_preclamp_intermediates_finite": True}
        for role in ("candidate", "reference"):
            for operation, count in (("l2_products", 16384), ("l2_additions", 16384),
                                     ("output_products", 32), ("output_additions", 32)):
                row[role + "_" + operation] = count
        bounds = {"q_l1_upper": 200.0, "absolute_cp_upper": float(abs(material)+500)}
        stderr = ("float_subnormal_policy=x86-ftz-daz; mxcsr_control=0x9fc0; mxcsr_status=0x20\n"
            "nonlinear_native_structure=verified; mode=" + GATE.MODE + "; epochs=3; q_l1_upper=200.0; absolute_cp_upper="
            + str(bounds["absolute_cp_upper"]) + "\ncomplete count=1; nonlinear absolute STM; both stored-native float bridges, FT prefixes, every L2/output product/add finite before clamp\n").encode()
        result = p.core_summary(json.dumps(row).encode()+b"\n", stderr, [sfen], bounds, GATE)
        self.assertEqual(result, {"count": 1, "maximum_material_difference_cp": 0, "maximum_stored_float_core_bridge_cp": 0.0})
        for key, value in (("index", 1), ("candidate_l2_additions", 16383), ("ft_prefixes_checked", False),
            ("nearest_core_cp", material+1), ("native_quantized_float_cp", float(material+2))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                changed = deepcopy(row)
                changed[key] = value
                p.core_summary(json.dumps(changed).encode(), stderr, [sfen], bounds, GATE)
        with self.assertRaises(ValueError):
            p.core_summary(b"", stderr, [sfen], bounds, GATE)
        with self.assertRaises(ValueError):
            p.core_summary(json.dumps(row).encode(), stderr.replace(b"0x9fc0", b"0x1fc0"), [sfen], bounds, GATE)

    def test_sidecar_loader_format_and_model_context_sha_fnv(self):
        # A serialization-only byte fixture; this is not a trained NNUE state.
        native = b"SEKIRW03" + bytes(GATE.NATIVE_BYTES-8)
        completed = {"mode": "weekly-public-fixture-v1", "outputs": {
            "native03": ref("/fixture/model.bin", native), "checkpoint": ref("/fixture/state.json", b"state")},
            "recipe": ref("/fixture/recipe.json", b"recipe"), "source_binding": ref("/fixture/binding.json", b"binding")}
        completion_ref = ref("/fixture/completion.json", b"completion")
        value = p.sidecar(native, completion_ref, completed, GATE)
        self.assertEqual(value["format"], "sekirei-nnue-output-v1")
        self.assertEqual(value["nnue_output"], "absolute")
        self.assertEqual(value["checkpoint_hash"], GATE.fnv(native))
        self.assertEqual(value["weight_sha256"], hashlib.sha256(native).hexdigest())
        self.assertEqual(value["training_completion"], completion_ref)
        self.assertEqual(value["mode"], completed["mode"])
        self.assertFalse(value["model_adopted"])
        with self.assertRaises(ValueError):
            p.sidecar(b"SEKIRW01" + native[8:], completion_ref, completed, GATE)

    def test_compile_command_pins_white_core_and_abort_abi(self):
        compiler = {"path": "/fixed/rustc"}
        rlib = {"path": "/fixed/deps/libsekirei_core.rlib"}
        source = {"path": "/new/src/probe.rs"}
        self.assertEqual(p.compile_command(compiler, rlib, source, Path("/new/probe")),
            ["/fixed/rustc", "--edition=2024", "-C", "target-cpu=x86-64-v3", "-C", "opt-level=3",
             "-C", "panic=abort", "--extern", "sekirei_core=/fixed/deps/libsekirei_core.rlib",
             "-L", "dependency=/fixed/deps", "/new/src/probe.rs", "-o", "/new/probe"])

    def test_weekly_joins_preserve_order_and_reject_leakage_or_wrong_game(self):
        fixtures = json.loads(p.FIXTURES.read_bytes())
        sfens = [" ".join(f["sfen"].split(" ")[:3])+" 16" for f in fixtures[:2]]
        raw, refs = {}, {"manifest": None, "fixtures": None}
        manifest = {"schema_version": 1, "teacher_identity": GATE.TEACHER, "positions": {"train": 1, "holdout": 1},
            "derivation": {"kind": "whole-pack-hash-ranked-frozen-holdout-v2"}, "files": {},
            "games": [{"game_id": "train-game", "split": "train"}, {"game_id": "holdout-game", "split": "holdout"}]}
        for split, sfen in zip(("train", "holdout"), sfens):
            position = {"schema_version": 1, "sfen": sfen, "source": {"kind": "gensfen-pack", "path": split+"-game", "ply": 16},
                "tags": {"phase": "middlegame", "side_to_move": "black" if sfen.split(" ")[1]=="b" else "white"}}
            label = {"sfen": sfen, "score_cp": 1, "label_depth": 0, "teacher_identity": GATE.TEACHER}
            for role, value in (("positions", position), ("labels", label)):
                key, path = split+"_"+role, "/fixture/"+split+"."+role
                raw[path] = json.dumps(value).encode()+b"\n"
                refs[key] = ref(path, raw[path])
                manifest["files"][split+"."+role+".jsonl"] = p.small(refs[key])
        for key, value in (("manifest", manifest), ("fixtures", fixtures)):
            path = "/fixture/"+key
            raw[path] = json.dumps(value).encode()
            refs[key] = ref(path, raw[path])
        with patch.object(p, "COUNTS", {"train": 1, "holdout": 1, "fixtures": 15}):
            rows, _ = p.dataset_rows(Memory(raw), refs, GATE)
            self.assertEqual(rows["train"], [sfens[0]])
            changed = deepcopy(raw)
            row = json.loads(changed[refs["train_positions"]["path"]])
            row["source"]["path"] = "holdout-game"
            path = refs["train_positions"]["path"]
            changed[path] = json.dumps(row).encode()+b"\n"
            bad_refs = deepcopy(refs)
            bad_refs["train_positions"] = ref(path, changed[path])
            bad_manifest = deepcopy(manifest)
            bad_manifest["files"]["train.positions.jsonl"] = p.small(bad_refs["train_positions"])
            changed[refs["manifest"]["path"]] = json.dumps(bad_manifest).encode()
            bad_refs["manifest"] = ref(refs["manifest"]["path"], changed[refs["manifest"]["path"]])
            with self.assertRaises(ValueError):
                p.dataset_rows(Memory(changed), bad_refs, GATE)

    def test_weekly_outer_reparses_raw_proofs_and_rejects_identity_or_payload_changes(self):
        # The numerical checkpoint and full-size dataset have separate tests.
        # Here their bounded fixture adapters allow one row per scope while the
        # real outer checks source activation, ABI, lifecycles and raw payloads.
        raw = {}
        def add(name, value):
            data = value if type(value) is bytes else json.dumps(value).encode()
            path = "/fixture/"+name
            raw[path] = data
            return ref(path, data)
        fixture = json.loads(p.FIXTURES.read_bytes())[0]
        sfen, material = fixture["sfen"], fixture["expected_cp_stm"]
        bounds = {"q_l1_upper": 200.0, "absolute_cp_upper": float(abs(material)+500)}
        native = b"SEKIRW03"+bytes(GATE.NATIVE_BYTES-8)
        recipe_ref, source_binding_ref = add("recipe", b"recipe"), add("source-binding", b"binding")
        recipe = {"mode": "weekly-public-fixture-v1", "manifest": add("manifest", b"manifest"),
            "source_binding": source_binding_ref, "reference03": add("reference03", b"reference"),
            "positions": add("positions", b"positions"), "labels": add("labels", b"labels")}
        dataset = {split+"."+role+".jsonl": add(split+"-"+role, b"split fixture")
                   for split in ("train", "holdout") for role in ("positions", "labels")}
        sb = {"selected_plan": add("plan", b"plan"), "training_build": add("build", b"build"),
              "engine_build": add("engine-manifest", b"engine"), "dataset_inputs": dataset,
              "training_binary": add("trainer", b"trainer")}
        completed = {"mode": recipe["mode"], "recipe": recipe_ref, "source_binding": source_binding_ref,
            "outputs": {"native03": add("model.bin", native), "checkpoint": add("checkpoint", b"state")},
            "initialized_io_parent": add("initialized-parent", {})}
        completion_ref = add("completion", b"completion")
        identity_ref = add("engine-identity", {})
        allow_ref = add("allowlist", {"schema": "sekirei.weekly-nonlinear-heavy-process-allowlist.v1",
            "status": "frozen-before-preflight", "services": []})
        context = {"recipe": recipe_ref, "source_binding": source_binding_ref, "manifest": recipe["manifest"],
            "reference03": recipe["reference03"], "positions": recipe["positions"], "labels": recipe["labels"], "source_files": {}}
        producer = p.physical_ref(Path(p.__file__).resolve())
        raw[producer["path"]] = Path(producer["path"]).read_bytes()
        public = p.physical_ref(p.FIXTURES)
        raw[public["path"]] = p.FIXTURES.read_bytes()
        activation = {"schema": "sekirei.weekly-nonlinear-proof-source-activation.v1", "status": "source-prepared",
            "mode": recipe["mode"], "training_completion": completion_ref, "sources": {}, "native_contract_model_mode": GATE.MODE,
            "candidate_identity_bound_by_outer_context": True, "compiled": False, "probe_started": False,
            "model_adopted": False, "final_used": False}
        for kind, (name, _) in p.PROOF_SOURCES.items():
            data = (p.PROOF_DIR/name).read_bytes()
            activation["sources"][kind] = {"prototype": add("prototype/"+name, data),
                "enabled": add("enabled/"+name, p.enabled_probe(data, kind)), "production_code_preserved": True,
                "prototype_only": False if kind != "native_contract" else None,
                "sole_delta": "PROTOTYPE_ONLY true -> false" if kind != "native_contract" else "none"}
        compiler, rlib = add("rustc", b"compiler"), add("deps/core.rlib", b"rlib")
        manifest = {"core_link": {"rlib": rlib}, "compiler_files": {compiler["path"]: p.small(compiler)},
                    "release_dependencies_after_probes": {rlib["path"]: p.small(rlib)}}
        engine_inventory = {reference["path"]: p.small(reference) for reference in (compiler, rlib)}
        relocation = {"schema": "sekirei.weekly-nonlinear-engine-public-relocation.v1",
            "status": "same-byte-public-inputs-bound", "engine_manifest": sb["engine_build"],
            "engine_identity": identity_ref, "old_public_root": str(p.OLD_PUBLIC_ROOT),
            "current_public_root": str(p.REPO), "relocations": {}, "original_inventory": engine_inventory,
            "current_inventory": engine_inventory, "historical_manifest_preserved": True}
        def execution(argv, stdin, stdout, stderr, number):
            outcome = {"command": argv, "pid": number, "pgid": number, "returncode": 0, "waited": True,
                "reaped": True, "timed_out": False, "group_empty_scans": [True, True], "log": stderr, "wall_seconds": 1.0}
            receipt = add("child"+str(number), {"schema": "sekirei.weekly-nonlinear-split-child.v1", "status": "observed-exit",
                "execution": outcome, "signal": None, "stdin_before": stdin, "stdin_after": stdin, "stdout": stdout})
            return outcome, receipt
        compilations = {}
        for index, kind in enumerate(("core", "incremental")):
            binary = add(kind+"-probe", b"binary"+kind.encode())
            sources = {"probe": activation["sources"][kind]["enabled"], "native_contract": activation["sources"]["native_contract"]["enabled"]}
            inp, out, err = add(kind+"-compile-in", b""), add(kind+"-compile-out", b""), add(kind+"-compile-err", b"warning")
            argv = p.compile_command(compiler, rlib, sources["probe"], Path(binary["path"]))
            process, receipt = execution(argv, inp, out, err, 10+index)
            compilations[kind] = {"schema": "sekirei.weekly-nonlinear-proof-compilation.v1", "status": "complete", "kind": kind,
                "engine_manifest": sb["engine_build"], "engine_identity": identity_ref, "compiler": compiler, "core_rlib": rlib,
                "release_dependencies_before": manifest["release_dependencies_after_probes"],
                "release_dependencies_after": manifest["release_dependencies_after_probes"], "sources": sources,
                "binary": binary, "command": argv, "execution": process, "stdin": inp, "stdout": out, "stderr": err,
                "child_receipt": receipt, "inputs_before": dict(manifest["release_dependencies_after_probes"]),
                "inputs_after": dict(manifest["release_dependencies_after_probes"])}
        row = {"index": 0, "native_core_cp": material, "nearest_core_cp": material, "native_quantized_float_cp": float(material),
            "nearest_quantized_float_cp": float(material), "material_cp": material, "ft_prefixes_checked": True,
            "all_preclamp_intermediates_finite": True}
        for role in ("candidate", "reference"):
            for operation, count in (("l2_products", 16384), ("l2_additions", 16384), ("output_products", 32), ("output_additions", 32)):
                row[role+"_"+operation] = count
        stderr = ("float_subnormal_policy=x86-ftz-daz; mxcsr_control=0x9fc0; mxcsr_status=0x20\n"
            "nonlinear_native_structure=verified; mode="+GATE.MODE+"; epochs=3; q_l1_upper=200.0; absolute_cp_upper="
            +str(bounds["absolute_cp_upper"])+"\ncomplete count=1; nonlinear absolute STM; both stored-native float bridges, FT prefixes, every L2/output product/add finite before clamp\n").encode()
        core = {}
        for index, split in enumerate(("train", "holdout", "fixtures")):
            inp, out, err = add(split+"-in", (sfen+"\n").encode()), add(split+"-out", json.dumps(row).encode()), add(split+"-err", stderr)
            argv = [compilations["core"]["binary"]["path"], completed["outputs"]["native03"]["path"], recipe["reference03"]["path"], "x86-ftz-daz"]
            process, receipt = execution(argv, inp, out, err, 20+index)
            core[split] = {"positions": public if split == "fixtures" else dataset[split+".positions.jsonl"],
                "stdin": inp, "stdout": out, "stderr": err, "execution": process, "child_receipt": receipt,
                "inputs_before": {inp["path"]: p.small(inp)}, "inputs_after": {inp["path"]: p.small(inp)},
                "count": 1, "maximum_material_difference_cp": 0, "maximum_stored_float_core_bridge_cp": 0.0}
        tsv, inp, out, err = add("fixtures-tsv", (str(material)+"\t"+sfen+"\n").encode()), add("incremental-in", b""), add("incremental-out", incremental_result()), add("incremental-err", b"")
        argv = [compilations["incremental"]["binary"]["path"], completed["outputs"]["native03"]["path"], recipe["reference03"]["path"], tsv["path"]]
        process, receipt = execution(argv, inp, out, err, 30)
        incremental = {"fixture_tsv": tsv, "stdin": inp, "stdout": out, "stderr": err, "execution": process,
            "child_receipt": receipt, "result": incremental_result(), "inputs_before": {inp["path"]: p.small(inp)},
            "inputs_after": {inp["path"]: p.small(inp)}}
        value = {"schema": "sekirei.weekly-nonlinear-model-technical-proof.v1", "status": "complete", "mode": recipe["mode"],
            "producer_source": producer, "training_completion": completion_ref, "recipe": recipe_ref,
            "source_binding": source_binding_ref, "plan": sb["selected_plan"], "training_build": sb["training_build"],
            "engine_manifest": sb["engine_build"], "engine_identity": identity_ref, "context": context,
            "engine_input_relocation": add("engine-relocation", relocation),
            "native03": completed["outputs"]["native03"], "checkpoint": completed["outputs"]["checkpoint"], "reference03": recipe["reference03"],
            "metadata": add("sidecar", p.sidecar(native, completion_ref, completed, GATE)), "source_activation": add("activation", activation),
            "compilations": compilations, "counts": p.COUNTS, "total_core_rows": 118591, "core": core, "incremental": incremental,
            "native_functional_bounds": bounds, "inputs_before": {}, "inputs_after": {}, "original_inputs_before": {}, "original_inputs_after": {},
            "lock_records": [{"path": "/fixture/lock"+str(i), "exclusive": True, "nonblocking": True, "acquired": True} for i in range(5)],
            "process_scan_records_before": [{"conflicts": [], "excluded_preexisting_services": []}, {"conflicts": [], "excluded_preexisting_services": []}],
            "process_scan_records_after": [{"conflicts": [], "excluded_preexisting_services": []}, {"conflicts": [], "excluded_preexisting_services": []}],
            "source_unchanged": True, "build_unchanged": True, "core_fullrows_verified": True, "incremental_verified": True,
            "protected_material_ft_bits_preserved": True, "whole_board_ft_preservation_claimed": False,
            "universal_integer_bound_claimed": False, "native_bitexact_covariance_claimed": False,
            "resume_used": False, "shuffle_used": False, "final_used": False, "model_adopted": False, "adoption_claimed": False}
        inventory = {path: GATE.digest(data) for path, data in raw.items()}
        value.update(inputs_before=inventory, inputs_after=deepcopy(inventory),
                     original_inputs_before=inventory, original_inputs_after=deepcopy(inventory))
        reader = GATE.Bytes(raw, inventory)
        bound = {"recipe": recipe, "source_binding": sb, "parent": {"process_evidence": {"allowlist": allow_ref}}}
        with patch.object(p, "validate_completion", return_value=(context, native, bounds)), \
             patch.object(p, "validate_initialized"), \
             patch.object(p, "engine_input_inventory", return_value=(engine_inventory, relocation)), \
             patch.object(p, "dataset_rows", return_value=({split: [sfen] for split in p.COUNTS}, [fixture])):
            self.assertEqual(p.validate_technical_proof(value, reader, bound, completed, completion_ref, manifest, identity_ref, GATE), bounds)
            cases = (("mode", GATE.MODE), ("adoption_claimed", True), ("core_fullrows_verified", 1), ("extra", False))
            for key, changed_value in cases:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    changed = deepcopy(value)
                    changed[key] = changed_value
                    p.validate_technical_proof(changed, reader, bound, completed, completion_ref, manifest, identity_ref, GATE)
            changed = deepcopy(value)
            changed["core"]["train"]["maximum_material_difference_cp"] = 1
            with self.assertRaises(ValueError):
                p.validate_technical_proof(changed, reader, bound, completed, completion_ref, manifest, identity_ref, GATE)
            changed = deepcopy(value)
            changed["compilations"]["core"]["command"][0] = "/wrong/compiler"
            with self.assertRaises(ValueError):
                p.validate_technical_proof(changed, reader, bound, completed, completion_ref, manifest, identity_ref, GATE)
            changed = deepcopy(value)
            changed["original_inputs_before"].pop(rlib["path"])
            changed["original_inputs_after"] = deepcopy(changed["original_inputs_before"])
            with self.assertRaises(ValueError):
                p.validate_technical_proof(changed, reader, bound, completed, completion_ref, manifest, identity_ref, GATE)

    def test_scans_require_bound_service_identity_and_strict_observation(self):
        allowlist = {"schema": "sekirei.weekly-nonlinear-heavy-process-allowlist.v1",
            "status": "frozen-before-preflight", "services": []}
        scans = [{"conflicts": [], "excluded_preexisting_services": []} for _ in range(2)]
        p.validate_scans(scans, allowlist, GATE)
        changed = deepcopy(scans)
        changed[0]["excluded_preexisting_services"] = [{"pid": 10}]
        with self.assertRaises(ValueError):
            p.validate_scans(changed, allowlist, GATE)
        with self.assertRaises(ValueError):
            p.validate_scans([{"conflicts": []}, {"conflicts": []}], allowlist, GATE)
        service = {"pid": 10, "uid": 1000, "comm": "fixture-service", "starttime_ticks": 1234, "cmdline_sha256": None}
        allowlist["services"] = [service]
        scans[0]["excluded_preexisting_services"] = [deepcopy(service)]
        p.validate_scans(scans, allowlist, GATE)
        scans[0]["excluded_preexisting_services"][0]["starttime_ticks"] += 1
        with self.assertRaises(ValueError):
            p.validate_scans(scans, allowlist, GATE)


class SplitSupervisorTests(unittest.TestCase):
    def test_exclusive_writer_marks_only_created_outputs_as_owned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path, state = root/"record", {"owned": None}
            path.write_bytes(b"preserved")
            with self.assertRaises(FileExistsError):
                p.write_owned_json(path, {"status": "complete"}, state, "owned", path)
            self.assertIsNone(state["owned"])
            self.assertEqual(path.read_bytes(), b"preserved")
            fresh = root/"fresh"
            with patch.object(p.os, "fsync", side_effect=OSError("fixture disk failure")), self.assertRaises(OSError):
                p.write_owned_json(fresh, {"status": "complete"}, state, "owned", fresh)
            self.assertEqual(state["owned"], fresh)
            self.assertTrue(fresh.exists())

    def test_independent_streams_and_raw_stdin_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            inp, out, err = root/"in", root/"out", root/"err"
            inp.write_bytes(b"fixture input\n")
            argv = [sys.executable, "-c", "import sys;sys.stdout.buffer.write(sys.stdin.buffer.read());sys.stderr.write('diagnostic\\n')"]
            result = p.run_split(argv, inp, out, err, dict(os.environ), 5, root)
            GATE.lifecycle(result, argv, 5)
            self.assertEqual(out.read_bytes(), inp.read_bytes())
            self.assertEqual(err.read_bytes(), b"diagnostic\n")
            saved = json.loads(Path(str(err)+".child-outcome.json").read_bytes())
            self.assertEqual(saved["stdin_before"], saved["stdin_after"])
            self.assertEqual(saved["stdout"], p.physical_ref(out))
            self.assertEqual(saved["execution"], result)

    def test_nonzero_is_observed_and_consumer_rejects_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            inp = root/"in"
            inp.write_bytes(b"")
            result = p.run_split([sys.executable, "-c", "raise SystemExit(7)"], inp,
                root/"out", root/"err", dict(os.environ), 5, root)
            self.assertEqual(result["returncode"], 7)
            self.assertTrue(result["reaped"])
            with self.assertRaises(ValueError):
                GATE.lifecycle(result)

    def test_timeout_reaps_and_preserves_failed_raw_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            inp = root/"in"
            inp.write_bytes(b"")
            with self.assertRaises(subprocess.TimeoutExpired) as caught:
                p.run_split([sys.executable, "-c", "import time;time.sleep(30)"], inp,
                    root/"out", root/"err", dict(os.environ), 0.1, root)
            result = caught.exception.child_outcome
            self.assertTrue(result["timed_out"] and result["reaped"] and result["waited"])
            self.assertEqual(result["group_empty_scans"], [True, True])
            self.assertFalse(p.group_alive(result["pgid"]))
            saved = json.loads((root/"err.child-outcome.json").read_bytes())
            self.assertEqual(saved["status"], "failed")

    def test_changed_stdin_and_existing_outputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            inp = root/"in"
            inp.write_bytes(b"old")
            argv = [sys.executable, "-c", "from pathlib import Path;import sys;Path(sys.argv[1]).write_bytes(b'new')", str(inp)]
            with self.assertRaises(ValueError):
                p.run_split(argv, inp, root/"out", root/"err", dict(os.environ), 5, root)
            (root/"exists").write_bytes(b"preserved")
            with self.assertRaises(FileExistsError):
                p.run_split([sys.executable, "-c", "raise SystemExit(0)"], inp,
                    root/"exists", root/"err2", dict(os.environ), 5, root)
            self.assertEqual((root/"exists").read_bytes(), b"preserved")

    def test_sigterm_propagates_child_signal_and_reaps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            code = """
import os,signal,sys,threading
from pathlib import Path
import weekly_nonlinear_proof as p
root=Path(sys.argv[1]);(root/'in').write_bytes(b'')
timer=threading.Timer(0.2,lambda:os.kill(os.getpid(),signal.SIGTERM));timer.start()
try:
 p.run_split([sys.executable,'-c','import time;time.sleep(30)'],root/'in',root/'out',root/'err',dict(os.environ),10,root)
except BaseException as error:
 assert type(error).__name__=='TrainingCancelled',repr(error)
 assert error.child_signal==15
else:raise AssertionError('cancel returned success')
timer.join()
"""
            result = subprocess.run([sys.executable, "-B", "-c", code, str(root)],
                env=dict(os.environ, PYTHONPATH=str(Path(p.__file__).parent)),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            saved = json.loads((root/"err.child-outcome.json").read_bytes())
            self.assertEqual(saved["signal"], 15)
            self.assertEqual(saved["status"], "failed")
            self.assertTrue(saved["execution"]["reaped"])
            self.assertEqual(saved["execution"]["group_empty_scans"], [True, True])

    def test_cancellation_during_spawn_keeps_handle_and_reaps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            inp = root/"in"
            inp.write_bytes(b"")
            original = subprocess.Popen
            def spawn_and_cancel(*args, **kwargs):
                child = original(*args, **kwargs)
                os.kill(os.getpid(), signal.SIGTERM)
                return child
            with patch.object(p.subprocess, "Popen", side_effect=spawn_and_cancel), self.assertRaises(ValueError) as caught:
                p.run_split([sys.executable, "-c", "import time;time.sleep(30)"], inp,
                    root/"out", root/"err", dict(os.environ), 5, root)
            self.assertEqual(caught.exception.child_signal, 15)
            outcome = caught.exception.child_outcome
            self.assertTrue(outcome["waited"] and outcome["reaped"])
            self.assertEqual(outcome["group_empty_scans"], [True, True])
            saved = json.loads((root/"err.child-outcome.json").read_bytes())
            self.assertEqual(saved["signal"], 15)
            self.assertEqual(saved["status"], "failed")


if __name__ == "__main__":
    unittest.main()
