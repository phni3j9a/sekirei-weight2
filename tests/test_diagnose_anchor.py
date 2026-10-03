"""Synthetic contract fixtures; no actual diagnosis subprocess or real data."""
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import diagnose_anchor as diag
import functional_anchor as anchor
from export_nearest import fnv1a
from test_functional_anchor import fixture


def contract():
    original_bytes, files, digest = fixture()
    spec = anchor.build_spec(original_bytes, files, expected_manifest_sha256=digest)
    view = anchor.derive_train_view(original_bytes, files, spec, expected_manifest_sha256=digest)
    original = anchor._json(original_bytes)
    binding = dict(diag.FIXED, schema=diag.BINDING_SCHEMA,
                   original_manifest_sha256=digest, spec_sha256=anchor.spec_sha256(spec),
                   trainer_build_manifest_sha256="1" * 64, train_script_sha256="2" * 64,
                   initial_weights_sha256=anchor.MATERIAL_INIT_SHA256,
                   initial_metadata_sha256="3" * 64, positions=original["positions"],
                   original_teacher_identity=original["teacher_identity"],
                   training_target_identity=anchor.derived_identity(spec))
    return {"diagnosis_binding": binding}, spec, original, view


def training_fixture():
    prereg, spec, original, view = contract()
    binding = prereg["diagnosis_binding"]
    root, dataset, init = Path("/private/train-run"), Path("/private/derived"), Path("/private/material.bin")
    build = {"binary": "/private/trainer/build/release/train", "binary_sha256": "4" * 64,
             "patch_sha256": "5" * 64, "source_main_sha256": "6" * 64,
             "upstream_commit": diag.LOCK["sources"]["sekirei"]["commit"],
             "rustflags": diag.LOCK["rustflags"]}
    initial = {"path": str(init), "metadata_path": str(init.with_suffix(".meta.json")),
               "sha256": binding["initial_weights_sha256"],
               "metadata_sha256": binding["initial_metadata_sha256"],
               "bytes": diag.NNUE_WEIGHT_BYTES, "checkpoint_hash": "a" * 16,
               "nnue_output": "absolute", "architecture": "INPUT=2420 L1=256 L2=32",
               "optimizer": "fresh Adam moments and step=0; inference parameters only"}
    argv = diag.training_argv(build["binary"], root, dataset, init, binding["training_target_identity"])
    metadata_paths = [str(root / f"checkpoints/weights.epoch{epoch}.meta.json") for epoch in (1, 2, 3)]
    n = binding["positions"]["train"]
    final_sha = "9" * 64
    run = {"schema_version": 1, "status": "complete", "returncode": 0,
           "initial_weights_unchanged": True, "trainer_build": build, "argv": argv,
           "dataset_manifest_sha256": "7" * 64, "positions_sha256": "8" * 64,
           "weight_sha256": final_sha, "weight_bytes": diag.NNUE_WEIGHT_BYTES,
           "script_sha256": binding["train_script_sha256"], "epoch_metadata": metadata_paths,
           "initialization": "inference weights; fresh Adam state", "initial_weights": initial,
           "positions": n, "epochs": 3, "learning_rate": .0001, "min_lr": 0.,
           "lr_schedule": "step-half", "lr_schedule_epochs": 3, "timeout_seconds": 1200,
           "float_subnormal_policy": "x86-ftz-daz"}
    epochs = {}
    for epoch in (1, 2, 3):
        metadata = {"epoch": epoch, "epochs": 3, "train_count": n, "valid_count": 0,
                    "cache_hits": n, "cache_misses": 0, "init_seed": 42, "split_seed": 42,
                    "shuffle_seed": 42, "lr_schedule_epochs": 3, "warmup_epochs": 0,
                    "label_depth": 0, "source_cap": 0, "split_hash": 0,
                    "nnue_output": "absolute", "teacher_eval": "external",
                    "teacher_identity": binding["training_target_identity"],
                    "lr_schedule": "StepHalf", "float_subnormal_policy": "x86-ftz-daz",
                    "architecture": "INPUT=2420 L1=256 L2=32", "cache_only": True,
                    "exclude_mate_labels": True, "side_balance": False,
                    "games_dir": None, "teacher_weights": None, "wdl_lambda": None,
                    "phase_weights": {}, "validation_ratio": 0., "teacher_score_cap": 30000.,
                    "search_target_weight": 1., "lr": diag.float32(.0001), "min_lr": 0.,
                    "positions": str(root / "positions.jsonl")}
        epochs[epoch] = {"metadata": metadata, "adam_step": epoch * n, "weight_sha256": final_sha}
    kwargs = {"derived_manifest_sha256": "7" * 64, "positions_sha256": "8" * 64,
              "final_sha256": final_sha, "final_bytes": diag.NNUE_WEIGHT_BYTES,
              "expected_argv": argv, "expected_metadata_paths": metadata_paths}
    return run, binding, view["manifest"], build, initial, epochs, kwargs


class AnchorDiagnosisBindingTests(unittest.TestCase):
    def test_exact_preregistered_split_identities(self):
        prereg, spec, original, view = contract()
        self.assertEqual(diag.validate_binding(prereg, spec, original, view["manifest"]),
                         prereg["diagnosis_binding"])

    def test_wrong_original_or_target_identity_rejected(self):
        for name in ("original_teacher_identity", "training_target_identity"):
            with self.subTest(name=name):
                prereg, spec, original, view = contract()
                prereg["diagnosis_binding"][name] = "external:wrong"
                with self.assertRaises(ValueError):
                    diag.validate_binding(prereg, spec, original, view["manifest"])

    def test_spec_or_original_hash_mismatch_rejected(self):
        for name in ("spec_sha256", "original_manifest_sha256"):
            with self.subTest(name=name):
                prereg, spec, original, view = contract()
                prereg["diagnosis_binding"][name] = "f" * 64
                with self.assertRaises(ValueError):
                    diag.validate_binding(prereg, spec, original, view["manifest"])

    def test_fitted_initial_or_recipe_change_rejected(self):
        for name, value in (("initial_weights_sha256", "f" * 64), ("epochs", 2),
                            ("lr_schedule", "constant"), ("selected_epoch", 1),
                            ("seed", True), ("shuffle_seed", 41), ("learning_rate", float("nan"))):
            with self.subTest(name=name):
                prereg, spec, original, view = contract()
                prereg["diagnosis_binding"][name] = value
                with self.assertRaises(ValueError):
                    diag.validate_binding(prereg, spec, original, view["manifest"])

    def test_split_counts_and_mixed_identity_are_bound(self):
        prereg, spec, original, view = contract()
        prereg["diagnosis_binding"]["positions"] = {"train": 2, "holdout": True}
        with self.assertRaises(ValueError):
            diag.validate_binding(prereg, spec, original, view["manifest"])
        prereg, spec, original, view = contract()
        view["manifest"]["split_teacher_identities"]["holdout"] = anchor.derived_identity(spec)
        with self.assertRaises(ValueError):
            diag.validate_binding(prereg, spec, original, view["manifest"])

    def test_original_holdout_used_in_diagnosis_argv(self):
        prereg, spec, original, _ = contract()
        command = diag.diagnosis_argv(Path("/private/trainer/build/train"), Path("/private/original"),
                                     Path("/private/run/checkpoints/weights.epoch3.bin"),
                                     original["teacher_identity"])
        self.assertEqual(command[0:2], ["/private/trainer/build/train", "diagnose-external"])
        self.assertEqual(command[3], "/private/original/holdout.positions.jsonl")
        self.assertEqual(command[5], "/private/original/holdout.labels.jsonl")
        self.assertEqual(command[7], original["teacher_identity"])
        self.assertNotIn(prereg["diagnosis_binding"]["training_target_identity"], command)
        self.assertEqual(command[-1], "/private/run/checkpoints/weights.epoch3.adam.json")

    def test_existing_checkpoint_guard_uses_D_and_actual_FNV(self):
        prereg, _, original, _ = contract()
        data = b"SEKIRW01" + b"synthetic-weight-content"
        meta = {"nnue_output": "absolute", "teacher_identity": prereg["diagnosis_binding"]["training_target_identity"],
                "checkpoint_hash": fnv1a(data)}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "weights.epoch3.bin"
            path.write_bytes(data)
            diag.verify_checkpoint_metadata(path, meta, meta["teacher_identity"])
            with self.assertRaises(ValueError):
                diag.verify_checkpoint_metadata(path, meta, original["teacher_identity"])
            for field, value in (("nnue_output", "residual"), ("checkpoint_hash", "0" * 16)):
                changed = dict(meta, **{field: value})
                with self.assertRaises(ValueError):
                    diag.verify_checkpoint_metadata(path, changed, meta["teacher_identity"])
            path.write_bytes(data + b"changed")
            with self.assertRaises(ValueError):
                diag.verify_checkpoint_metadata(path, meta, meta["teacher_identity"])


class AnchorTrainingContractTests(unittest.TestCase):
    def validate(self, fixture_data):
        run, binding, manifest, build, initial, epochs, kwargs = fixture_data
        return diag.validate_training(run, binding, manifest, build, initial, epochs, **kwargs)

    def test_complete_fixed_epoch3_contract(self):
        self.assertIsNone(self.validate(training_fixture()))

    def test_wrong_run_status_hash_or_recipe_rejected(self):
        for name, value in (("status", "timeout"), ("returncode", True), ("epochs", 2),
                            ("positions", 1), ("dataset_manifest_sha256", "f" * 64),
                            ("script_sha256", "f" * 64), ("initial_weights_unchanged", False),
                            ("weight_sha256", "f" * 64), ("lr_schedule", "constant")):
            with self.subTest(name=name):
                data = training_fixture()
                data[0][name] = value
                with self.assertRaises(ValueError):
                    self.validate(data)

    def test_training_argv_resume_or_holdout_leak_rejected(self):
        for change in (lambda a: a + ["--resume-adam", "/private/old.adam.json"],
                       lambda a: ["/private/original/holdout.labels.jsonl" if x.endswith("train.labels.jsonl") else x for x in a]):
            data = training_fixture()
            # argv is intentionally shared with expected_argv in the synthetic
            # fixture; replace it to model changed recorded training arguments.
            data[0]["argv"] = change(data[0]["argv"])
            with self.assertRaises(ValueError):
                self.validate(data)

    def test_wrong_epoch_teacher_counts_schedule_or_adam_rejected(self):
        for name, value in (("epoch", 2), ("teacher_identity", "external:original"),
                            ("train_count", 1), ("cache_misses", 1), ("cache_hits", True),
                            ("lr", .0001), ("lr_schedule", "Constant"), ("wdl_lambda", .5),
                            ("nnue_output", "residual"), ("float_subnormal_policy", "inherited")):
            with self.subTest(name=name):
                data = training_fixture()
                data[5][3]["metadata"][name] = value
                with self.assertRaises(ValueError):
                    self.validate(data)
        data = training_fixture()
        data[5][3]["adam_step"] += 1
        with self.assertRaises(ValueError):
            self.validate(data)

    def test_missing_checkpoint_final_and_initial_bindings_rejected(self):
        data = training_fixture()
        del data[5][1]
        with self.assertRaises(ValueError):
            self.validate(data)
        data = training_fixture()
        data[5][3]["weight_sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            self.validate(data)
        data = training_fixture()
        data[4]["metadata_sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            self.validate(data)

    def test_pinned_build_hash_commit_and_source_guards(self):
        _, binding, _, build, _, _, _ = training_fixture()
        hashes = dict(manifest_sha256="1" * 64, binary_sha256="4" * 64,
                      patch_sha256="5" * 64, main_sha256="6" * 64)
        diag.validate_build(build, binding, **hashes)
        for name in hashes:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    diag.validate_build(build, binding, **dict(hashes, **{name: "f" * 64}))
        for name, value in (("upstream_commit", "0" * 40), ("rustflags", "different")):
            with self.assertRaises(ValueError):
                diag.validate_build(dict(build, **{name: value}), binding, **hashes)

    def test_preregistered_helper_and_fixed_material_bindings(self):
        prereg, spec, _, _ = contract()
        actual = {"functional_anchor.py": "1" * 64, "material_init.py": anchor.MATERIAL_IMPLEMENTATION_SHA256,
                  "train_cpu.py": "2" * 64, "diagnose_weights.py": "3" * 64}
        prereg["source_helpers"] = dict(actual)
        diag.validate_source_helpers(prereg, actual, spec)
        for name in actual:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    diag.validate_source_helpers(prereg, dict(actual, **{name: "f" * 64}), spec)
        changed = dict(prereg, source_helpers={})
        with self.assertRaises(ValueError):
            diag.validate_source_helpers(changed, actual, spec)
        # Matching a changed declaration is insufficient: the fixed material
        # implementation itself is part of the spec identity.
        actual["material_init.py"] = "f" * 64
        prereg["source_helpers"]["material_init.py"] = "f" * 64
        with self.assertRaises(ValueError):
            diag.validate_source_helpers(prereg, actual, spec)


class AnchorOutputAndSummaryTests(unittest.TestCase):
    def test_dedicated_private_new_root(self):
        root = Path("/private/diagnostics")
        diag.validate_output_paths(root / "run1", root, [Path("/private/original"), Path("/private/train-run")],
                                   root_private=True, inside_git=False, output_exists=False)
        for flag in ("root_private", "inside_git", "output_exists"):
            flags = dict(root_private=True, inside_git=False, output_exists=False)
            flags[flag] = not flags[flag]
            with self.assertRaises(ValueError):
                diag.validate_output_paths(root / "run1", root, [], **flags)

    def test_ancestor_overlap_and_outside_output_rejected(self):
        root = Path("/private/diagnostics")
        for output, protected in ((root, []), (Path("/elsewhere/run"), []),
                                  (root / "run", [root / "existing-input.json"]),
                                  (root / "run", [Path("/private")])):
            with self.assertRaises(ValueError):
                diag.validate_output_paths(output, root, protected, root_private=True,
                                           inside_git=False, output_exists=False)

    def test_canonical_paths_reject_relative_parent_and_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp).resolve()
            self.assertEqual(diag.canonical_path(parent, exists=True), parent)
            for path in (Path("relative/run"), parent / "child/../run"):
                with self.assertRaises(ValueError):
                    diag.canonical_path(path)
            target = parent / "target"
            target.mkdir()
            alias = parent / "alias"
            alias.symlink_to(target, target_is_directory=True)
            with self.assertRaises(ValueError):
                diag.canonical_path(alias / "run")

    def test_output_git_errors_bare_and_worktree_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            for returncode, stdout, stderr in ((0, "true\n", ""), (0, "false\n", ""),
                                               (128, "", "fatal: dubious ownership")):
                check = Mock(returncode=returncode, stdout=stdout, stderr=stderr)
                with patch.object(diag.subprocess, "run", return_value=check):
                    with self.assertRaises(ValueError):
                        diag.output_guard(root / "new", root, [])
            check = Mock(returncode=128, stdout="", stderr="fatal: not a git repository")
            with patch.object(diag.subprocess, "run", return_value=check):
                diag.output_guard(root / "new", root, [])

    def rows(self):
        return [{"index": i, "teacher_cp_stm": t, "raw_float_cp": p + .25,
                 "quantized_float_cp": p + .1, "inference_cp": p, "material_cp": m}
                for i, (t, p, m) in enumerate(((100, 80, 200), (-100, -90, -200)))]

    def test_static_original_error_and_material_distribution(self):
        summary = diag.summarize_rows(self.rows(), [100, -100])
        self.assertEqual(summary["metrics"]["inference_cp"]["mae_cp"], 15)
        self.assertEqual(summary["prediction_minus_fixed_material"]["inference_cp"]["mean_cp"], -5)
        self.assertAlmostEqual(summary["core_bridge_delta"]["max_abs_cp"], .1)
        self.assertIn("neither blend loss nor 1M-node", summary["interpretation"])

    def test_result_order_original_cp_types_and_nonfinite_rejected(self):
        for name, value in (("index", True), ("index", 3), ("teacher_cp_stm", 50),
                            ("teacher_cp_stm", 100.), ("raw_float_cp", float("inf")),
                            ("quantized_float_cp", True), ("inference_cp", 80.),
                            ("material_cp", 200.)):
            with self.subTest(name=name):
                rows = self.rows()
                rows[0][name] = value
                with self.assertRaises(ValueError):
                    diag.summarize_rows(rows, [100, -100])
        with self.assertRaises(ValueError):
            diag.summarize_rows(list(reversed(self.rows())), [100, -100])

    def test_exact_bridge_threshold_and_missing_rows_rejected(self):
        rows = self.rows()
        rows[0]["quantized_float_cp"] = rows[0]["inference_cp"] + 1.002
        with self.assertRaises(ValueError):
            diag.summarize_rows(rows, [100, -100])
        with self.assertRaises(ValueError):
            diag.summarize_rows(rows[:-1], [100, -100])

    def test_cleanup_is_bounded_and_escalates_without_real_process(self):
        process = Mock(pid=123456)
        process.wait.side_effect = [subprocess.TimeoutExpired("synthetic", 5), 0]
        with patch.object(diag, "group_exists", side_effect=[True, True, False]), patch.object(diag.os, "killpg") as kill:
            diag.cleanup_group(process)
            self.assertEqual([call.args[1] for call in kill.call_args_list], [signal.SIGTERM, signal.SIGKILL])
            self.assertEqual([call.kwargs for call in process.wait.call_args_list], [{"timeout": 5}, {"timeout": 5}])

    def test_cleanup_race_with_disappearing_group_is_harmless(self):
        process = Mock(pid=123456)
        with patch.object(diag, "group_exists", side_effect=[True, False]), patch.object(diag.os, "killpg", side_effect=ProcessLookupError):
            diag.cleanup_group(process)

    def test_termination_handler_cancels_and_restores_without_real_signals(self):
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        for sig in (signal.SIGTERM, signal.SIGINT):
            with diag.termination_guard():
                with self.assertRaises(diag.DiagnosticCancelled):
                    signal.getsignal(sig)(sig, None)
                self.assertEqual(signal.getsignal(signal.SIGTERM), signal.SIG_IGN)
                self.assertEqual(signal.getsignal(signal.SIGINT), signal.SIG_IGN)
            self.assertEqual({item: signal.getsignal(item) for item in previous}, previous)

    def test_diagnose_cancellation_passes_through_inner_cleanup(self):
        process = Mock(pid=123456)
        def inner(_args, holder, resources):
            try:
                holder["process"] = process
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            finally:
                cleaned()
        cleaned = Mock()
        with patch.object(diag, "_diagnose", side_effect=inner), patch.object(diag, "cleanup_group") as cleanup:
            with self.assertRaises(diag.DiagnosticCancelled):
                diag.diagnose(Mock())
        cleaned.assert_called_once_with()
        cleanup.assert_called_once_with(process)

    def test_spawn_assignment_race_cleans_mock_child(self):
        process = Mock(pid=123456)
        def popen(*args, **kwargs):
            # No process is spawned. The pending cancellation is deferred until
            # the returned handle is captured, without blocking a child's mask.
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return process
        holder = {"process": None}
        with diag.termination_guard(), patch.object(diag.subprocess, "Popen", side_effect=popen), \
                patch.object(diag.signal, "pthread_sigmask") as masks, \
                patch.object(diag, "cleanup_group") as cleanup:
            with self.assertRaises(diag.DiagnosticCancelled):
                diag.spawn_process(["synthetic"], Path("/private/output"), Mock(), Mock(), holder)
            masks.assert_not_called()
        cleanup.assert_called_once_with(process)
        self.assertIs(holder["process"], process)

    def test_caller_holder_survives_cancellation_before_return_assignment(self):
        process = Mock(pid=123456)
        def inner(_args, holder, resources):
            # Simulate the cancellation after spawn has stored its handle and
            # returned, but before any separate caller local assignment.
            holder["process"] = process
            raise diag.DiagnosticCancelled("mock caller return/assignment race")
        with patch.object(diag, "_diagnose", side_effect=inner), patch.object(diag, "cleanup_group") as cleanup:
            with self.assertRaises(diag.DiagnosticCancelled):
                diag.diagnose(Mock())
        cleanup.assert_called_once_with(process)

    def test_owned_lock_releases_after_cancelled_child_cleanup(self):
        events = []
        process = Mock(pid=123456)
        def inner(_args, holder, resources):
            resources.callback(lambda: events.append("unlock"))
            holder["process"] = process
            raise diag.DiagnosticCancelled("mock cancellation")
        def cleanup(child):
            self.assertIs(child, process)
            self.assertNotIn("unlock", events)
            events.append("cleanup")
        with patch.object(diag, "_diagnose", side_effect=inner), patch.object(diag, "cleanup_group", side_effect=cleanup):
            with self.assertRaises(diag.DiagnosticCancelled):
                diag.diagnose(Mock())
        self.assertEqual(events, ["cleanup", "unlock"])


if __name__ == "__main__":
    unittest.main()
