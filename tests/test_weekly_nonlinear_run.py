"""Exercise real process ownership without engines or private training data."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import weekly_nonlinear_run as w


class WeeklySupervisorTests(unittest.TestCase):
    def test_launch_scans_keep_bound_allowlist_and_propagate_unobserved_process_failure(self):
        import weekly_nonlinear_preflight as preflight
        allowlist = {"schema": "sekirei.weekly-nonlinear-heavy-process-allowlist.v1",
                     "status": "frozen-before-preflight", "services": []}
        clear = {"conflicts": [], "excluded_preexisting_services": []}
        with patch.object(preflight, "observe_processes", return_value=clear) as observe:
            self.assertEqual(w.require_clear_processes(allowlist), [clear, clear])
            self.assertEqual(observe.call_count, 2)
            self.assertTrue(all(call.args == (allowlist,) for call in observe.call_args_list))
        with patch.object(preflight, "observe_processes", side_effect=ValueError("unobserved process")):
            with self.assertRaisesRegex(ValueError, "unobserved process"):
                w.require_clear_processes(allowlist)

    def test_success_receipt_binds_actual_log_and_reaped_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            log = root / "child.log"
            argv = [sys.executable, "-c", "print('actual child output')"]
            result = w.run_child(argv, log, dict(os.environ), 5, root)
            self.assertEqual(result["command"], argv)
            self.assertEqual(result["returncode"], 0)
            self.assertEqual(result["group_empty_scans"], [True, True])
            self.assertTrue(result["waited"] and result["reaped"])
            self.assertFalse(result["timed_out"])
            self.assertEqual(result["log"], w.physical_ref(log))
            receipt = json.loads(Path(str(log) + ".child-outcome.json").read_bytes())
            self.assertEqual(receipt["outcome"], result)
            self.assertEqual(log.read_bytes(), b"actual child output\n")

    def test_nonzero_is_recorded_as_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            result = w.run_child([sys.executable, "-c", "raise SystemExit(7)"],
                                 root / "failed.log", dict(os.environ), 5, root)
            self.assertEqual(result["returncode"], 7)
            self.assertTrue(result["reaped"])
            self.assertEqual(result["group_empty_scans"], [True, True])

    def test_timeout_reaps_and_saves_failed_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            log = root / "timeout.log"
            with self.assertRaises(subprocess.TimeoutExpired) as caught:
                w.run_child([sys.executable, "-c", "import time; time.sleep(30)"],
                            log, dict(os.environ), 0.1, root)
            result = caught.exception.child_outcome
            self.assertTrue(result["timed_out"] and result["reaped"] and result["waited"])
            self.assertEqual(result["group_empty_scans"], [True, True])
            self.assertFalse(w.group_alive(result["pgid"]))
            receipt = json.loads(Path(str(log) + ".child-outcome.json").read_bytes())
            self.assertEqual(receipt["status"], "failed")
            self.assertEqual(receipt["outcome"], result)

    def test_existing_log_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            log = root / "existing.log"
            log.write_bytes(b"preserved")
            with self.assertRaises(FileExistsError):
                w.run_child([sys.executable, "-c", "raise SystemExit(0)"],
                            log, dict(os.environ), 5, root)
            self.assertEqual(log.read_bytes(), b"preserved")

    def test_sigterm_parent_cancels_reaps_and_records_child_before_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            code = """
import os, signal, sys, threading
from pathlib import Path
import weekly_nonlinear_run as w
root = Path(sys.argv[1])
timer = threading.Timer(0.2, lambda: os.kill(os.getpid(), signal.SIGTERM))
timer.start()
try:
    w.run_child([sys.executable, '-c', 'import time; time.sleep(30)'],
                root/'cancel.log', dict(os.environ), 10, root)
except BaseException as error:
    assert type(error).__name__ == 'TrainingCancelled', repr(error)
    assert error.signal == 15, repr(error)
else:
    raise AssertionError('cancelled parent returned success')
timer.join()
"""
            scripts = str(Path(w.__file__).resolve().parent)
            result = subprocess.run([sys.executable, "-B", "-c", code, str(root)],
                                    env=dict(os.environ, PYTHONPATH=scripts),
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            receipt = json.loads((root / "cancel.log.child-outcome.json").read_bytes())
            self.assertEqual(receipt["status"], "failed")
            self.assertEqual(receipt["signal"], 15)
            outcome = receipt["outcome"]
            self.assertTrue(outcome["waited"] and outcome["reaped"])
            self.assertEqual(outcome["group_empty_scans"], [True, True])
            self.assertFalse(w.group_alive(outcome["pgid"]))

    def test_immutable_map_and_canonical_policy_reject_changed_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "source.txt"
            path.write_bytes(b"old")
            expected = {str(path): w.info(path)}
            self.assertEqual(w.verify_map(expected), expected)
            path.write_bytes(b"new")
            with self.assertRaises(ValueError):
                w.verify_map(expected)
            alias = root / "alias.txt"
            alias.symlink_to(path)
            with self.assertRaises(ValueError):
                w.physical_ref(alias)

    def test_strict_json_rejects_duplicate_keys_and_nonfinite(self):
        for raw in (b'{"value":1,"value":2}', b'{"value":NaN}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                w.strict_json(raw)

    def test_training_argv_contains_each_full_reference(self):
        ref = {"path": "/private/recipe.json", "bytes": 17, "sha256": "a" * 64}
        recipe = {name: {"path": "/private/" + name, "bytes": 23, "sha256": "b" * 64}
                  for name in ("source_binding", "manifest", "reference03", "positions", "labels")}
        argv = w.command(ref, recipe, "/private/bin/train", Path("/private/output"))
        self.assertEqual(len(argv), 40)
        self.assertEqual(argv[:2], ["/private/bin/train", "train-paired-nonlinear"])
        options = dict(zip(argv[2::2], argv[3::2]))
        self.assertEqual(options["--recipe-sha256"], "a" * 64)
        self.assertEqual(options["--source-binding"], "/private/source_binding")
        self.assertEqual(options["--output"], "/private/output")


if __name__ == "__main__":
    unittest.main()
