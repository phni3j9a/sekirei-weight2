"""Small child lifecycle/log fixtures; no Cargo or private input is accessed."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import weekly_nonlinear_build as build


class BuildTests(unittest.TestCase):
    def child(self, root, process):
        return build.supervised_command(["public-fixture"], root / "child.log",
                                        {}, 12, root)

    def test_success_owns_session_reaps_and_records_hash_bound_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = Mock(pid=12345, returncode=0)
            process.wait.return_value = 0
            with patch.object(build.subprocess, "Popen", return_value=process) as popen, \
                    patch.object(build, "cleanup") as cleanup, \
                    patch.object(build, "group_alive", return_value=False):
                outcome = self.child(root, process)
            self.assertEqual(outcome["returncode"], 0)
            self.assertTrue(outcome["waited"] and outcome["reaped"])
            self.assertEqual(outcome["group_empty_scans"], [True, True])
            self.assertTrue(popen.call_args.kwargs["start_new_session"])
            cleanup.assert_called_once_with(process)
            saved = json.loads((root / "child.log.child-outcome.json").read_text())
            self.assertEqual(saved["status"], "observed-complete")
            self.assertEqual(saved["outcome"]["log"], build.ref(root / "child.log"))

    def test_timeout_failure_is_preserved_and_no_success_returned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = Mock(pid=12345, returncode=-15)
            process.wait.side_effect = subprocess.TimeoutExpired("public-fixture", 12)
            with patch.object(build.subprocess, "Popen", return_value=process), \
                    patch.object(build, "cleanup") as cleanup, \
                    patch.object(build, "group_alive", return_value=False):
                with self.assertRaises(subprocess.TimeoutExpired):
                    self.child(root, process)
            cleanup.assert_called_once_with(process)
            saved = json.loads((root / "child.log.child-outcome.json").read_text())
            self.assertEqual(saved["status"], "failed")
            self.assertTrue(saved["outcome"]["timed_out"])
            self.assertEqual(saved["outcome"]["returncode"], -15)

    def test_cancel_during_spawn_still_owns_handle_and_reaps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = Mock(pid=12345, returncode=-15)
            def spawn(*_args, **_kwargs):
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
                return process
            with patch.object(build.subprocess, "Popen", side_effect=spawn), \
                    patch.object(build, "cleanup") as cleanup, \
                    patch.object(build, "group_alive", return_value=False):
                with self.assertRaises(ValueError):
                    self.child(root, process)
            cleanup.assert_called_once_with(process)
            saved = json.loads((root / "child.log.child-outcome.json").read_text())
            self.assertEqual(saved["signal"], signal.SIGTERM)
            self.assertTrue(saved["outcome"]["reaped"])

    def test_nonzero_exit_and_surviving_group_fail(self):
        for code, alive in ((2, False), (0, True)):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                process = Mock(pid=12345, returncode=code)
                process.wait.return_value = code
                with patch.object(build.subprocess, "Popen", return_value=process), \
                        patch.object(build, "cleanup"), \
                        patch.object(build, "group_alive", return_value=alive):
                    with self.assertRaises(ValueError):
                        self.child(root, process)
                saved = json.loads((root / "child.log.child-outcome.json").read_text())
                self.assertEqual(saved["status"], "failed")

    def test_test_log_requires_actual_names_summary_and_categories(self):
        names = ["paired_nonlinear_" + name for name in
                 ("reader", "sha", "unique", "native", "adapter", "float_policy", "initialized", "weekly")]
        names += [f"paired_nonlinear_extra_{i}" for i in range(16)]
        good = "\n".join("test " + name + " ... ok" for name in names)
        good += "\ntest result: ok. 24 passed; 0 failed; 0 ignored;\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.log"
            path.write_text(good)
            self.assertEqual(build.parse_test_log(path)["passed"], 24)
            for bad in (good.replace("0 failed", "1 failed"),
                        good.replace("24 passed", "25 passed"),
                        good.replace("paired_nonlinear_weekly", "unrelated"),
                        good.replace(names[-1], names[-2])):
                path.write_text(bad)
                with self.assertRaises(ValueError):
                    build.parse_test_log(path)


if __name__ == "__main__":
    unittest.main()
