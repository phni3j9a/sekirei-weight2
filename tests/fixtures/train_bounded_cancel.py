#!/usr/bin/env python3
"""Public synthetic TERM regression fixture; prepared for parent execution only.

Run: python3 THIS_FILE --scripts /absolute/path/to/public/scripts
The launcher isolates the self-signalling test in a new-session child. All
dataset/build/helper/model preflights are mocked and no engine is started.
"""
import argparse
from contextlib import ExitStack
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def run_fixture(scripts):
    sys.path.insert(0, str(scripts))
    import train_bounded as bounded
    import prepare_bounded as preparation

    class ReturnedSupervisorCancellation(unittest.TestCase):
        def test_term_after_supervisor_return_reaches_cleanup_and_cancelled_receipt(self):
            previous_handler = signal.getsignal(signal.SIGTERM)
            previous_umask = os.umask(0o077)
            try:
                with tempfile.TemporaryDirectory(prefix="public-bounded-cancel-") as folder:
                    private = Path(folder).resolve()
                    runtime, dataset = private / "trainer", private / "dataset"
                    runtime.mkdir(mode=0o700); dataset.mkdir(mode=0o700)
                    positions = b'{"sfen":"public synthetic cancellation fixture"}\n'
                    (dataset / "train.positions.jsonl").write_bytes(positions)
                    (dataset / "manifest.json").write_text('{"public_fixture":true}\n')
                    initial_path = private / "initial.bin"
                    initial_path.write_bytes(b"public fixture; never loaded as a model")
                    initial = {"path": str(initial_path), "sha256": bounded.INITIAL_SHA256}
                    output, prereg_path = runtime / "run", private / "preregistration.json"
                    prereg = {"schema": bounded.SCHEMA, "status": "frozen-before-training",
                        "candidate": "bounded-material-residual-100cp-e3-v1",
                        "plan_sha256": bounded.PLAN_SHA256,
                        "dataset_manifest_sha256": bounded.DATASET_SHA256,
                        "teacher_identity": bounded.TEACHER, "initial_weights": initial,
                        "build_manifest_sha256": "a" * 64,
                        "source_helpers": {name: "b" * 64 for name in bounded.HELPERS},
                        "training": copy.deepcopy(bounded.FIXED),
                        "dataset": str(dataset), "trainer": str(runtime), "output": str(output)}
                    prereg_path.write_text(json.dumps(prereg) + "\n")
                    actual_sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                    args = SimpleNamespace(preregistration=prereg_path,
                        preregistration_sha256=actual_sha(prereg_path))
                    manifest = {"positions": copy.deepcopy(bounded.FIXED["positions"]),
                        "teacher_identity": bounded.TEACHER,
                        "files": {"train.positions.jsonl": {"sha256": actual_sha(dataset / "train.positions.jsonl")}}}
                    events = {"returned": False, "stop_calls": 0, "outer_handler": None}

                    def synthetic_sha(path):
                        return bounded.DATASET_SHA256 if Path(path) == dataset / "manifest.json" else actual_sha(path)

                    def fake_supervisor(_command, _output, _log, _seconds, record):
                        outer = signal.getsignal(signal.SIGTERM)
                        self.assertTrue(callable(outer), "outer TERM guard missing; refusing to self-signal")
                        events["outer_handler"] = outer
                        # Model the real supervisor's nested guard and restore.
                        with bounded.termination_guard():
                            self.assertIsNot(signal.getsignal(signal.SIGTERM), outer)
                        self.assertIs(signal.getsignal(signal.SIGTERM), outer)
                        record.update(trainer_pid=654321, trainer_pgid=654321, cleanup_status="ok")
                        events["returned"] = True
                        return {"status": "finished", "returncode": 0, "cleanup_status": "ok"}

                    def assert_locks_still_held():
                        for name in (".build.lock", ".training.lock"):
                            with (runtime / name).open("a+") as contender:
                                try:
                                    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                except BlockingIOError:
                                    continue
                                fcntl.flock(contender, fcntl.LOCK_UN)
                                self.fail(f"{name} released before cancellation cleanup")

                    def fake_stop(record):
                        events["stop_calls"] += 1
                        self.assertTrue(events["returned"])
                        assert_locks_still_held()
                        if events["stop_calls"] == 1:
                            handler = signal.getsignal(signal.SIGTERM)
                            self.assertTrue(callable(handler), "TERM guard missing; refusing to self-signal")
                            self.assertIs(handler, events["outer_handler"])
                            blocked = signal.pthread_sigmask(signal.SIG_BLOCK, set())
                            self.assertNotIn(signal.SIGTERM, blocked, "TERM unexpectedly masked")
                            os.kill(os.getpid(), signal.SIGTERM)
                            self.fail("outer TERM handler failed to raise TrainingCancelled")
                        self.assertEqual(events["stop_calls"], 2)
                        self.assertEqual(record["status"], "cancelled")
                        record["trainer_process_group_stopped"] = True

                    with ExitStack() as mocks:
                        mocks.enter_context(patch.object(preparation, "PRIVATE", private))
                        mocks.enter_context(patch.object(preparation, "outside_git", return_value=None))
                        mocks.enter_context(patch.object(preparation, "verify_build",
                            return_value={"binary": "/public/unused-trainer"}))
                        mocks.enter_context(patch.object(bounded, "sha256", side_effect=synthetic_sha))
                        mocks.enter_context(patch.object(bounded, "verify_helpers", return_value=None))
                        mocks.enter_context(patch.object(bounded, "verified_split",
                            return_value=(manifest, [0] * 112681)))
                        mocks.enter_context(patch.object(bounded, "verify_initial_weights", return_value=initial))
                        mocks.enter_context(patch.object(bounded.shutil, "disk_usage",
                            return_value=SimpleNamespace(free=6 * 2**30)))
                        mocks.enter_context(patch.object(bounded, "supervised_training", side_effect=fake_supervisor))
                        mocks.enter_context(patch.object(bounded, "stop_recorded_group", side_effect=fake_stop))
                        no_epochs = mocks.enter_context(patch.object(bounded, "verify_epochs",
                            side_effect=AssertionError("post-cancellation epoch check reached")))
                        # Any unexpected process launch fails instead of running an engine.
                        mocks.enter_context(patch.object(subprocess, "Popen",
                            side_effect=AssertionError("synthetic worker must not launch processes")))
                        with self.assertRaises(bounded.TrainingCancelled):
                            bounded.train(args)
                        no_epochs.assert_not_called()
                    saved = json.loads((output / "run.json").read_text())
                    self.assertEqual(events["stop_calls"], 2)
                    self.assertEqual(saved["status"], "cancelled")
                    self.assertEqual(saved["cleanup_status"], "ok")
                    self.assertTrue(saved["trainer_process_group_stopped"])
                    self.assertIn("TrainingCancelled", saved["error"])
                    self.assertNotIn("epoch_validation", saved)
                    self.assertIs(signal.getsignal(signal.SIGTERM), previous_handler)
            finally:
                os.umask(previous_umask)

    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ReturnedSupervisorCancellation))
    return 0 if result.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scripts", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    scripts = args.scripts.resolve(strict=True)
    if args.worker:
        return run_fixture(scripts)
    command = [sys.executable, str(Path(__file__).resolve()), "--scripts", str(scripts), "--worker"]
    try:
        return subprocess.run(command, start_new_session=True, timeout=20).returncode
    except subprocess.TimeoutExpired:
        print("isolated synthetic worker timed out", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
