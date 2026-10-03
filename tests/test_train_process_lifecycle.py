"""Synthetic child handles; never starts a trainer or reads private inputs."""
import io
from argparse import Namespace
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch, call

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_cpu


class TrainingProcessLifecycleTests(unittest.TestCase):
    def run_child(self, process, spawn=None):
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            with patch.object(train_cpu.subprocess, "Popen", side_effect=spawn,
                              return_value=process) as popen:
                result = train_cpu.supervised_training(["public-fixture"], Path("/unused-fixture"),
                                                      io.StringIO(), 12)
                self.assertTrue(popen.call_args.kwargs["start_new_session"])
                self.assertNotIn("preexec_fn", popen.call_args.kwargs)
                return result
        finally:
            self.assertEqual({sig: signal.getsignal(sig) for sig in previous}, previous)

    def test_normal_exit_is_reaped_without_signals(self):
        process = Mock(pid=12345, returncode=0)
        process.wait.return_value = 0
        with patch.object(train_cpu.os, "killpg") as kill:
            self.assertEqual(self.run_child(process),
                             {"status": "finished", "returncode": 0, "cleanup_status": "ok"})
        kill.assert_not_called()
        process.wait.assert_called_once_with(timeout=12)

    def test_timeout_terminates_and_reaps_before_return(self):
        process = Mock(pid=12345, returncode=-15)
        process.wait.side_effect = [subprocess.TimeoutExpired("fixture", 12), -15]
        with patch.object(train_cpu.os, "killpg") as kill:
            self.assertEqual(self.run_child(process),
                             {"status": "timeout", "returncode": -15, "cleanup_status": "ok"})
        kill.assert_called_once_with(12345, signal.SIGTERM)
        self.assertEqual(process.wait.call_args_list, [call(timeout=12), call(timeout=5)])

    def test_timeout_escalates_to_kill_then_reaps(self):
        process = Mock(pid=12345, returncode=-9)
        process.wait.side_effect = [subprocess.TimeoutExpired("fixture", 12),
                                   subprocess.TimeoutExpired("fixture", 5), -9]
        with patch.object(train_cpu.os, "killpg") as kill:
            self.assertEqual(self.run_child(process)["returncode"], -9)
        self.assertEqual(kill.call_args_list,
                         [call(12345, signal.SIGTERM), call(12345, signal.SIGKILL)])
        self.assertEqual(process.wait.call_count, 3)

    def test_cancellation_during_spawn_defers_until_handle_is_stored(self):
        for requested in (signal.SIGTERM, signal.SIGINT):
            with self.subTest(signal=requested):
                process = Mock(pid=12345, returncode=-15)
                process.wait.return_value = -15
                def spawn(*_args, **_kwargs):
                    signal.getsignal(requested)(requested, None)
                    return process
                with patch.object(train_cpu.os, "killpg") as kill:
                    with self.assertRaises(train_cpu.TrainingCancelled):
                        self.run_child(process, spawn)
                kill.assert_called_once_with(12345, signal.SIGTERM)
                process.wait.assert_called_once_with(timeout=5)

    def test_repeated_cancellation_during_cleanup_does_not_interrupt_reap(self):
        process = Mock(pid=12345, returncode=-15)
        attempts = []
        def wait(timeout):
            attempts.append(timeout)
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return -15
        process.wait.side_effect = wait
        with patch.object(train_cpu.os, "killpg") as kill:
            with self.assertRaises(train_cpu.TrainingCancelled):
                self.run_child(process)
        kill.assert_called_once_with(12345, signal.SIGTERM)
        self.assertEqual(attempts, [12, 5])

    def test_reap_failure_is_not_success(self):
        process = Mock(pid=12345, returncode=None)
        process.wait.side_effect = subprocess.TimeoutExpired("fixture", 5)
        with patch.object(train_cpu.os, "killpg") as kill:
            with self.assertRaises(subprocess.TimeoutExpired):
                self.run_child(process)
        self.assertEqual(kill.call_args_list,
                         [call(12345, signal.SIGTERM), call(12345, signal.SIGKILL)])

    def test_spawn_failure_restores_handlers_and_does_not_signal_unknown_pid(self):
        with patch.object(train_cpu.os, "killpg") as kill:
            with self.assertRaises(OSError):
                self.run_child(None, lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("fixture")))
        kill.assert_not_called()

    def test_training_lock_is_held_through_cancel_reap_and_record_is_not_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, trainer, output = root / "dataset", root / "trainer", root / "run"
            dataset.mkdir(); trainer.mkdir()
            positions = dataset / "train.positions.jsonl"
            positions.write_text('{}\n')
            (dataset / "manifest.json").write_text(json.dumps({"teacher_identity": "external:fixture",
                "files": {positions.name: {"sha256": train_cpu.sha256(positions)}}}))
            binary = trainer / "train"
            binary.write_bytes(b"synthetic executable identity only")
            (trainer / "build-manifest.json").write_text(json.dumps({"binary": str(binary),
                "binary_sha256": train_cpu.sha256(binary),
                "patch_sha256": train_cpu.sha256(train_cpu.REPO / "patches/sekirei-train-external-labels.patch")}))
            args = Namespace(dataset=dataset, trainer=trainer, output=output, epochs=1, seconds=12,
                lr=.001, min_lr=0., lr_schedule="constant", lr_schedule_epochs=None,
                max_positions=None, init_weights=None)
            process = Mock(pid=12345, returncode=-15)
            observed = []
            def wait(timeout):
                observed.append(timeout)
                with (trainer / ".training.lock").open() as other:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if timeout == 12:
                    signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
                return -15
            process.wait.side_effect = wait
            with patch.object(train_cpu.subprocess, "Popen", return_value=process), \
                    patch.object(train_cpu.os, "killpg"):
                with self.assertRaises(train_cpu.TrainingCancelled):
                    train_cpu.train(args)
            self.assertEqual(observed, [12, 5])
            record = json.loads((output / "run.json").read_text())
            self.assertEqual(record["status"], "cancelled")
            self.assertEqual(record["cleanup_status"], "ok")
            self.assertEqual(record["trainer_pid"], 12345)
            with (trainer / ".training.lock").open() as other:
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)


if __name__ == "__main__":
    unittest.main()
