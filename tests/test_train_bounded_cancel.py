import subprocess
import sys
from pathlib import Path
import unittest


class ReturnedSupervisorCancellationTests(unittest.TestCase):
    def test_outer_guard_keeps_locks_and_records_cancel_after_supervisor_returns(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, str(root / "tests/fixtures/train_bounded_cancel.py"),
             "--scripts", str(root / "scripts")],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Ran 1 test", result.stdout + result.stderr)
        self.assertIn("OK", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
