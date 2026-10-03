import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_bounded as prepare


def identity():
    return {"schema": "sekirei.bounded-trainer-build-identity.v1",
            "upstream_commit": prepare.UPSTREAM, "toolchain_lock_sha256": prepare.LOCK_SHA,
            "external_patch_path": str(prepare.REPO / "patches/sekirei-train-external-labels.patch"),
            "external_patch_sha256": prepare.EXTERNAL_PATCH_SHA,
            "bounded_patch_path": str(prepare.REPO / "patches/sekirei-train-bounded-material.patch"),
            "bounded_patch_sha256": prepare.BOUNDED_PATCH_SHA,
            "expected_source_main_sha256": prepare.BOUNDED_MAIN_SHA,
            "expected_source_trainer_sha256": prepare.BOUNDED_TRAINER_SHA,
            "rustflags": prepare.RUSTFLAGS, "rustc": prepare.RUSTC,
            "cargo": "cargo 1.96.0 (public synthetic fixture)",
            "rust_tests": sorted(prepare.REQUIRED_TESTS), "recipe": copy.deepcopy(prepare.RECIPE)}


class BoundedBuilderTests(unittest.TestCase):
    def load_fixture(self, value, expected=None):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "identity.json"
            data = json.dumps(value).encode(); path.write_bytes(data)
            return prepare.load_identity(path, expected or hashlib.sha256(data).hexdigest())

    def test_known_patch_identity_loads_without_build(self):
        data = identity()
        self.assertEqual(self.load_fixture(data), data)

    def test_wrong_patch_source_or_frozen_recipe_rejected(self):
        for key in ("bounded_patch_sha256", "expected_source_main_sha256",
                    "expected_source_trainer_sha256", "external_patch_sha256"):
            data = identity(); data[key] = "a" * 64
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load_fixture(data)
        data = identity(); data["recipe"]["residual_budget_cp"] = 100
        with self.assertRaises(ValueError):
            self.load_fixture(data)

    def test_missing_duplicate_tests_or_document_digest_rejected(self):
        for names in (sorted(prepare.REQUIRED_TESTS)[:-1],
                      sorted(prepare.REQUIRED_TESTS) + [sorted(prepare.REQUIRED_TESTS)[0]]):
            data = identity(); data["rust_tests"] = names
            with self.assertRaises(ValueError):
                self.load_fixture(data)
        with self.assertRaises(ValueError):
            self.load_fixture(identity(), "b" * 64)

    def test_actual_harness_completion_requires_every_named_pass(self):
        names = sorted(prepare.REQUIRED_TESTS)
        log = "\n".join(f"test {name} ... ok" for name in names)
        log += "\ntest result: ok. 9 passed; 0 failed; 0 ignored; 100 filtered out; finished in 0.01s\n"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tests.log"; path.write_text(log)
            self.assertEqual(prepare.parse_tests(path, names)["passed"], names)
            for broken in (log.replace(" ... ok", " ... ignored", 1),
                           log.replace("9 passed", "8 passed"),
                           log + f"test {names[0]} ... ok\n"):
                path.write_text(broken)
                with self.assertRaises(ValueError):
                    prepare.parse_tests(path, names)

    def test_git_path_lists_reject_traversal_duplicates_and_truncation(self):
        self.assertEqual(prepare.paths_from_nul(b"src/main.rs\0Cargo.lock\0"),
                         ["Cargo.lock", "src/main.rs"])
        for raw in (b"../secret\0", b"/absolute\0", b"same\0same\0", b"truncated"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                prepare.paths_from_nul(raw)


if __name__ == "__main__":
    unittest.main()
