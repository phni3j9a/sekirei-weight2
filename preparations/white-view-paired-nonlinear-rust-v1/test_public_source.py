"""Portable, standard-library SOURCE checks; no Rust or training execution."""
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import unittest


ROOT = Path(__file__).resolve().parent / "compiled-source"
MANIFEST_SHA256 = "dca8feb29475c5d3c492f64e1a3148b4055912adcffa44d5615b38bb94912fbf"
UPSTREAM = "f09c13026e9485a19b4ba41b91ed2e1bbdf5e1c9"
MODULES = (
    "actual", "adapter", "bound_io", "checkpoint_io", "checkpoint_native",
    "cli", "float", "parent_binding", "positions", "sha256",
)
TRAIN = "source/crates/sekirei-train/src/"
RUST_MODULES = {TRAIN + "paired_nonlinear_" + name + ".rs" for name in MODULES}
CHANGED = RUST_MODULES | {
    "source/crates/sekirei-core/Cargo.toml", "source/crates/sekirei-core/src/nnue.rs",
    "source/crates/sekirei-train/Cargo.toml", TRAIN + "main.rs", TRAIN + "trainer.rs",
    "source/crates/sekirei-usi/Cargo.toml",
}
FILES = CHANGED | {"LICENSE", "LICENSE-MIT", "LICENSE-APACHE", "nonlinear-complete.patch"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def parse_json(raw):
    def nonfinite(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=nonfinite)


def relative_path(value):
    require(type(value) is str and value and "\\" not in value and "\x00" not in value,
            "canonical relative POSIX path required")
    require(not value.startswith("/") and ":" not in value, "absolute/drive path rejected")
    require(all(part not in ("", ".", "..") for part in value.split("/")),
            "empty/dot path component rejected")
    require(str(PurePosixPath(value)) == value, "noncanonical path")
    return value


def digest(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            "lowercase SHA-256 required")


def validate_manifest(value):
    require(type(value) is dict and set(value) == {
        "schema", "upstream_commit", "measurement_source_head", "training_build_sha256",
        "private_data_included", "complete_patch_includes_untracked_modules", "files",
    }, "manifest shape")
    require(value["schema"] == "sekirei.paired-nonlinear-compiled-source-snapshot.v1", "schema")
    require(value["upstream_commit"] == UPSTREAM, "upstream commit")
    require(type(value["measurement_source_head"]) is str and
            re.fullmatch(r"[0-9a-f]{40}", value["measurement_source_head"]) is not None,
            "measurement commit")
    digest(value["training_build_sha256"])
    require(value["private_data_included"] is False and
            value["complete_patch_includes_untracked_modules"] is True, "SOURCE scope")
    require(type(value["files"]) is dict and set(value["files"]) == FILES, "exact SOURCE set")
    for path, record in value["files"].items():
        relative_path(path)
        require(type(record) is dict and set(record) == {"bytes", "sha256"}, "file identity shape")
        require(type(record["bytes"]) is int and record["bytes"] > 0, "positive integer bytes")
        digest(record["sha256"])
    return value


def verify_bytes(raw, record):
    require(len(raw) == record["bytes"] and hashlib.sha256(raw).hexdigest() == record["sha256"],
            "SOURCE bytes drift")


def verify_patch_targets(raw, source_bytes):
    """Check each unified hunk's complete postimage and every new-file byte.

    This validates patch-to-snapshot binding, not application to the upstream
    tree. Root's independent clean-tree apply check is a separate observation.
    """
    lines = raw.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.startswith(b"diff --git ")]
    require(starts and starts[0] == 0, "unified patch sections required")
    starts.append(len(lines))
    targets, new_targets = set(), set()
    for first, last in zip(starts, starts[1:]):
        section = lines[first:last]
        header = section[0].decode("utf-8").rstrip("\n")
        match = re.fullmatch(r"diff --git a/(\S+) b/(\S+)", header)
        require(match is not None and match[1] == match[2], "renamed/noncanonical patch target")
        path = "source/" + relative_path(match[2])
        require(path in source_bytes and path not in targets, "unknown/duplicate patch target")
        targets.add(path)
        target_lines = source_bytes[path].splitlines(keepends=True)
        require(b"+++ b/" + match[2].encode() + b"\n" in section, "target header")
        is_new = b"new file mode 100644\n" in section
        if is_new:
            require(b"--- /dev/null\n" in section, "new-file old header")
            new_targets.add(path)
        hunks = [i for i, line in enumerate(section) if line.startswith(b"@@ ")]
        require(hunks, "missing patch hunk")
        hunks.append(len(section))
        full_new, previous_end = [], 0
        for hfirst, hlast in zip(hunks, hunks[1:]):
            h = re.fullmatch(rb"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[^\n]*\n", section[hfirst])
            require(h is not None, "hunk shape")
            old_count, new_start, new_count = int(h[2] or b"1"), int(h[3]), int(h[4] or b"1")
            post, old_seen = [], 0
            for line in section[hfirst + 1:hlast]:
                require(line[:1] in (b" ", b"+", b"-"), "unsupported patch row")
                if line[:1] != b"-":
                    post.append(line[1:])
                if line[:1] != b"+":
                    old_seen += 1
            require(old_seen == old_count and len(post) == new_count, "hunk counts")
            offset = max(new_start - 1, 0)
            require(offset >= previous_end, "overlapping/out-of-order hunks")
            require(target_lines[offset:offset + new_count] == post, "patch postimage drift")
            previous_end = offset + new_count
            full_new.extend(post)
        if is_new:
            require(b"".join(full_new) == source_bytes[path], "new-file omission")
    require(targets == CHANGED and new_targets == RUST_MODULES, "complete 16/10 patch coverage")
    return targets, new_targets


def control_word(environment, raw):
    require(type(environment) is str and environment == "1", "exact environment")
    require(type(raw) is int and 0 <= raw < 2**32, "u32 control word")
    require(raw & ~0x3f == 0x9fc0, "RNE/masks/FTZ/DAZ control")
    return raw & 0x3f


class PublicSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest_raw = (ROOT / "public-source-manifest.json").read_bytes()
        cls.manifest = validate_manifest(parse_json(cls.manifest_raw))
        cls.raw = {path: (ROOT / path).read_bytes() for path in FILES}

    def test_exact_source_inventory_and_hashes(self):
        self.assertEqual(hashlib.sha256(self.manifest_raw).hexdigest(), MANIFEST_SHA256)
        found = {str(path.relative_to(ROOT)) for path in ROOT.rglob("*") if path.is_file()}
        self.assertEqual(found, FILES | {"public-source-manifest.json"})
        self.assertFalse(any(path.is_symlink() for path in ROOT.rglob("*")))
        for path, record in self.manifest["files"].items():
            verify_bytes(self.raw[path], record)

    def test_complete_patch_postimages_and_ten_modules(self):
        targets, new = verify_patch_targets(self.raw["nonlinear-complete.patch"], self.raw)
        self.assertEqual((len(targets), len(new)), (16, 10))

    def test_patch_detects_new_module_and_tracked_hunk_drift(self):
        for path in (TRAIN + "paired_nonlinear_float.rs", "source/crates/sekirei-core/Cargo.toml"):
            changed = dict(self.raw)
            changed[path] = changed[path].replace(b"nnue_white_view_aux_tied", b"other_feature", 1) if path.endswith("toml") else b"// drift\n" + changed[path]
            with self.assertRaises(ValueError):
                verify_patch_targets(self.raw["nonlinear-complete.patch"], changed)

    def test_recursive_duplicate_json_and_nonfinite_rejected(self):
        for raw in ('{"x":{"y":1,"y":2}}', '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}'):
            with self.assertRaises(ValueError):
                parse_json(raw)

    def test_strict_identity_types_scope_and_unknown_fields(self):
        for field, value in (("bytes", True), ("bytes", 1.0), ("bytes", 0), ("sha256", "A" * 64)):
            changed = copy.deepcopy(self.manifest)
            changed["files"]["LICENSE"][field] = value
            with self.assertRaises(ValueError):
                validate_manifest(changed)
        for field, value in (("private_data_included", 0), ("complete_patch_includes_untracked_modules", 1)):
            changed = copy.deepcopy(self.manifest); changed[field] = value
            with self.assertRaises(ValueError):
                validate_manifest(changed)
        changed = copy.deepcopy(self.manifest); changed["files"]["LICENSE"]["path"] = "other"
        with self.assertRaises(ValueError):
            validate_manifest(changed)

    def test_relative_path_and_raw_corruption_rejected(self):
        for path in ("/tmp/foreign", "../a", "a/../b", "a/./b", "a//b", "a/", "C:/a", "a\\b", "a\x00b"):
            with self.assertRaises(ValueError):
                relative_path(path)
        with self.assertRaises(ValueError):
            verify_bytes(self.raw["LICENSE"] + b" ", self.manifest["files"]["LICENSE"])

    def test_prefix_budget_integer_boundary(self):
        # Independent integer safety calculation; no optimizer/native execution.
        self.assertEqual((64 - 41 * 797, 64 + 41 * 797), (-32613, 32741))
        self.assertLessEqual(64 + 41 * 797, 32767)
        self.assertGreater(64 + 41 * 798, 32767)

    def test_supplied_float_control_word_boundary(self):
        # Supplied-word fixture, not an actual CPU read-back or policy setting.
        for status in range(64):
            self.assertEqual(control_word("1", 0x9fc0 | status), status)
        for bit in range(6, 32):
            with self.assertRaises(ValueError):
                control_word("1", 0x9fc0 ^ (1 << bit))
        for value in ("01", "1 ", "true", 1):
            with self.assertRaises(ValueError):
                control_word(value, 0x9fc0)
        with self.assertRaises(ValueError):
            control_word("1", True)

    def test_native_layout_and_symmetry_integer_fixture(self):
        # Byte count and permutations only; no full model allocation or loader.
        self.assertEqual(8 + 2420 * 256 * 2 + 256 * 2 + 512 * 32 * 4 + 32 * 4 + 32 * 4 + 4, 1305356)
        for square in range(81):
            self.assertEqual(80 - (80 - square), square)
        for bank in range(4):
            self.assertEqual((bank ^ 3) ^ 3, bank)


if __name__ == "__main__":
    unittest.main()
