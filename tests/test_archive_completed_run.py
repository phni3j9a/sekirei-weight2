"""Tiny public copy/inventory/physical-restore fixtures; no real NAS writes."""
import fcntl
import errno
import ctypes
import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import archive_completed_run as archive


class ArchiveTests(unittest.TestCase):
    def inputs(self, root):
        source = root / "completed-run"
        source.mkdir(mode=0o700)
        (source / "empty").mkdir(mode=0o700)
        (source / "nested").mkdir(mode=0o750)
        (source / "nested" / "weights.bin").write_bytes(bytes(range(128)))
        (source / "nested" / "weights.bin").chmod(0o640)
        (source / "result.json").write_text('{"synthetic":true}\n')
        (source / "result.json").chmod(0o600)
        single = root / "sidecar.json"
        single.write_text('{"public_fixture":"sidecar"}\n')
        single.chmod(0o600)
        completion, stop, lock = root / "completion.json", root / "stop.json", root / "writer.lock"
        completion.write_text('{"synthetic_completed":true}\n')
        stop.write_text('{"synthetic_writers_stopped":true}\n')
        lock.touch(mode=0o600)
        selection = {"schema": archive.SELECTION_SCHEMA, "completed": True, "writers_stopped": True,
                     "roots": [{"name": "run", "path": str(source), "kind": "directory"},
                               {"name": "sidecar", "path": str(single), "kind": "file"}],
                     "evidence": {"completion": archive.full_ref(completion), "stop": archive.full_ref(stop)},
                     "locks": [str(lock)]}
        control = root / "selection.json"
        control.write_bytes(archive.json_bytes(selection))
        inventory = root / "inventory.json"
        return source, single, selection, control, inventory

    def freeze(self, root):
        source, single, selection, control, inventory = self.inputs(root)
        result = archive.build_inventory(control, archive.full_ref(control)["sha256"], inventory)
        return source, single, selection, control, inventory, result["manifest_sha256"]

    def copied(self, root):
        values = self.freeze(root)
        destination = root / "archive"
        result = archive.archive_run(values[4], values[5], destination, reserve_bytes=0)
        self.assertEqual(result["status"], "archive-copy-verified")
        return *values, destination

    def test_full_inventory_copy_and_fresh_physical_restore_preserve_originals_and_paths(self):
        with tempfile.TemporaryDirectory(dir=Path.home()) as directory:
            root = Path(directory)
            source, single, selection, control, inventory, digest, destination = self.copied(root)
            manifest = json.loads(inventory.read_bytes())
            self.assertEqual(manifest["summary"], {"roots": 2, "files": 3, "directories": 5, "bytes": 176})
            self.assertEqual(manifest["files"]["payload/run/nested/weights.bin"]["original_path"],
                             str(source / "nested" / "weights.bin"))
            self.assertEqual(manifest["selection"]["evidence"], selection["evidence"])
            self.assertEqual(manifest["producer_ref"], archive.full_ref(Path(archive.__file__).resolve()))
            before = archive.selected_inventory(manifest["roots"])
            restored = root / "fresh-ssd-restoration"
            result = archive.restore_run(destination, digest, restored, reserve_bytes=0)
            self.assertEqual(result["status"], "physical-restore-verified")
            self.assertFalse(result["standalone_environment_restore"])
            self.assertTrue(result["original_reference_paths_preserved"])
            self.assertEqual(archive.selected_inventory(manifest["roots"]), before)
            self.assertEqual(archive.walk_tree(restored), archive.walk_tree(destination))
            self.assertEqual((restored / "manifest.json").read_bytes(), inventory.read_bytes())
            self.assertEqual((restored / "payload" / "sidecar").read_bytes(), single.read_bytes())
            for relative, info in manifest["files"].items():
                original = Path(info["original_path"])
                saved, copied = destination / relative, restored / relative
                self.assertEqual(original.read_bytes(), copied.read_bytes())
                self.assertEqual(len({original.stat().st_ino, saved.stat().st_ino, copied.stat().st_ino}), 3)
            self.assertEqual(inventory.stat().st_mode & 0o777, 0o600)
            self.assertEqual(restored.stat().st_mode & 0o777, 0o700)
            self.assertTrue((restored / "payload/run/empty").is_dir())
            self.assertEqual(archive.verify_archive(destination, digest)["status"], "archive-verified")
            self.assertFalse(list(root.glob(".*.stage-*")))

    def test_selection_and_inventory_contracts_reject_unpinned_ambiguous_or_incomplete_inputs(self):
        modes = ("notcomplete", "notstopped", "boolalias", "duplicate-root", "overlap", "evidence", "wrongsha",
                 "root-symlink", "nested-symlink", "fifo", "duplicate-json")
        for mode in modes:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir=Path.home()) as directory:
                root = Path(directory)
                source, single, selection, control, inventory = self.inputs(root)
                if mode == "notcomplete":
                    selection["completed"] = False
                elif mode == "notstopped":
                    selection["writers_stopped"] = False
                elif mode == "boolalias":
                    selection["completed"] = 1
                elif mode == "duplicate-root":
                    selection["roots"].append(selection["roots"][0])
                elif mode == "overlap":
                    selection["roots"].append({"name": "overlap", "path": str(source / "nested"), "kind": "directory"})
                elif mode == "evidence":
                    Path(selection["evidence"]["stop"]["path"]).write_text("changed")
                elif mode == "root-symlink":
                    link = root / "link"; link.symlink_to(source)
                    selection["roots"][0]["path"] = str(link)
                elif mode == "nested-symlink":
                    (source / "link").symlink_to(single)
                elif mode == "fifo":
                    os.mkfifo(source / "fifo")
                control.write_bytes(archive.json_bytes(selection))
                if mode == "duplicate-json":
                    control.write_text('{"schema":"x","schema":"x"}')
                digest = "a" * 64 if mode == "wrongsha" else archive.full_ref(control)["sha256"]
                with self.assertRaises((ValueError, FileNotFoundError)):
                    archive.build_inventory(control, digest, inventory)
                self.assertFalse(inventory.exists())

    def test_changed_source_and_controls_refuse_archive_without_publishing(self):
        for mode in ("changed", "missing", "extra", "extra-empty-dir", "symlink", "mode", "control", "evidence"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir=Path.home()) as directory:
                root = Path(directory)
                source, _, selection, control, inventory, digest = self.freeze(root)
                path = source / "result.json"
                if mode == "changed":
                    path.write_text("changed")
                elif mode == "missing":
                    path.unlink()
                elif mode == "extra":
                    (source / "extra").write_text("extra")
                elif mode == "extra-empty-dir":
                    (source / "extra-empty").mkdir()
                elif mode == "symlink":
                    (source / "link").symlink_to(path)
                elif mode == "mode":
                    path.chmod(0o644)
                elif mode == "control":
                    control.write_text("{}")
                else:
                    Path(selection["evidence"]["completion"]["path"]).write_text("changed")
                with self.assertRaises((ValueError, FileNotFoundError)):
                    archive.archive_run(inventory, digest, root / "destination", reserve_bytes=0)
                self.assertFalse((root / "destination").exists())
                self.assertFalse(list(root.glob(".destination.stage-*")))

    def test_archive_integrity_checks_extra_missing_corrupt_directories_symlinks_and_control_sha(self):
        for mode in ("corrupt", "missing", "extra", "missing-dir", "extra-dir", "symlink", "manifest", "wrongsha"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir=Path.home()) as directory:
                root = Path(directory)
                *_, digest, destination = self.copied(root)
                path = destination / "payload/run/result.json"
                if mode == "corrupt":
                    path.write_text("corrupt")
                elif mode == "missing":
                    path.unlink()
                elif mode == "extra":
                    (destination / "extra").write_text("extra")
                elif mode == "missing-dir":
                    (destination / "payload/run/empty").rmdir()
                elif mode == "extra-dir":
                    (destination / "empty-extra").mkdir()
                elif mode == "symlink":
                    path.unlink(); path.symlink_to(root / "sidecar.json")
                elif mode == "manifest":
                    (destination / "manifest.json").write_text("{}")
                else:
                    digest = "a" * 64
                with self.assertRaises((ValueError, FileNotFoundError)):
                    archive.verify_archive(destination, digest)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    archive.restore_run(destination, digest, root / "restore", reserve_bytes=0)
                self.assertFalse((root / "restore").exists())

    def test_inventory_manifest_rejects_namespace_escape_boolean_sizes_and_missing_roots(self):
        for mode in ("escape", "wrong-original", "boolsize", "boolmode", "missing-root", "missing-parent", "extra-key"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir=Path.home()) as directory:
                root = Path(directory)
                *_, inventory, digest = self.freeze(root)
                body = json.loads(inventory.read_bytes())
                relative = "payload/run/nested/weights.bin"
                if mode == "escape":
                    body["files"]["../escape"] = body["files"].pop(relative)
                elif mode == "wrong-original":
                    body["files"][relative]["original_path"] = "/different/original"
                elif mode == "boolsize":
                    body["files"][relative]["bytes"] = True
                elif mode == "boolmode":
                    body["directories"]["payload/run"]["mode"] = True
                elif mode == "missing-root":
                    body["files"].pop("payload/sidecar")
                elif mode == "missing-parent":
                    body["directories"].pop("payload/run/nested")
                else:
                    body["extra"] = "unexpected"
                with self.assertRaises(ValueError):
                    archive.validate_manifest(body)

    def test_mutation_during_copy_and_after_publication_never_leaves_success_manifest(self):
        for mode in ("copy", "publish", "inventory-publication"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir=Path.home()) as directory:
                root = Path(directory)
                if mode == "inventory-publication":
                    source, _, _, control, inventory = self.inputs(root)
                    original = archive.create_file
                    def mutate(path, data, *args):
                        original(path, data, *args)
                        (source / "result.json").write_text("changed after inventory publication")
                    with patch.object(archive, "create_file", side_effect=mutate), self.assertRaises(ValueError):
                        archive.build_inventory(control, archive.full_ref(control)["sha256"], inventory)
                    self.assertFalse(inventory.exists())
                    self.assertTrue(inventory.with_name(inventory.name + ".failed").is_file())
                    continue
                source, _, _, _, inventory, digest = self.freeze(root)
                destination = root / "destination"
                if mode == "copy":
                    original = archive.copy_payload
                    def mutate(*args, **kwargs):
                        original(*args, **kwargs)
                        (source / "result.json").write_text("changed during copy")
                    target = "copy_payload"
                else:
                    original = archive.publish_without_replace
                    def mutate(*args, **kwargs):
                        original(*args, **kwargs)
                        (source / "result.json").write_text("changed during publication")
                    target = "publish_without_replace"
                with patch.object(archive, target, side_effect=mutate), self.assertRaises(ValueError):
                    archive.archive_run(inventory, digest, destination, reserve_bytes=0)
                self.assertFalse((destination / "manifest.json").exists())
                if mode == "publish":
                    self.assertTrue((destination / "manifest.failed.json").is_file())
                else:
                    self.assertTrue(list(root.glob(".destination.stage-*")))

    def test_archive_mutation_during_restore_refuses_success(self):
        for mode in ("copy", "publish"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir=Path.home()) as directory:
                root = Path(directory)
                *_, digest, destination = self.copied(root)
                restored = root / "restored"
                original = getattr(archive, "copy_payload" if mode == "copy" else "publish_without_replace")
                def mutate(*args, **kwargs):
                    original(*args, **kwargs)
                    (destination / "payload/run/result.json").write_text("changed during restore")
                with patch.object(archive, "copy_payload" if mode == "copy" else "publish_without_replace", side_effect=mutate), \
                        self.assertRaises(ValueError):
                    archive.restore_run(destination, digest, restored, reserve_bytes=0)
                self.assertFalse((restored / "manifest.json").exists())
                if mode == "publish":
                    self.assertTrue((restored / "manifest.failed.json").is_file())

    def test_busy_locks_capacity_existing_outputs_and_git_paths_are_refused(self):
        with tempfile.TemporaryDirectory(dir=Path.home()) as directory:
            root = Path(directory)
            _, _, selection, control, inventory, digest = self.freeze(root)
            with Path(selection["locks"][0]).open("rb") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    archive.archive_run(inventory, digest, root / "locked", reserve_bytes=0)
            with patch.object(archive.shutil, "disk_usage", return_value=SimpleNamespace(free=0)), self.assertRaises(ValueError):
                archive.archive_run(inventory, digest, root / "too-full", reserve_bytes=0)
            existing = root / "existing"; existing.mkdir(); (existing / "sentinel").write_text("kept")
            with self.assertRaises(ValueError):
                archive.archive_run(inventory, digest, existing, reserve_bytes=0)
            self.assertEqual((existing / "sentinel").read_text(), "kept")
            repo = root / "repository"; repo.mkdir()
            subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
            with self.assertRaisesRegex(ValueError, "outside Git"):
                archive.archive_run(inventory, digest, repo / "copy", reserve_bytes=0)

    def test_publication_race_does_not_replace_a_new_destination(self):
        with tempfile.TemporaryDirectory(dir=Path.home()) as directory:
            root = Path(directory)
            *_, inventory, digest = self.freeze(root)
            destination = root / "race"
            original = archive.publish_without_replace
            def racing(stage, target, **kwargs):
                target.mkdir(); (target / "sentinel").write_text("untouched concurrent destination")
                original(stage, target, **kwargs)
            with patch.object(archive, "publish_without_replace", side_effect=racing), self.assertRaises(FileExistsError):
                archive.archive_run(inventory, digest, destination, reserve_bytes=0)
            self.assertEqual((destination / "sentinel").read_text(), "untouched concurrent destination")
            self.assertTrue(list(root.glob(".race.stage-*")))

    def unsupported_rename(self, code=errno.EINVAL):
        class Rename:
            def __call__(self, *_):
                ctypes.set_errno(code)
                return -1
        return patch.object(archive.ctypes, "CDLL", return_value=SimpleNamespace(renameat2=Rename()))

    def test_unsupported_filesystem_copy_publication_preserves_stage_and_physical_restoration(self):
        for code in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
            with self.subTest(errno=code), tempfile.TemporaryDirectory(dir=Path.home()) as directory:
                root = Path(directory)
                *_, inventory, digest = self.freeze(root)
                destination = root / "copy-published"
                with self.unsupported_rename(code):
                    result = archive.archive_run(inventory, digest, destination, reserve_bytes=0)
                publication = result["publication"]
                self.assertEqual(publication["method"], "exclusive-directory-copy-manifest-last")
                self.assertFalse(publication["whole_directory_atomic"])
                self.assertTrue(publication["source_stage_retained"])
                stage = Path(publication["source_stage"])
                self.assertTrue(stage.is_dir())
                self.assertEqual(archive.walk_tree(stage), archive.walk_tree(destination))
                manifest = json.loads(inventory.read_bytes())
                originals = archive.selected_inventory(manifest["roots"])
                saved = archive.walk_tree(destination)
                with self.unsupported_rename(code):
                    restored = archive.restore_run(destination, digest, root / "restored-copy", reserve_bytes=0)
                self.assertEqual(restored["publication"]["method"], "exclusive-directory-copy-manifest-last")
                self.assertFalse(restored["publication"]["whole_directory_atomic"])
                self.assertEqual(archive.walk_tree(Path(restored["restoration"])), saved)
                self.assertEqual(archive.walk_tree(destination), saved)
                self.assertEqual(archive.selected_inventory(manifest["roots"]), originals)
                for name in manifest["files"]:
                    self.assertNotEqual((destination / name).stat().st_ino,
                                        (Path(restored["restoration"]) / name).stat().st_ino)

    def test_only_unsupported_fs_errors_fall_back_and_existing_targets_never_change(self):
        for code in (errno.EEXIST, errno.EACCES, errno.EXDEV, errno.EIO):
            with self.subTest(errno=code), tempfile.TemporaryDirectory(dir=Path.home()) as directory:
                root = Path(directory)
                *_, inventory, digest = self.freeze(root)
                with self.unsupported_rename(code), patch.object(archive, "copy_publish_without_replace") as fallback, \
                        self.assertRaises(OSError):
                    archive.archive_run(inventory, digest, root / "destination", reserve_bytes=0)
                fallback.assert_not_called()
                self.assertFalse((root / "destination").exists())
        for kind in ("file", "directory"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(dir=Path.home()) as directory:
                root = Path(directory)
                *_, inventory, digest = self.freeze(root)
                destination = root / "race"
                original = archive.copy_publish_without_replace
                def racing(stage, target, reserve):
                    if kind == "file":
                        target.write_text("existing file kept")
                    else:
                        target.mkdir(); (target / "sentinel").write_text("existing directory kept")
                    return original(stage, target, reserve)
                with self.unsupported_rename(), patch.object(archive, "copy_publish_without_replace", side_effect=racing), \
                        self.assertRaises(FileExistsError):
                    archive.archive_run(inventory, digest, destination, reserve_bytes=0)
                path = destination if kind == "file" else destination / "sentinel"
                self.assertEqual(path.read_text(), "existing file kept" if kind == "file" else "existing directory kept")

    def test_fallback_manifest_last_partial_failures_and_stage_or_destination_changes_are_preserved(self):
        modes = ("copy-failure", "extra", "missing", "altered", "mode", "stage-altered", "partial-manifest")
        for mode in modes:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir=Path.home()) as directory:
                root = Path(directory)
                *_, inventory, digest = self.freeze(root)
                destination = root / "destination"
                original = archive.copy_file
                def intervene(source, target, expected):
                    in_fallback = target.is_relative_to(destination)
                    if in_fallback:
                        self.assertFalse((destination / "manifest.json").exists())
                        with self.assertRaises(FileNotFoundError):
                            archive.verify_archive(destination, digest)
                        if mode == "partial-manifest" and target.name == "manifest.json":
                            archive.create_file(target, b'{"partial":')
                            raise OSError("synthetic partial manifest failure")
                    original(source, target, expected)
                    if not in_fallback or target.name == "manifest.json":
                        return
                    if mode == "copy-failure":
                        raise OSError("synthetic copied-payload failure")
                    if mode == "extra":
                        (destination / "extra").write_text("extra")
                    elif mode == "missing":
                        target.unlink()
                    elif mode == "altered":
                        target.write_text("changed")
                    elif mode == "mode":
                        target.chmod(0o777)
                    elif mode == "stage-altered":
                        source.write_text("changed source stage")
                manifest = json.loads(inventory.read_bytes())
                originals = archive.selected_inventory(manifest["roots"])
                with self.unsupported_rename(), patch.object(archive, "copy_file", side_effect=intervene), \
                        self.assertRaises((ValueError, OSError)):
                    archive.archive_run(inventory, digest, destination, reserve_bytes=0)
                self.assertTrue(destination.is_dir())
                self.assertTrue(list(root.glob(".destination.stage-*")))
                self.assertFalse((destination / "manifest.json").exists())
                if mode == "partial-manifest":
                    self.assertEqual((destination / "manifest.failed.json").read_bytes(), b'{"partial":')
                with self.assertRaises(FileNotFoundError):
                    archive.verify_archive(destination, digest)
                self.assertEqual(archive.selected_inventory(manifest["roots"]), originals)

    def test_failed_restore_fallback_preserves_the_complete_source_archive(self):
        for mode in ("extra", "missing", "altered", "mode"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir=Path.home()) as directory:
                root = Path(directory)
                *_, digest, source_archive = self.copied(root)
                before = archive.walk_tree(source_archive)
                destination = root / "failed-restoration"
                original = archive.copy_file
                def intervene(source, target, expected):
                    original(source, target, expected)
                    if target.is_relative_to(destination) and target.name != "manifest.json":
                        if mode == "extra":
                            (destination / "unexpected").write_text("extra")
                        elif mode == "missing":
                            target.unlink()
                        elif mode == "altered":
                            target.write_text("changed")
                        else:
                            target.chmod(0o777)
                with self.unsupported_rename(), patch.object(archive, "copy_file", side_effect=intervene), \
                        self.assertRaises(ValueError):
                    archive.restore_run(source_archive, digest, destination, reserve_bytes=0)
                self.assertFalse((destination / "manifest.json").exists())
                self.assertTrue(destination.is_dir())
                self.assertTrue(list(root.glob(".failed-restoration.stage-*")))
                self.assertEqual(archive.walk_tree(source_archive), before)
                self.assertEqual(archive.verify_archive(source_archive, digest)["status"], "archive-verified")

    def test_fallback_capacity_accounts_for_simultaneous_stage_and_destination(self):
        with tempfile.TemporaryDirectory(dir=Path.home()) as directory:
            root = Path(directory)
            *_, inventory, digest = self.freeze(root)
            destination = root / "too-full"
            original = archive.capacity
            observed = []
            def limited(parent, manifest, data, reserve):
                observed.append((len(manifest["files"]), len(data), reserve))
                if len(observed) == 2:
                    with patch.object(archive.shutil, "disk_usage", return_value=SimpleNamespace(free=0)):
                        original(parent, manifest, data, reserve)
                else:
                    original(parent, manifest, data, reserve)
            with self.unsupported_rename(), patch.object(archive, "capacity", side_effect=limited), \
                    self.assertRaisesRegex(ValueError, "insufficient capacity"):
                archive.archive_run(inventory, digest, destination, reserve_bytes=73)
            self.assertEqual(len(observed), 2)
            self.assertEqual(observed[0][0], 3)
            self.assertEqual(observed[1], (4, 0, 73))
            self.assertFalse(destination.exists())
            self.assertTrue(list(root.glob(".too-full.stage-*")))

    def test_inventory_cli_smoke_and_all_subcommand_help_contracts(self):
        script = str(Path(archive.__file__).resolve())
        with tempfile.TemporaryDirectory(dir=Path.home()) as directory:
            root = Path(directory)
            *_, control, inventory = self.inputs(root)
            completed = subprocess.run([sys.executable, script, "inventory", "--selection", str(control),
                                        "--expected-selection-sha256", archive.full_ref(control)["sha256"],
                                        "--output", str(inventory)], check=True, text=True, capture_output=True)
            result = json.loads(completed.stdout)
            self.assertEqual(result["status"], "inventory-frozen")
            self.assertEqual(result["manifest_sha256"], archive.full_ref(inventory)["sha256"])
        for command in ("inventory", "archive", "verify", "restore"):
            result = subprocess.run([sys.executable, script, command, "--help"],
                                    check=True, text=True, capture_output=True)
            if command in ("archive", "restore"):
                self.assertIn("--destination", result.stdout)
                self.assertIn("--reserve-bytes", result.stdout)
            if command == "restore":
                self.assertIn("--ssd-root", result.stdout)


if __name__ == "__main__":
    unittest.main()
