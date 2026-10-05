#!/usr/bin/env python3
"""Copy immutable selected run roots and verify a separate physical restoration.

Archive manifests are private: they retain original absolute reference paths.
No operation deletes, relocates, resumes, or overwrites an original or archive.
Selection completion/stop evidence must be established by the calling Root;
this helper binds those files without claiming to reproduce their experiments.
"""
import argparse
from contextlib import contextmanager
import ctypes
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile


SELECTION_SCHEMA = "sekirei.completed-run-selection.v1"
INVENTORY_SCHEMA = "sekirei.completed-run-inventory.v1"
DEFAULT_NAS_ROOT = Path("/mnt/storage/NAS/sekirei-weight2")
DEFAULT_SSD_ROOT = Path("/home/server/.local/share/sekirei-weight2")
SHA_RE = re.compile(r"[0-9a-f]{64}")
NAME_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}")


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def bad_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad_constant)


def exact_keys(value, names):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError("unexpected contract fields")


def require_sha(value):
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        raise ValueError("expected an externally pinned lowercase SHA-256")


def canonical(path, existing=True):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("paths must be absolute and canonical")
    resolved = path.resolve(strict=existing)
    if path != resolved:
        raise ValueError("symlink/noncanonical path component is forbidden")
    return resolved


def original_path(value):
    """Validate archived reference text without consulting a live original."""
    if (not isinstance(value, str) or not Path(value).is_absolute()
            or ".." in Path(value).parts or str(Path(value)) != value):
        raise ValueError("invalid original absolute reference")
    return Path(value)


def require_regular(path):
    path = canonical(path)
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("expected a regular file, without a symlink")
    return path


def relative_path(value):
    if (not isinstance(value, str) or not value or "\\" in value
            or PurePosixPath(value).is_absolute() or any(p in ("", ".", "..") for p in value.split("/"))):
        raise ValueError("invalid archive-relative path")
    return value


def open_read(path):
    path = require_regular(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError("source was replaced by a nonregular file")
    return os.fdopen(fd, "rb")


def stat_identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mode,
            info.st_mtime_ns, info.st_ctime_ns)


def file_info(path):
    with open_read(path) as stream:
        before = os.fstat(stream.fileno())
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        after = os.fstat(stream.fileno())
    if stat_identity(before) != stat_identity(after) or stat_identity(after) != stat_identity(Path(path).lstat()):
        raise ValueError("file changed during inventory hashing")
    return {"bytes": before.st_size, "sha256": digest, "mode": stat.S_IMODE(before.st_mode)}


def full_ref(path):
    path = require_regular(path)
    info = file_info(path)
    return {"path": str(path), "bytes": info["bytes"], "sha256": info["sha256"]}


def validate_ref(ref, read=True):
    exact_keys(ref, ("path", "bytes", "sha256"))
    require_sha(ref["sha256"])
    if type(ref["bytes"]) is not int or ref["bytes"] < 0:
        raise ValueError("reference bytes must be a nonnegative integer")
    if read:
        if full_ref(ref["path"]) != ref:
            raise ValueError("evidence/reference size or SHA mismatch")
    else:
        original_path(ref["path"])


def pinned_json(path, expected_sha):
    require_sha(expected_sha)
    with open_read(path) as stream:
        data = stream.read()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError("JSON file differs from the externally pinned SHA")
    return strict_json(data), data


def overlap(a, b):
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def validate_selection(selection, read=True):
    exact_keys(selection, ("schema", "completed", "writers_stopped", "roots", "evidence", "locks"))
    if (selection["schema"] != SELECTION_SCHEMA or selection["completed"] is not True
            or selection["writers_stopped"] is not True):
        raise ValueError("only completed, stopped runs may be selected")
    exact_keys(selection["evidence"], ("completion", "stop"))
    for ref in selection["evidence"].values():
        validate_ref(ref, read=read)
    if not isinstance(selection["roots"], list) or not selection["roots"]:
        raise ValueError("selection needs nonempty roots")
    names, roots = set(), []
    for root in selection["roots"]:
        exact_keys(root, ("name", "path", "kind"))
        if (not isinstance(root["name"], str) or NAME_RE.fullmatch(root["name"]) is None
                or root["name"] in names or root["kind"] not in ("file", "directory")):
            raise ValueError("invalid/duplicated selected root")
        path = canonical(root["path"]) if read else original_path(root["path"])
        if read:
            mode = path.lstat().st_mode
            if not (stat.S_ISREG(mode) if root["kind"] == "file" else stat.S_ISDIR(mode)):
                raise ValueError("selected root kind differs from the filesystem")
        if any(overlap(path, other) for other in roots):
            raise ValueError("selected roots overlap")
        roots.append(path); names.add(root["name"])
    if not isinstance(selection["locks"], list) or len(set(selection["locks"])) != len(selection["locks"]):
        raise ValueError("invalid lock collection")
    for path in selection["locks"]:
        if read:
            require_regular(path)
        else:
            original_path(path)
    return sorted(selection["roots"], key=lambda root: root["name"])


@contextmanager
def selection_locks(selection):
    handles = []
    try:
        for path in sorted(selection["locks"]):
            handle = open_read(path)
            handles.append(handle)
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
        for path, handle in zip(sorted(selection["locks"]), handles):
            current, held = Path(path).lstat(), os.fstat(handle.fileno())
            if (current.st_dev, current.st_ino, current.st_mode) != (held.st_dev, held.st_ino, held.st_mode):
                raise ValueError("writer lock path changed while held")
    finally:
        for handle in reversed(handles):
            handle.close()


def walk_tree(root):
    """Include empty directories and reject symlinks and every special file."""
    root = canonical(root)
    files, directories = {}, {}
    def walk(path, relative):
        before = path.lstat()
        mode = before.st_mode
        if stat.S_ISLNK(mode):
            raise ValueError("symlink in source/archive tree")
        if stat.S_ISREG(mode):
            files[relative] = file_info(path)
        elif stat.S_ISDIR(mode):
            directories[relative] = stat.S_IMODE(mode)
            before_names = sorted(entry.name for entry in os.scandir(path))
            for name in before_names:
                child = name if relative == "." else f"{relative}/{name}"
                walk(path / name, child)
            if before_names != sorted(entry.name for entry in os.scandir(path)):
                raise ValueError("directory membership changed during inventory")
            if stat_identity(before) != stat_identity(path.lstat()):
                raise ValueError("directory changed during inventory")
        else:
            raise ValueError("special file in source/archive tree")
    walk(root, ".")
    return {"files": files, "directories": directories}


def selected_inventory(roots):
    files, directories = {}, {".": {"mode": 0o700, "original_path": None},
                               "payload": {"mode": 0o700, "original_path": None}}
    for root in roots:
        source = canonical(root["path"])
        base = f"payload/{root['name']}"
        tree = walk_tree(source)
        for relative, info in tree["files"].items():
            target = base if relative == "." else f"{base}/{relative}"
            original = source if relative == "." else source / relative
            files[target] = dict(info, original_path=str(original))
        for relative, mode in tree["directories"].items():
            target = base if relative == "." else f"{base}/{relative}"
            original = source if relative == "." else source / relative
            directories[target] = {"mode": mode, "original_path": str(original)}
    return {"files": dict(sorted(files.items())), "directories": dict(sorted(directories.items()))}


def private_new_output(output, protected):
    output = canonical(output, existing=False)
    parent = canonical(output.parent)
    if not parent.is_dir() or output.exists() or output.is_symlink():
        raise ValueError("output must be fresh, with an existing parent directory")
    if any(overlap(output, canonical(path, existing=False)) for path in protected):
        raise ValueError("output overlaps a protected source/control")
    if any((path / ".git").exists() for path in (parent, *parent.parents)):
        raise ValueError("private archive/manifest/restore output must be outside Git")
    if parent.stat().st_uid != os.getuid():
        raise ValueError("output parent must be owned by the current user")
    if not any(path.stat().st_uid == os.getuid() and stat.S_IMODE(path.stat().st_mode) == 0o700
               for path in (parent, *parent.parents)):
        raise ValueError("output requires an owned 0700 private ancestor")
    return output


def create_file(path, data, mode=0o600):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush(); os.fsync(stream.fileno())
    os.chmod(path, mode)


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def directory_sync(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def copy_publish_without_replace(stage, destination, reserve_bytes):
    """Reserve a fresh directory, copy its payload, and publish manifest last.

    This fallback is deliberately not a whole-directory atomic operation. The
    stage remains intact; until the last manifest's full SHA matches, normal
    verification cannot accept the destination. Failed copies remain available.
    """
    stage = canonical(stage)
    frozen = walk_tree(stage)
    if ("manifest.json" not in frozen["files"] or not stage.is_dir()
            or frozen["directories"].get(".") != 0o700):
        raise ValueError("copy publication requires a private complete stage with a regular manifest")
    # The stage is already allocated on this filesystem. Check the extra full
    # physical copy against the *current* free space, including the manifest,
    # so both stage and destination fit simultaneously with the same reserve.
    capacity(destination.parent, frozen, b"", reserve_bytes)
    destination.mkdir(mode=0o700, exist_ok=False)
    try:
        for relative in sorted(frozen["directories"], key=lambda value: (value.count("/"), value)):
            if relative != ".":
                (destination / relative).mkdir(mode=0o700, exist_ok=False)
        for relative, expected in sorted(frozen["files"].items()):
            if relative != "manifest.json":
                copy_file(stage / relative, destination / relative, expected)
        for relative, mode in sorted(frozen["directories"].items(),
                                     key=lambda item: item[0].count("/"), reverse=True):
            path = destination if relative == "." else destination / relative
            os.chmod(path, mode)
            directory_sync(path)
        without_manifest = {"files": {name: info for name, info in frozen["files"].items()
                                      if name != "manifest.json"},
                            "directories": frozen["directories"]}
        if walk_tree(destination) != without_manifest or walk_tree(stage) != frozen:
            raise ValueError("copy publication payload or source stage changed before manifest")
        # O_EXCL/O_NOFOLLOW applies here too. A partial file may be visible, but
        # read_archive requires its complete externally pinned size/SHA first.
        copy_file(stage / "manifest.json", destination / "manifest.json", frozen["files"]["manifest.json"])
        if walk_tree(destination) != frozen or walk_tree(stage) != frozen:
            raise ValueError("copy publication complete tree or source stage changed")
        directory_sync(destination)
        directory_sync(destination.parent)
    except BaseException:
        # Preserve every copied byte and prevent a final-check failure from
        # leaving a valid success-manifest path. Never remove a failed output.
        manifest = destination / "manifest.json"
        if manifest.exists() and not (destination / "manifest.failed.json").exists():
            os.rename(manifest, destination / "manifest.failed.json")
        raise
    return "exclusive-directory-copy-manifest-last"


def publish_without_replace(stage, destination, reserve_bytes=512 * 2**20):
    """Prefer atomic no-replace rename; only unsupported-FS errors may copy."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        return copy_publish_without_replace(stage, destination, reserve_bytes)
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(stage), -100, os.fsencode(destination), 1) != 0:
        code = ctypes.get_errno()
        if code in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
            return copy_publish_without_replace(stage, destination, reserve_bytes)
        raise OSError(code, os.strerror(code), str(destination))
    directory_sync(destination.parent)
    return "atomic-rename-noreplace"


def publication_receipt(method, stage):
    retained = method == "exclusive-directory-copy-manifest-last"
    return {"method": method, "whole_directory_atomic": not retained,
            "source_stage_retained": retained, "source_stage": str(stage) if retained else None}


def build_inventory(selection_path, expected_selection_sha, output):
    selection, _ = pinned_json(selection_path, expected_selection_sha)
    roots = validate_selection(selection)
    controls = [Path(selection_path), Path(__file__), *(Path(ref["path"]) for ref in selection["evidence"].values())]
    output = private_new_output(output, [*(Path(root["path"]) for root in roots), *controls])
    with selection_locks(selection):
        producer = full_ref(Path(__file__).resolve())
        selection_ref = full_ref(selection_path)
        before = selected_inventory(roots)
        manifest = {"schema": INVENTORY_SCHEMA, "selection_ref": selection_ref, "selection": selection,
                    "producer_ref": producer, "roots": roots, **before,
                    "summary": {"roots": len(roots), "files": len(before["files"]),
                                "directories": len(before["directories"]),
                                "bytes": sum(info["bytes"] for info in before["files"].values())},
                    "source_originals_preserved": True, "standalone_environment_restore": False}
        def check_sources():
            if (before != selected_inventory(roots) or full_ref(selection_path) != selection_ref
                    or full_ref(Path(__file__).resolve()) != producer):
                raise ValueError("source/control changed during inventory preparation")
            validate_selection(selection)
        check_sources()
        create_file(output, json_bytes(manifest))
        try:
            check_sources()
        except Exception:
            os.rename(output, output.with_name(output.name + ".failed"))
            raise
    return {"status": "inventory-frozen", "manifest": str(output),
            "manifest_sha256": full_ref(output)["sha256"], **manifest["summary"]}


def validate_manifest(manifest):
    exact_keys(manifest, ("schema", "selection_ref", "selection", "producer_ref", "roots", "files",
                          "directories", "summary", "source_originals_preserved", "standalone_environment_restore"))
    if (manifest["schema"] != INVENTORY_SCHEMA or manifest["source_originals_preserved"] is not True
            or manifest["standalone_environment_restore"] is not False):
        raise ValueError("unrecognized archive inventory contract")
    validate_ref(manifest["selection_ref"], read=False)
    validate_ref(manifest["producer_ref"], read=False)
    roots = validate_selection(manifest["selection"], read=False)
    if roots != manifest["roots"]:
        raise ValueError("manifest root collection differs from its selection")
    if not isinstance(manifest["files"], dict) or not isinstance(manifest["directories"], dict):
        raise ValueError("invalid inventory collections")
    expected_dirs = {".": {"mode": 0o700, "original_path": None},
                     "payload": {"mode": 0o700, "original_path": None}}
    for relative, info in manifest["files"].items():
        relative_path(relative)
        exact_keys(info, ("bytes", "sha256", "mode", "original_path"))
        require_sha(info["sha256"])
        if type(info["bytes"]) is not int or info["bytes"] < 0:
            raise ValueError("invalid file size in inventory")
    for relative, info in manifest["directories"].items():
        if relative != ".":
            relative_path(relative)
        exact_keys(info, ("mode", "original_path"))
    for relative, info in {**manifest["files"], **manifest["directories"]}.items():
        if type(info["mode"]) is not int or not 0 <= info["mode"] <= 0o7777:
            raise ValueError("invalid mode in inventory")
        if relative in expected_dirs:
            if info != expected_dirs[relative]:
                raise ValueError("private archive namespace mode mismatch")
            continue
        pieces = relative.split("/")
        if len(pieces) < 2 or pieces[0] != "payload":
            raise ValueError("inventory path escapes the payload namespace")
        root = next((root for root in roots if root["name"] == pieces[1]), None)
        if root is None or (root["kind"] == "file" and len(pieces) != 2):
            raise ValueError("inventory path does not belong to a selected root")
        expected_original = str(Path(root["path"]).joinpath(*pieces[2:]))
        if info["original_path"] != expected_original:
            raise ValueError("inventory original path differs from selected root mapping")
    if set(manifest["files"]) & set(manifest["directories"]):
        raise ValueError("file/directory inventory overlap")
    if any(relative not in manifest["directories"] for relative in expected_dirs):
        raise ValueError("archive namespace directories missing")
    for relative in manifest["files"]:
        if str(PurePosixPath(relative).parent) not in manifest["directories"]:
            raise ValueError("parent directory missing from inventory")
    for relative in manifest["directories"]:
        if relative != "." and str(PurePosixPath(relative).parent) not in manifest["directories"]:
            raise ValueError("parent directory missing from directory inventory")
    for root in roots:
        target = f"payload/{root['name']}"
        if target not in (manifest["files"] if root["kind"] == "file" else manifest["directories"]):
            raise ValueError("selected root missing from inventory")
    summary = {"roots": len(roots), "files": len(manifest["files"]),
               "directories": len(manifest["directories"]),
               "bytes": sum(info["bytes"] for info in manifest["files"].values())}
    if manifest["summary"] != summary or any(type(v) is not int for v in manifest["summary"].values()):
        raise ValueError("inventory summary mismatch")
    return manifest


def expected_tree(manifest, manifest_data):
    files = {relative: {key: info[key] for key in ("bytes", "sha256", "mode")}
             for relative, info in manifest["files"].items()}
    files["manifest.json"] = {"bytes": len(manifest_data), "sha256": hashlib.sha256(manifest_data).hexdigest(), "mode": 0o600}
    return {"files": dict(sorted(files.items())),
            "directories": {relative: info["mode"] for relative, info in manifest["directories"].items()}}


def capacity(parent, manifest, manifest_data, reserve_bytes):
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError("capacity reserve must be a nonnegative integer")
    block = os.statvfs(parent).f_frsize or 4096
    required = sum(((info["bytes"] + block - 1) // block) * block for info in manifest["files"].values())
    required += block * (len(manifest["directories"]) + 1) + len(manifest_data) + reserve_bytes
    if shutil.disk_usage(parent).free < required:
        raise ValueError("insufficient capacity for full copy and reserve")


def copy_file(source, destination, expected):
    with open_read(source) as src:
        before = os.fstat(src.fileno())
        fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        digest, count = hashlib.sha256(), 0
        with os.fdopen(fd, "wb") as dst:
            while data := src.read(1024 * 1024):
                dst.write(data); digest.update(data); count += len(data)
            dst.flush(); os.fsync(dst.fileno())
        after = os.fstat(src.fileno())
    if stat_identity(before) != stat_identity(after) or stat_identity(after) != stat_identity(Path(source).lstat()):
        raise ValueError("source changed during copy")
    if count != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
        raise ValueError("copied file differs from frozen inventory")
    os.chmod(destination, expected["mode"])


def copy_payload(stage, manifest, manifest_data, archive_source=None):
    for relative in sorted(manifest["directories"], key=lambda value: (value.count("/"), value)):
        if relative != ".":
            (stage / relative).mkdir(mode=0o700)
    for relative, info in sorted(manifest["files"].items()):
        source = archive_source / relative if archive_source is not None else Path(info["original_path"])
        copy_file(source, stage / relative, info)
    create_file(stage / "manifest.json", manifest_data)
    for relative, info in sorted(manifest["directories"].items(), key=lambda item: item[0].count("/"), reverse=True):
        path = stage if relative == "." else stage / relative
        os.chmod(path, info["mode"])
        directory_sync(path)


def read_archive(archive, expected_manifest_sha):
    archive = canonical(archive)
    if not archive.is_dir():
        raise ValueError("archive must be a directory")
    manifest, data = pinned_json(archive / "manifest.json", expected_manifest_sha)
    validate_manifest(manifest)
    expected = expected_tree(manifest, data)
    if walk_tree(archive) != expected:
        raise ValueError("archive has missing/extra/changed files or directories")
    return manifest, data, expected


def archive_run(inventory_path, expected_manifest_sha, destination, reserve_bytes=512 * 2**20):
    manifest, data = pinned_json(inventory_path, expected_manifest_sha)
    validate_manifest(manifest)
    selection = manifest["selection"]
    validate_selection(selection)
    protected = [Path(root["path"]) for root in manifest["roots"]]
    protected += [Path(inventory_path), Path(manifest["selection_ref"]["path"]), Path(manifest["producer_ref"]["path"])]
    protected += [Path(ref["path"]) for ref in selection["evidence"].values()]
    destination = private_new_output(destination, protected)
    capacity(destination.parent, manifest, data, reserve_bytes)
    with selection_locks(selection):
        def check_sources():
            if selected_inventory(manifest["roots"]) != {key: manifest[key] for key in ("files", "directories")}:
                raise ValueError("selected source inventory changed")
            validate_selection(selection)
            for ref in (manifest["selection_ref"], manifest["producer_ref"]):
                validate_ref(ref)
            if full_ref(Path(__file__).resolve()) != manifest["producer_ref"]:
                raise ValueError("archive producer source differs from frozen inventory")
            if full_ref(inventory_path)["sha256"] != expected_manifest_sha:
                raise ValueError("inventory control changed")
        check_sources()
        stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
        copy_payload(stage, manifest, data)
        check_sources()
        if walk_tree(stage) != expected_tree(manifest, data):
            raise ValueError("staged archive differs from complete frozen inventory")
        method = publish_without_replace(stage, destination, reserve_bytes=reserve_bytes)
        # Root must also observe this process terminal before treating the final
        # directory as an archive success; a terminal check can still fail.
        try:
            check_sources()
            read_archive(destination, expected_manifest_sha)
        except Exception:
            # Preserve failed bytes, while making a later verify refuse this
            # directory at its success-manifest path.
            os.rename(destination / "manifest.json", destination / "manifest.failed.json")
            raise
    return {"status": "archive-copy-verified", "archive": str(destination),
            "manifest_sha256": expected_manifest_sha, "source_originals_preserved": True,
            "publication": publication_receipt(method, stage),
            "standalone_environment_restore": False, **manifest["summary"]}


def verify_archive(archive, expected_manifest_sha):
    manifest, _, _ = read_archive(archive, expected_manifest_sha)
    return {"status": "archive-verified", "manifest_sha256": expected_manifest_sha,
            "standalone_environment_restore": False, **manifest["summary"]}


def restore_run(archive, expected_manifest_sha, destination, reserve_bytes=512 * 2**20):
    archive = canonical(archive)
    manifest, data, expected = read_archive(archive, expected_manifest_sha)
    protected = [archive, *(Path(root["path"]) for root in manifest["roots"])]
    destination = private_new_output(destination, protected)
    capacity(destination.parent, manifest, data, reserve_bytes)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    copy_payload(stage, manifest, data, archive_source=archive)
    if walk_tree(stage) != expected:
        raise ValueError("physical restoration differs from frozen inventory")
    if read_archive(archive, expected_manifest_sha)[2] != expected:
        raise ValueError("archive changed during restoration")
    method = publish_without_replace(stage, destination, reserve_bytes=reserve_bytes)
    try:
        if walk_tree(destination) != expected:
            raise ValueError("published restoration differs from frozen inventory")
        read_archive(archive, expected_manifest_sha)
    except Exception:
        os.rename(destination / "manifest.json", destination / "manifest.failed.json")
        raise
    return {"status": "physical-restore-verified", "restoration": str(destination),
            "manifest_sha256": expected_manifest_sha, "original_reference_paths_preserved": True,
            "publication": publication_receipt(method, stage),
            "standalone_environment_restore": False, **manifest["summary"]}


def nas_guard(nas_root):
    nas_root = canonical(nas_root)
    mount = Path("/mnt/storage")
    if not os.path.ismount(mount) or not nas_root.is_relative_to(mount):
        raise ValueError("existing /mnt/storage mount is required; no fallback directory is created")
    info = nas_root.stat()
    if not nas_root.is_dir() or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError("NAS project boundary must be an existing owned 0700 directory")
    return nas_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inventory")
    inventory.add_argument("--selection", type=Path, required=True)
    inventory.add_argument("--expected-selection-sha256", required=True)
    inventory.add_argument("--output", type=Path, required=True)
    archive = commands.add_parser("archive")
    archive.add_argument("--inventory", type=Path, required=True)
    for command in (archive, commands.add_parser("verify"), commands.add_parser("restore")):
        command.add_argument("--expected-manifest-sha256", required=True)
        command.add_argument("--nas-root", type=Path, default=DEFAULT_NAS_ROOT)
        if command is not archive:
            command.add_argument("--archive", type=Path, required=True)
        if command.prog.endswith(("archive", "restore")):
            command.add_argument("--destination", type=Path, required=True)
            command.add_argument("--reserve-bytes", type=int, default=512 * 2**20)
        if command.prog.endswith("restore"):
            command.add_argument("--ssd-root", type=Path, default=DEFAULT_SSD_ROOT)
    args = parser.parse_args()
    if args.command == "inventory":
        result = build_inventory(args.selection, args.expected_selection_sha256, args.output)
    else:
        nas = nas_guard(args.nas_root)
        target = canonical(args.destination, existing=False) if args.command == "archive" else canonical(args.archive)
        if not target.is_relative_to(nas):
            raise ValueError("archive/destination must be inside the existing NAS project boundary")
        if args.command == "archive":
            result = archive_run(args.inventory, args.expected_manifest_sha256, args.destination, args.reserve_bytes)
        elif args.command == "verify":
            result = verify_archive(args.archive, args.expected_manifest_sha256)
        else:
            ssd = canonical(args.ssd_root)
            destination = canonical(args.destination, existing=False)
            if (not destination.is_relative_to(ssd) or ssd.stat().st_dev == nas.stat().st_dev
                    or destination.is_relative_to(nas)):
                raise ValueError("restoration requires a fresh destination on the existing SSD filesystem")
            result = restore_run(args.archive, args.expected_manifest_sha256, destination, args.reserve_bytes)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
