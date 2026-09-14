#!/usr/bin/env python3
"""Build pinned tools and import a locally supplied teacher; no sudo required."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
LOCK = json.loads((REPO / "config/toolchain.lock.json").read_text())
DEFAULT_RUNTIME = Path.home() / ".local/share/sekirei-weight2" / LOCK["runtime_profile"]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(args, **kwargs):
    print("+", " ".join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def capture(args, **kwargs):
    return subprocess.check_output(list(map(str, args)), text=True, **kwargs).strip()


def verify_weight(runtime):
    spec = LOCK["primary_teacher"]
    weight = runtime / "models" / spec["id"] / "nn.bin"
    if weight.stat().st_size != spec["weight_bytes"]:
        raise RuntimeError(f"teacher weight size mismatch: {weight}")
    if sha256(weight) != spec["weight_sha256"]:
        raise RuntimeError(f"teacher weight hash mismatch: {weight}")
    return weight


def import_weight(runtime, archive):
    spec = LOCK["primary_teacher"]
    if sha256(archive) != spec["archive_sha256"]:
        raise RuntimeError("archive hash mismatch; no extraction performed")
    target = runtime / "models" / spec["id"]
    target.mkdir(parents=True, exist_ok=True)
    if (target / "nn.bin").exists():
        verify_weight(runtime)
        print("Teacher already imported and verified.")
        return
    # Extract just the required member from the verified archive. Never unpack paths.
    with tempfile.TemporaryDirectory(prefix="teacher-", dir=runtime) as temp:
        staged = Path(temp) / "nn.bin"
        with staged.open("wb") as output:
            run(["7z", "e", "-so", archive, spec["archive_member"]], stdout=output)
        if staged.stat().st_size != spec["weight_bytes"]:
            raise RuntimeError("extracted teacher size mismatch")
        if sha256(staged) != spec["weight_sha256"]:
            raise RuntimeError("extracted teacher hash mismatch")
        staged.rename(target / "nn.bin")


def parse_7z_slt(text):
    """Parse 7z's stable key/value listing into entry dictionaries."""
    records, record = [], {}
    for line in text.splitlines():
        if not line.strip():
            if record:
                records.append(record)
                record = {}
            continue
        if " = " in line:
            key, value = line.split(" = ", 1)
            record[key] = value
    if record:
        records.append(record)
    return records


def corpus_archive_spec(archive_hash):
    matches = []
    for corpus_id, corpus in LOCK["teacher_corpora"].items():
        for archive in corpus["archives"]:
            if archive["archive_sha256"] == archive_hash:
                matches.append((corpus_id, corpus, archive))
    if len(matches) != 1:
        raise RuntimeError(f"archive is not a unique pinned teacher corpus: {archive_hash}")
    return matches[0]


def archive_pack_members(archive, prefix):
    listing = capture(["7z", "l", "-slt", archive])
    members = []
    for entry in parse_7z_slt(listing):
        name = entry.get("Path", "")
        path = PurePosixPath(name)
        if not name.startswith(prefix) or path.suffix.lower() != ".pack":
            continue
        if path.is_absolute() or ".." in path.parts or entry.get("Folder") == "+":
            raise RuntimeError(f"unsafe pack member in verified archive: {name}")
        try:
            size = int(entry["Size"])
        except (KeyError, ValueError) as error:
            raise RuntimeError(f"missing pack size in archive listing: {name}") from error
        members.append((name, size))
    if not members:
        raise RuntimeError(f"no .pack members under pinned prefix {prefix!r}")
    return sorted(members)


def write_corpus_manifest(corpus_root, corpus_id, corpus, archive_record):
    path = corpus_root / "manifest.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    archives = {item["archive_sha256"]: item for item in old.get("archives", [])}
    archives[archive_record["archive_sha256"]] = archive_record
    archives_list = sorted(archives.values(), key=lambda item: item["archive_sha256"])
    unique = {}
    for imported in archives_list:
        for item in imported["files"]:
            stored = unique.setdefault(item["sha256"], {
                "sha256": item["sha256"], "bytes": item["bytes"],
                "path": item["path"], "sources": [],
            })
            stored["sources"].append({
                "archive_sha256": imported["archive_sha256"],
                "member": item["member"],
            })
    manifest = {
        "schema_version": 1,
        "corpus_id": corpus_id,
        "teacher_id_claim": corpus["teacher_id"],
        "requested_nodes_claim": corpus["requested_nodes"],
        "format": corpus["format"],
        "archives": archives_list,
        "unique_files": sorted(unique.values(), key=lambda item: item["sha256"]),
    }
    staged = corpus_root / ".manifest.json.tmp"
    staged.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    staged.replace(path)
    return path, manifest


def import_corpus(runtime, archive):
    archive_hash = sha256(archive)
    corpus_id, corpus, archive_spec = corpus_archive_spec(archive_hash)
    members = archive_pack_members(archive, archive_spec["member_prefix"])
    corpus_root = runtime / "data" / "teachers" / corpus_id
    packs = corpus_root / "packs"
    packs.mkdir(parents=True, exist_ok=True)
    required = sum(size for _, size in members) + 2**30
    if shutil.disk_usage(corpus_root).free < required:
        raise RuntimeError(f"corpus import needs at least {required / 2**30:.1f} GiB free")

    imported = []
    with tempfile.TemporaryDirectory(prefix="corpus-", dir=corpus_root) as temp:
        for index, (member, expected_size) in enumerate(members):
            staged = Path(temp) / f"{index}.pack"
            with staged.open("wb") as output:
                run(["7z", "e", "-so", archive, member], stdout=output)
            if staged.stat().st_size != expected_size:
                raise RuntimeError(f"extracted pack size mismatch: {member}")
            digest = sha256(staged)
            target = packs / f"{digest}.pack"
            if target.exists():
                if target.stat().st_size != expected_size or sha256(target) != digest:
                    raise RuntimeError(f"existing content-addressed pack is corrupt: {target}")
            else:
                staged.rename(target)
            imported.append({
                "member": member,
                "bytes": expected_size,
                "sha256": digest,
                "path": str(target.relative_to(corpus_root)),
            })

    archive_record = {
        "archive_sha256": archive_hash,
        "archive_name": archive.name,
        "member_prefix": archive_spec["member_prefix"],
        "source_article": archive_spec["source_article"],
        "files": imported,
    }
    manifest_path, manifest = write_corpus_manifest(
        corpus_root, corpus_id, corpus, archive_record)
    duplicate_count = sum(len(item["sources"]) - 1 for item in manifest["unique_files"])
    print(f"Corpus manifest: {manifest_path}")
    print(f"Unique packs: {len(manifest['unique_files'])}; duplicate copies: {duplicate_count}")


def install_audit_deps(runtime):
    requirements = REPO / "config/audit-requirements.txt"
    venv = runtime / "venv"
    if not (venv / "bin/python").exists():
        run([sys.executable, "-m", "venv", venv])
    python = venv / "bin/python"
    run([python, "-m", "pip", "install", "--only-binary=:all:",
         "--requirement", requirements])
    manifest = {
        "requirements_sha256": sha256(requirements),
        "python": capture([python, "--version"]),
        "packages": capture([python, "-m", "pip", "freeze", "--all"]).splitlines(),
    }
    path = runtime / "audit-environment.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Audit environment:", path)


def source(runtime, name):
    spec = LOCK["sources"][name]
    dest = runtime / "sources" / name
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "init", dest])
        run(["git", "-C", dest, "remote", "add", "origin", spec["url"]])
        run(["git", "-C", dest, "fetch", "--depth", "1", "origin", spec["commit"]])
        run(["git", "-C", dest, "checkout", "--detach", spec["commit"]])
    if capture(["git", "-C", dest, "rev-parse", "HEAD"]) != spec["commit"]:
        raise RuntimeError(f"{dest}: wrong revision; use a new runtime for different pins")
    if capture(["git", "-C", dest, "remote", "get-url", "origin"]) != spec["url"]:
        raise RuntimeError(f"{dest}: unexpected origin")
    run(["git", "-C", dest, "diff", "--exit-code", "HEAD", "--"], stdout=subprocess.DEVNULL)
    return dest


def build(runtime, jobs):
    env = dict(os.environ, CARGO_BUILD_JOBS=str(jobs), CARGO_INCREMENTAL="0",
               CARGO_PROFILE_RELEASE_DEBUG="0", RUSTFLAGS=LOCK["rustflags"])
    for name in ("sekirei", "shogiesa"):
        dest = source(runtime, name)
        command = ["cargo", "build", "--locked", "--release", "--target-dir", runtime / "build" / name]
        for package in LOCK["sources"][name]["packages"]:
            command += ["-p", package]
        run(command, cwd=dest, env=env)
    dest = source(runtime, "yaneuraou")
    spec = LOCK["sources"]["yaneuraou"]
    run(["make", f"-j{jobs}", "normal", "COMPILER=g++", "PYTHON=python3",
         f"TARGET_CPU={spec['target_cpu']}", f"YANEURAOU_EDITION={spec['edition']}"], cwd=dest / "source")
    binaries = {
        "sekirei": runtime / "build/sekirei/release/sekirei",
        "sekirei-train": runtime / "build/sekirei/release/train",
        "shogiesa": runtime / "build/shogiesa/release/shogiesa",
        "yaneuraou": dest / "source/YaneuraOu-by-gcc",
    }
    bindir = runtime / "bin"
    bindir.mkdir(exist_ok=True)
    for name, binary in binaries.items():
        link = bindir / name
        if link.is_symlink() and link.resolve() == binary:
            continue
        if link.exists() or link.is_symlink():
            raise RuntimeError(f"refusing to replace existing binary/link: {link}")
        link.symlink_to(binary)
    manifest = {
        "lock_sha256": sha256(REPO / "config/toolchain.lock.json"),
        "sources": LOCK["sources"], "rustflags": LOCK["rustflags"],
        "rustc": capture(["rustc", "--version"]),
        "cargo": capture(["cargo", "--version"]),
        "g++": capture(["g++", "--version"]).splitlines()[0],
        "binaries": {name: {"path": str(path), "sha256": sha256(path)} for name, path in binaries.items()},
    }
    (runtime / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Build manifest:", runtime / "build-manifest.json")


def doctor(runtime):
    required = ("git", "cargo", "rustc", "python3", "g++", "make", "7z")
    missing = [name for name in required if shutil.which(name) is None]
    for name in required:
        print(f"{name}: {shutil.which(name) or 'MISSING'}")
    print(f"Free disk: {shutil.disk_usage(runtime).free / 2**30:.1f} GiB")
    if missing:
        raise RuntimeError("missing tools: " + ", ".join(missing))
    if sys.platform != "linux" or os.uname().machine != "x86_64":
        raise RuntimeError("this build recipe targets Linux x86_64")
    flags = Path("/proc/cpuinfo").read_text()
    if "avx2" not in flags or "bmi2" not in flags:
        raise RuntimeError("AVX2 and BMI2 are required for the pinned build")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=(
        "doctor", "build", "import-teacher", "import-corpus", "audit-deps"))
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.action in ("import-teacher", "import-corpus") and not args.archive:
        parser.error(f"{args.action} requires --archive")
    runtime = args.runtime.expanduser().resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    with (runtime / ".prepare.lock").open("a") as lockfile:
        fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        doctor(runtime)
        if args.action == "build":
            if shutil.disk_usage(runtime).free < 4 * 2**30:
                raise RuntimeError("build requires at least 4 GiB free disk")
            build(runtime, args.jobs)
        elif args.action == "import-teacher":
            import_weight(runtime, args.archive.expanduser().resolve())
        elif args.action == "import-corpus":
            import_corpus(runtime, args.archive.expanduser().resolve())
        elif args.action == "audit-deps":
            install_audit_deps(runtime)
        print("OK")


if __name__ == "__main__":
    main()
