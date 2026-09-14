#!/usr/bin/env python3
"""Build pinned tools and import a locally supplied teacher; no sudo required."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
LOCK = json.loads((REPO / "config/toolchain.lock.json").read_text())
DEFAULT_RUNTIME = Path.home() / ".local/share/sekirei-weight2"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(args, **kwargs):
    print("+", " ".join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def capture(args, **kwargs):
    return subprocess.check_output(list(map(str, args)), text=True, **kwargs).strip()


def verify_weight(runtime):
    weight = runtime / "models/suisho11plus/nn.bin"
    if sha256(weight) != LOCK["suisho11plus"]["weight_sha256"]:
        raise RuntimeError(f"teacher weight hash mismatch: {weight}")
    return weight


def import_weight(runtime, archive):
    spec = LOCK["suisho11plus"]
    if sha256(archive) != spec["archive_sha256"]:
        raise RuntimeError("archive hash mismatch; no extraction performed")
    target = runtime / "models/suisho11plus"
    target.mkdir(parents=True, exist_ok=True)
    if (target / "nn.bin").exists():
        verify_weight(runtime)
        print("Teacher already imported and verified.")
        return
    # Extract just the required member from the verified archive. Never unpack paths.
    with tempfile.TemporaryDirectory(prefix="teacher-", dir=runtime) as temp:
        staged = Path(temp) / "nn.bin"
        with staged.open("wb") as output:
            run(["7z", "e", "-so", archive, "nn.bin"], stdout=output)
        if sha256(staged) != spec["weight_sha256"]:
            raise RuntimeError("extracted teacher hash mismatch")
        staged.rename(target / "nn.bin")


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
    parser.add_argument("action", choices=("doctor", "build", "import-teacher"))
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.action == "import-teacher" and not args.archive:
        parser.error("import-teacher requires --archive")
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
        print("OK")


if __name__ == "__main__":
    main()
