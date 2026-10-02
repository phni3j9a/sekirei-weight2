#!/usr/bin/env python3
"""Build the pinned trainer with an explicit external-label cache identity."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from prepare import DEFAULT_RUNTIME, LOCK, REPO, sha256
from benchmark import atomic_write_json, nonblocking_lock


def prepare(output):
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with nonblocking_lock(output / ".build.lock", exclusive=True):
        patch = REPO / "patches/sekirei-train-external-labels.patch"
        spec = LOCK["sources"]["sekirei"]
        identity = {"upstream_commit": spec["commit"], "patch_sha256": sha256(patch),
                    "rustflags": LOCK["rustflags"]}
        manifest_path = output / "build-manifest.json"
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text())
            if any(old.get(k) != v for k, v in identity.items()):
                raise RuntimeError("trainer build identity changed; use a new output directory")
            if sha256(Path(old["binary"])) != old["binary_sha256"]:
                raise RuntimeError("trainer binary hash mismatch")
            return old
        if shutil.disk_usage(output).free < 4 * 2**30:
            raise RuntimeError("trainer build requires 4 GiB free")
        source = output / "source"
        if source.exists():
            raise RuntimeError("incomplete build source exists; inspect it before resuming")
        subprocess.run(["git", "clone", "--no-hardlinks", "--no-checkout",
                        str(DEFAULT_RUNTIME / "sources/sekirei"), str(source)], check=True)
        subprocess.run(["git", "checkout", "--detach", spec["commit"]], cwd=source, check=True)
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=source, check=True)
        subprocess.run(["git", "apply", str(patch)], cwd=source, check=True)
        changed = subprocess.check_output(["git", "diff", "--name-only"], cwd=source, text=True).splitlines()
        if changed != ["crates/sekirei-train/src/main.rs"]:
            raise RuntimeError("trainer patch touched unexpected files")
        env = dict(os.environ, CARGO_TARGET_DIR=str(output / "build"),
                   RUSTFLAGS=LOCK["rustflags"], CARGO_BUILD_JOBS="2")
        subprocess.run(["cargo", "build", "--release", "--locked", "-j", "2",
                        "-p", "sekirei-train", "--bin", "train"], cwd=source, env=env, check=True)
        binary = output / "build/release/train"
        manifest = dict(identity, binary=str(binary), binary_sha256=sha256(binary),
                        source_main_sha256=sha256(source / changed[0]),
                        rustc=subprocess.check_output(["rustc", "--version"], text=True).strip(),
                        purpose="external pack labels; inference engine unmodified")
        atomic_write_json(manifest_path, manifest)
        return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(prepare(parser.parse_args().output), indent=2))
