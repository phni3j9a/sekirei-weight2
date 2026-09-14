#!/usr/bin/env python3
"""Exercise pinned USI engines and shogiesa; this is not a model benchmark."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import tempfile
import threading
import time

from prepare import DEFAULT_RUNTIME, LOCK, REPO, capture, run, sha256, verify_weight


class Engine:
    def __init__(self, binary, cwd, timeout, log):
        self.timeout = timeout
        self.log = log
        self.lines = queue.Queue()
        self.process = subprocess.Popen(
            [str(binary)], cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True,
            env=dict(os.environ, RAYON_NUM_THREADS="1"),
        )
        self.reader = threading.Thread(target=self.read, daemon=True)
        self.reader.start()

    def read(self):
        try:
            for line in self.process.stdout:
                self.lines.put(line.rstrip())
        finally:
            self.lines.put(None)

    def send(self, line):
        self.log.write("> " + line + "\n")
        self.log.flush()
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()

    def until(self, predicate):
        deadline = time.monotonic() + self.timeout
        received = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("USI deadline exceeded")
            try:
                line = self.lines.get(timeout=remaining)
            except queue.Empty as error:
                raise TimeoutError("USI deadline exceeded") from error
            if line is None:
                raise RuntimeError("USI engine exited before completing the command")
            self.log.write("< " + line + "\n")
            self.log.flush()
            if re.search(r"failed|file not found|error!|unknown option|no such option|panic", line, re.I):
                raise RuntimeError(f"USI error: {line}")
            received.append(line)
            if predicate(line):
                return received

    def close(self):
        try:
            if self.process.poll() is None:
                self.send("quit")
                self.process.wait(timeout=5)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            if self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=5)
        finally:
            self.process.stdin.close()
            self.reader.join(timeout=5)
            self.process.stdout.close()
        if self.process.returncode != 0:
            raise RuntimeError(f"USI engine exited with {self.process.returncode}")


def parse_result(lines, side, *, allow_bounds=False):
    best = next(line.split()[1] for line in reversed(lines) if line.startswith("bestmove "))
    if not re.fullmatch(r"(?:[1-9][a-i][1-9][a-i]\+?|[PLNSGBR]\*[1-9][a-i])", best):
        raise RuntimeError(f"unexpected smoke bestmove: {best}")
    infos = []
    for line in lines:
        if not line.startswith("info ") or " score " not in line or " pv " not in line:
            continue
        tokens = line.split()
        if "multipv" in tokens and tokens[tokens.index("multipv") + 1] != "1":
            continue
        infos.append(tokens)
    if not infos:
        raise RuntimeError("no scored PV")
    tokens = infos[-1]
    bound = next((name for name in ("lowerbound", "upperbound") if name in tokens), "exact")
    if bound != "exact" and not allow_bounds:
        raise RuntimeError("final scored PV is a bound")
    if tokens[tokens.index("pv") + 1] != best:
        raise RuntimeError("final scored PV and bestmove disagree")
    score_at = tokens.index("score")
    kind, value = tokens[score_at + 1:score_at + 3]
    if kind != "cp":
        raise RuntimeError(f"expected finite cp score for smoke position, got {kind}")
    fields = {name: int(tokens[tokens.index(name) + 1]) for name in ("nodes", "depth")}
    sente_bound = bound
    if side == "white" and bound != "exact":
        sente_bound = "lowerbound" if bound == "upperbound" else "upperbound"
    return dict(fields, bestmove=best, reported_cp_stm=int(value),
                score_bound_stm=bound, score_bound_sente=sente_bound,
                score_cp_stm=int(value) if bound == "exact" else None,
                score_cp_sente=int(value) * (1 if side == "black" else -1) if bound == "exact" else None)


def probe(runtime, name, options, config, output):
    with (output / f"{name}.usi.log").open("w") as log:
        engine = Engine(runtime / "bin" / name, output, config["timeout_seconds"], log)
        try:
            engine.send("usi")
            handshake = engine.until(lambda line: line == "usiok")
            advertised = {match[1] for line in handshake
                          if (match := re.match(r"option name (.+?) type ", line))}
            if missing := options.keys() - advertised:
                raise RuntimeError(f"{name}: unsupported options {sorted(missing)}")
            for key, value in options.items():
                engine.send(f"setoption name {key} value {value}")
            engine.send("isready")
            startup = engine.until(lambda line: line == "readyok")
            engine.send("usinewgame")
            engine.send("position " + config["position"])
            started = time.monotonic()
            engine.send(f"go nodes {config['requested_nodes']}")
            lines = engine.until(lambda line: line.startswith("bestmove "))
            # A node-limited search may finish during an aspiration fail-low/high.
            # Preserve its bound for environment diagnostics; never create a point score.
            result = parse_result(lines, config["side_to_move"], allow_bounds=True)
            # Smoke sanity envelope, not the eventual benchmark acceptance policy.
            if not config["requested_nodes"] * 0.98 <= result["nodes"] <= config["requested_nodes"] * 1.02:
                raise RuntimeError(f"unexpected node count: {result['nodes']}")
            result.update(wall_seconds=round(time.monotonic() - started, 3),
                          options=options, requested_nodes=config["requested_nodes"],
                          engine_id=[line for line in handshake if line.startswith("id ")],
                          startup_warnings=[line for line in startup if "warning" in line.lower()])
            return result
        finally:
            engine.close()


def check_shogiesa(runtime, output, teacher_options, weight, config):
    binary = runtime / "bin/shogiesa"
    positions, labeled = output / "positions.jsonl", output / "labeled.jsonl"
    run([binary, "extract", "--input", REPO / "tests/fixtures/opening.csa",
         "--min-ply", "3", "--max-ply", "3", "--out", positions], cwd=output)
    records = [json.loads(line) for line in positions.read_text().splitlines()]
    if len(records) != 1:
        raise RuntimeError(f"expected one extracted smoke position, got {len(records)}")
    command = [binary, "label", "--input", positions, "--engine", runtime / "bin/yaneuraou",
               "--nodes", str(config["requested_nodes"]), "--jobs", "1", "--multipv", "1", "--usi-strict",
               "--timeout-ms", str(config["timeout_seconds"] * 1000),
               "--weight-file", weight, "--out", labeled,
               "--manifest", output / "label-manifest.json"]
    for key, value in teacher_options.items():
        command += ["--engine-option", f"{key}={value}"]
    run(command, cwd=output, timeout=config["timeout_seconds"] + 30)
    run([binary, "validate", "--input", labeled, "--strict"], cwd=output)
    rows = [json.loads(line) for line in labeled.read_text().splitlines()]
    if len(rows) != 1 or len(rows[0]["observations"]) != 1:
        raise RuntimeError("shogiesa label coverage mismatch")
    observation = rows[0]["observations"][0]
    if observation.get("requested_nodes") != config["requested_nodes"]:
        raise RuntimeError("shogiesa node budget mismatch")
    if observation.get("weight_sha256") != LOCK["suisho11plus"]["weight_sha256"]:
        raise RuntimeError("shogiesa weight attribution mismatch")
    if observation.get("was_timeout_salvaged"):
        raise RuntimeError("shogiesa salvaged a timed out search")
    if observation.get("score_bound") not in ("exact", "lowerbound", "upperbound"):
        raise RuntimeError("shogiesa returned an unrecognized score bound")
    if observation.get("score", {}).get("kind") != "cp":
        raise RuntimeError("shogiesa did not return a finite cp score")
    if not config["requested_nodes"] * 0.98 <= (observation.get("nodes") or 0) <= config["requested_nodes"] * 1.02:
        raise RuntimeError("shogiesa observation has unexpected node count")
    return {"version": capture([binary, "--version"]), "observations": len(rows),
            "score_bound": observation["score_bound"], "nodes": observation["nodes"],
            "history_mode": "standalone_sfen_not_benchmark"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    args = parser.parse_args()
    runtime = args.runtime.expanduser().resolve()
    config = json.loads((REPO / "config/smoke.json").read_text())
    with (runtime / ".prepare.lock").open("a") as lockfile:
        fcntl.flock(lockfile, fcntl.LOCK_SH | fcntl.LOCK_NB)
        weight = verify_weight(runtime)
        manifest = json.loads((runtime / "build-manifest.json").read_text())
        if manifest["lock_sha256"] != sha256(REPO / "config/toolchain.lock.json"):
            raise RuntimeError("build manifest does not match current toolchain lock")
        for name, binary in manifest["binaries"].items():
            if sha256(runtime / "bin" / name) != binary["sha256"]:
                raise RuntimeError(f"binary hash mismatch: {name}")
        (runtime / "runs").mkdir(exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="environment-", dir=runtime / "runs"))
        print("Logs:", output, flush=True)
        teacher_options = dict(config["teacher_options"], EvalDir=str(weight.parent))
        result = {
            "purpose": "environment_smoke_not_model_quality",
            "position": config["position"], "side_to_move": config["side_to_move"],
            "build_manifest": manifest, "weight_sha256": sha256(weight),
            "config_sha256": sha256(REPO / "config/smoke.json"),
            "sekirei_model": "material_fallback_no_trained_weights",
        }
        result["sekirei"] = probe(runtime, "sekirei", config["sekirei_options"], config, output)
        result["teacher"] = probe(runtime, "yaneuraou", teacher_options, config, output)
        result["shogiesa"] = check_shogiesa(runtime, output, teacher_options, weight, config)
        (output / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({key: result[key] for key in ("sekirei", "teacher", "shogiesa")}, indent=2))
        print("PASS:", output / "summary.json")


if __name__ == "__main__":
    main()
