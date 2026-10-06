#!/usr/bin/env python3
"""Source-only preparation consumer. No model, compiler, engine or subprocess.

Actual prepare/recount invocations are queued behind Root's serial heavy gates.
Sample policy is fixed by original row number before reading model predictions.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct

MODE = "white-view-diverse-games-seed42-e3-v1"
PROFILE_SHA = "f49ff28c6ff16001c3c54ec3f4679a6bbcf4a38d086f439751c029c66f7efceb"
MANIFEST_SHA = "59bb51b46d7b886099442bb8df93d7ba224f6c213b8cc903d668bd6d38f676d8"
MASTER_SHA = "7c2f9793bc1e3660f0202631cce2c3ae1c85b6cf7293e9285e16e60a60772f88"
NATIVE_SHA = "44d1b412da22688542a406599dd40b40bc3243285aaaeae67b21d003db9a0429"
COUNTS = {"train": 112681, "holdout": 5895}
POLICY = "ascending floor(i*(N-1)/255), i=0..255, original split row order"

def require(ok, message):
    if not ok:
        raise ValueError(message)

def exact(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b

def unique(raw):
    def pairs(items):
        obj = {}
        for key, value in items:
            require(key not in obj, "duplicate JSON key")
            obj[key] = value
        return obj
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))

def fullref(value):
    require(type(value) is dict and set(value) == {"path", "bytes", "sha256"}, "fullref exact keys")
    require(type(value["path"]) is str, "path string")
    path = Path(value["path"])
    require(type(value["path"]) is str and path.is_absolute() and str(path) == value["path"]
            and ".." not in path.parts and "/./" not in value["path"], "canonical absolute path")
    require(type(value["bytes"]) is int and 0 < value["bytes"] <= 512 * 1024 * 1024, "bounded byte size")
    require(type(value["sha256"]) is str and len(value["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in value["sha256"]), "lower SHA256")
    return path

def read_ref(value):
    path = fullref(value)
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and path.resolve(strict=True) == path, "physical regular input")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        require((before.st_dev, before.st_ino) == (opened.st_dev, opened.st_ino), "input rebound")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            raw = handle.read(value["bytes"] + 1)
        require(len(raw) == value["bytes"] and hashlib.sha256(raw).hexdigest() == value["sha256"], "raw pin differs")
        fingerprint = lambda s: (s.st_dev,s.st_ino,s.st_mode,s.st_uid,s.st_gid,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        require(fingerprint(before) == fingerprint(path.lstat()) and fingerprint(opened) == fingerprint(os.fstat(fd)),
                "input changed while reading")
        return raw
    finally:
        os.close(fd)

def indices(count, selected=256):
    require(type(count) is int and type(selected) is int and 2 <= selected <= count, "bounded sample count")
    answer = [i * (count - 1) // (selected - 1) for i in range(selected)]
    require(len(set(answer)) == selected and answer[0] == 0 and answer[-1] == count - 1, "sample index contract")
    return answer

def select_rows(raw, count, selected=256):
    require(raw and raw.endswith(b"\n") and b"\r" not in raw, "canonical LF JSONL")
    lines = raw[:-1].split(b"\n")
    require(len(lines) == count and all(lines), "full split physical row count")
    return [(index, unique(lines[index])) for index in indices(count, selected)]

def sample_rows(position_raw, count, games, split, selected=256):
    require(split in COUNTS, "train/frozen holdout only")
    answer, semantics = [], set()
    for index, row in select_rows(position_raw, count, selected):
        require(type(row) is dict and set(row) == {"schema_version", "sfen", "source", "tags"}, "position exact keys")
        require(type(row["schema_version"]) is int and row["schema_version"] == 1, "position version")
        source, tags, sfen = row["source"], row["tags"], row["sfen"]
        require(type(source) is dict and set(source) == {"kind", "path", "ply"}, "source keys")
        require(source["kind"] == "gensfen-pack" and source["path"] in games
                and games[source["path"]] == split, "game split membership")
        require(type(tags) is dict and set(tags) == {"phase", "side_to_move"}, "tag keys")
        require(type(sfen) is str and "\n" not in sfen and "\r" not in sfen and len(sfen.split(" ")) == 4, "single SFEN")
        parts = sfen.split(" ")
        require(parts[1] in {"b", "w"} and type(source["ply"]) is int and source["ply"] >= 16
                and parts[3] == str(source["ply"]), "STM/source ply")
        require(tags == {"phase": "middlegame", "side_to_move": "black" if parts[1] == "b" else "white"}, "fixed phase/STM")
        semantic = " ".join(parts[:3])
        require(semantic not in semantics, "duplicate selected board")
        semantics.add(semantic)
        answer.append({"split": split, "row_index": index, "position": row})
    return answer

def f32(bits):
    require(type(bits) is str and len(bits) == 8 and all(c in "0123456789abcdef" for c in bits), "f32 bits")
    value = struct.unpack("<f", bytes.fromhex(bits)[::-1])[0]
    require(math.isfinite(value) and abs(value) < 899000, "finite ordinary cp")
    return value

def aggregate(records, selected=256):
    require(len(records) == 2 * selected, "complete bounded probe count")
    result = {}
    for offset, split in enumerate(("train", "holdout")):
        totals = []
        ft_gaps, integer_gaps = [], []
        for local, row in enumerate(records[offset * selected:(offset + 1) * selected]):
            require(type(row) is dict and set(row) == {"split", "row_index", "raw_master_cp_f32_bits",
                    "native_dequant_cp_f32_bits", "native_core_cp", "all_state_bits_unchanged"}, "probe exact keys")
            require(row["split"] == split and type(row["row_index"]) is int
                    and row["row_index"] == indices(COUNTS[split], selected)[local], "probe order/index")
            require(row["all_state_bits_unchanged"] is True and type(row["native_core_cp"]) is int
                    and abs(row["native_core_cp"]) < 899000, "state/native result contract")
            raw, dequant = f32(row["raw_master_cp_f32_bits"]), f32(row["native_dequant_cp_f32_bits"])
            total = raw - row["native_core_cp"]
            totals.append(total); ft_gaps.append(raw - dequant); integer_gaps.append(dequant - row["native_core_cp"])
        summary = lambda values: {"count":len(values), "mean_signed_cp":math.fsum(values)/len(values),
                                  "mean_absolute_cp":math.fsum(abs(x) for x in values)/len(values),
                                  "max_absolute_cp":max(abs(x) for x in values)}
        result[split] = {"raw_master_minus_native_core":summary(totals),
                         "raw_master_minus_native_dequant":summary(ft_gaps),
                         "native_dequant_minus_native_core":summary(integer_gaps)}
    return {"schema":"sekirei.raw-master-quantization-sampled-summary.v1", "mode":MODE,
            "sample_count":2*selected, "splits":result, "stm_cp_divisor":64,
            "universal_bound_proved":False, "adoption_claimed":False,
            "development_used":False, "final_used":False, "teacher_queried":False,
            "optimizer_updates":0, "export_mutation":False}

def records_from_log(raw):
    require(raw and raw.endswith(b"\n") and b"\r" not in raw, "canonical complete diagnostic log")
    records, snapshots, terminal = [], {}, 0
    keys = {"schema","float_policy","environment_variable","environment_value","mxcsr_raw_bits",
            "mxcsr_control_bits","mxcsr_status_bits","mxcsr_status_mask","required_control_bits"}
    lines = raw.splitlines()
    require(lines[0] == b"Training floats: FTZ/DAZ enabled (MXCSR bits 15/6)",
            "exact startup banner required once as first line")
    for line in lines[1:]:
        if line.startswith(b"{"):
            records.append(unique(line))
        elif line.startswith((b"float_before=",b"float_after=")):
            name, value = line.split(b"=",1); name=name.decode()
            require(name not in snapshots, "duplicate FP observation")
            snapshot=unique(value)
            require(type(snapshot) is dict and set(snapshot)==keys,"FP exact keys")
            require(snapshot["schema"]=="sekirei.white-view-paired-nonlinear-float-snapshot.v1"
                    and snapshot["float_policy"]=="x86-ftz-daz"
                    and snapshot["environment_variable"]=="SEKIREI_TRAIN_FTZ_DAZ"
                    and snapshot["environment_value"]=="1", "FP policy/environment")
            for key in keys-{ "schema","float_policy","environment_variable","environment_value"}:
                require(type(snapshot[key]) is int and 0<=snapshot[key]<2**32,"FP strict word")
            require(snapshot["mxcsr_status_mask"]==63 and snapshot["required_control_bits"]==0x9fc0
                    and snapshot["mxcsr_control_bits"]==0x9fc0
                    and snapshot["mxcsr_raw_bits"] & ~63 == 0x9fc0
                    and snapshot["mxcsr_status_bits"]==snapshot["mxcsr_raw_bits"] & 63, "FP control/status readback")
            snapshots[name]=snapshot
        elif line==b"complete raw-master/native sampled count512; step338043 unchanged; optimizer_updates0; no teacher/search/export write":
            terminal+=1
        else:
            raise ValueError("unexpected diagnostic log line")
    require(len(records)==512 and set(snapshots)=={"float_before","float_after"} and terminal==1,
            "complete512/FP/terminal evidence required")
    return records, snapshots

def write_new(path, value):
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(raw)
    return {"path":str(path), "bytes":len(raw), "sha256":hashlib.sha256(raw).hexdigest()}

def prepare(plan, output):
    require(plan["mode"] == MODE and plan["status"] == "source-only-prepared", "own first candidate mode/phase")
    require(not output.exists() and output.is_absolute() and output.parent.resolve(strict=True) == output.parent,
            "fresh canonical private output")
    refs = plan["inputs"]
    completion = unique(read_ref(refs["completion"]))
    require(completion["status"] == "complete" and completion["mode"] == MODE and completion["poisoned"] is False,
            "completed own training required")
    require(exact(completion["outputs"]["checkpoint"], refs["master"])
            and exact(completion["outputs"]["native03"], refs["native03"])
            and exact(completion["outputs"]["export_stage"], refs["export_stage"]), "completion output pins")
    require(refs["master"]["sha256"] == MASTER_SHA and refs["native03"]["sha256"] == NATIVE_SHA, "own model pair pins")
    stage = unique(read_ref(refs["export_stage"]))
    require(stage["status"] == "three-epochs-export-readback-complete" and stage["global_step"] == 338043 and stage["epochs_completed"] == 3
            and stage["positions_per_epoch"] == 112681 and stage["resume_allowed"] is False
            and stage["checkpoint_all_state_bits_equal"] is True and stage["nearest_all_bytes_equal"] is True,
            "dedicated complete export stage")
    context = stage["context"]
    require(exact(context["recipe"], refs["recipe"]) and exact(context["manifest"], refs["manifest"]), "independent context pins")
    recipe = unique(read_ref(refs["recipe"]))
    require(recipe["mode"] == MODE and recipe["objective"] == "absolute-cp-mse"
            and recipe["feature_schema"] == "flat_white_view_aux_tied_v1", "own numeric recipe")
    require(refs["profile"]["sha256"] == PROFILE_SHA and refs["manifest"]["sha256"] == MANIFEST_SHA,
            "own profile/manifest pins")
    profile, manifest = unique(read_ref(refs["profile"])), unique(read_ref(refs["manifest"]))
    require(profile["kind"] == "whole-pack-hash-ranked-frozen-holdout-profile-v2", "profile kind")
    games = {}
    for game in manifest["games"]:
        require(game["game_id"] not in games and game["split"] in COUNTS, "unique game split")
        games[game["game_id"]] = game["split"]
    samples = []
    for split in ("train", "holdout"):
        name = split + ".positions.jsonl"
        require(exact(manifest["files"][name], {k:refs[name][k] for k in ("bytes","sha256")}),
                "manifest full split metadata")
        samples += sample_rows(read_ref(refs[name]), COUNTS[split], games, split)
    for r in refs.values():
        require(not output.is_relative_to(fullref(r)) and not fullref(r).is_relative_to(output), "output overlaps input")
    for protected in plan["protected_directories"]:
        path = Path(protected)
        require(path.is_absolute() and path.resolve(strict=True) == path and path.is_dir(), "canonical protected directory")
        require(not output.is_relative_to(path) and not path.is_relative_to(output), "output overlaps preserved directory")
    request = {"schema":"sekirei.raw-master-quantization-request.v1", "mode":MODE,
               "phase":"completed-epoch3-step338043", "policy":POLICY, "inputs":refs,
               "context":context, "samples":samples}
    output.mkdir(mode=0o700)
    return write_new(output / "request.json", request)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","aggregate"))
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--expected-sha256",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    raw=read_ref({"path":str(args.input),"bytes":args.input.lstat().st_size,"sha256":args.expected_sha256})
    if args.action=="prepare":
        receipt=prepare(unique(raw),args.output)
    else:
        require(not args.output.exists() and args.output.parent.resolve(strict=True)==args.output.parent,
                "fresh aggregate output")
        rows, snapshots=records_from_log(raw)
        summary=aggregate(rows);summary["float_observations"]=snapshots
        summary["lifecycle_must_be_verified_by_parent"]=True
        receipt=write_new(args.output,summary)
    print(json.dumps(receipt,sort_keys=True))

if __name__ == "__main__":
    main()
