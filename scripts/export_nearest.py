#!/usr/bin/env python3
"""Independent, strict post-export of pinned Sekirei FT weights to nearest-even."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess

INPUT, L1, L2 = 2420, 256, 32
MAGIC = b"SEKIRW01"
FT_COUNT = INPUT * L1 + L1
TAIL_OFFSET = 8 + 2 * FT_COUNT
FILE_BYTES = TAIL_OFFSET + 4 * (2 * L1 * L2 + 2 * L2 + 1)
SOURCE_HASHES = {
    "crates/sekirei-core/src/nnue.rs": "a467e8b1b75f6b82629f369a1c804476b1c610354a561822365f980e6e428561",
    "crates/sekirei-core/src/eval.rs": "69fce9b3522a96c68a7f6dce15a342ba34c90f0addd0fa13b7478bef4d25e839",
    "crates/sekirei-train/src/checkpoint.rs": "cbed19c892ccdebd97487db1b4910133e9f3bbd8e75cb14ba733aa2cd09b05be",
    "crates/sekirei-train/src/trainer.rs": "c86b294c35383ef25c65f209735eb1484b58dc15e43777c3f7134b5985724f53",
}
LENGTHS = {"ft": INPUT * L1, "ft_bias": L1, "l2": 2 * L1 * L2,
           "l2_bias": L2, "out": L2, "ft_m": INPUT * L1, "ft_v": INPUT * L1,
           "bias_m": L1, "bias_v": L1, "l2_m": 2 * L1 * L2, "l2_v": 2 * L1 * L2,
           "l2bias_m": L2, "l2bias_v": L2, "out_m": L2, "out_v": L2}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def fnv1a(data):
    h = 14695981039346656037
    for b in data:
        h = ((h ^ b) * 1099511628211) & ((1 << 64) - 1)
    return f"{h:016x}"


def f32(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("checkpoint values must be finite numbers")
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError("checkpoint value exceeds finite f32") from error
    if not math.isfinite(result):
        raise ValueError("checkpoint value exceeds finite f32")
    return result


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(data):
    def nonfinite(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    return json.loads(data, parse_constant=nonfinite, object_pairs_hook=strict_object)


def validate_adam(adam):
    if (not isinstance(adam, dict) or adam.get("schema") != "sekirei.adam-checkpoint.v1"
            or type(adam.get("version")) is not int or adam["version"] != 1):
        raise ValueError("unsupported Adam schema/version")
    if type(adam.get("step")) is not int or not 0 <= adam["step"] < 1 << 64:
        raise ValueError("invalid Adam step")
    for name, count in LENGTHS.items():
        values = adam.get(name)
        if not isinstance(values, list) or len(values) != count:
            raise ValueError(f"invalid {name} length; expected {count}")
        # Keep restored f32s, never do quantization on the JSON decimal-as-f64.
        adam[name] = [f32(value) for value in values]
    for name in ("out_bias", "obias_m", "obias_v"):
        adam[name] = f32(adam.get(name))
    return adam


def quantize(value, nearest):
    # Power-of-two scaling is exact in the relevant range. Clamp before any
    # f32 repack: very large finite raw f32s can overflow native *64 to inf,
    # whose native clamped result is still the appropriate endpoint.
    scaled = max(-32767.0, min(32767.0, value * 64.0))
    return round(scaled) if nearest else int(scaled)


def float_tail(adam):
    values = adam["l2"] + adam["l2_bias"] + adam["out"] + [adam["out_bias"]]
    return struct.pack(f"<{len(values)}f", *values)


def post_export(adam, native):
    if len(native) != FILE_BYTES or native[:8] != MAGIC:
        raise ValueError("native file is not pinned INPUT2420/L1=256/L2=32 SEKIRW01")
    ft = adam["ft"] + adam["ft_bias"]
    trunc = [quantize(value, False) for value in ft]
    legacy = MAGIC + struct.pack(f"<{FT_COUNT}h", *trunc) + float_tail(adam)
    if legacy != native:
        raise ValueError("raw Adam native trunc export differs from supplied .bin; refusing")
    nearest = [quantize(value, True) for value in ft]
    output = MAGIC + struct.pack(f"<{FT_COUNT}h", *nearest) + native[TAIL_OFFSET:]
    assert output[:8] == native[:8] and output[TAIL_OFFSET:] == native[TAIL_OFFSET:]
    stats = {}
    for name, lo, hi in (("ft", 0, INPUT * L1), ("ft_bias", INPUT * L1, FT_COUNT)):
        values, old, new = ft[lo:hi], trunc[lo:hi], nearest[lo:hi]
        stats[name] = {
            "count": len(values), "changed": sum(a != b for a, b in zip(old, new)),
            "half_ties": sum(abs(v * 64.0) < 32767 and abs(v * 64.0) % 1 == 0.5 for v in values),
            "clipped_raw_values": sum(abs(v * 64.0) > 32767 for v in values),
            "max_native_to_nearest_integer_change": max(abs(a - b) for a, b in zip(old, new)),
            "native_mean_abs_weight_error": math.fsum(abs(v - q / 64) for v, q in zip(values, old)) / len(values),
            "nearest_mean_abs_weight_error": math.fsum(abs(v - q / 64) for v, q in zip(values, new)) / len(values),
        }
    return output, stats


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main(args):
    source = args.source.resolve(strict=True)
    for name, expected in SOURCE_HASHES.items():
        if sha256((source / name).read_bytes()) != expected:
            raise ValueError(f"pinned source mismatch: {name}")
    paths = {"native": args.native.resolve(strict=True), "adam": args.adam.resolve(strict=True),
             "metadata": args.metadata.resolve(strict=True)}
    snapshots = {name: path.read_bytes() for name, path in paths.items()}
    metadata = load_json(snapshots["metadata"])
    if (metadata.get("nnue_output") != "absolute"
            or not args.teacher_identity.startswith("external:")
            or len(args.teacher_identity) <= 9
            or metadata.get("teacher_identity") != args.teacher_identity
            or metadata.get("checkpoint_hash") != fnv1a(snapshots["native"])):
        raise ValueError("native metadata mode, teacher identity, or checkpoint_hash mismatch")
    adam = validate_adam(load_json(snapshots["adam"]))
    output, stats = post_export(adam, snapshots["native"])
    for name, path in paths.items():
        if path.read_bytes() != snapshots[name]:
            raise ValueError("input changed during post-export")
    destination = args.output.resolve()
    if not destination.parent.is_dir():
        raise ValueError("output parent must already exist")
    git = subprocess.run(["git", "-C", str(destination.parent), "rev-parse", "--is-inside-work-tree"],
                         capture_output=True, text=True)
    if git.returncode == 0 and git.stdout.strip() == "true":
        raise ValueError("derived weights/provenance must stay outside every Git worktree")
    destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    # This is a derived inference artifact, never a replacement Adam checkpoint.
    recipe = {"format": "sekirei-nearest-ft-post-export-v1", "status": "running",
              "architecture": {"input": INPUT, "l1": L1, "l2": L2},
              "quantizer": "raw JSON restored to f32; scale 64; clamp [-32767,32767]; nearest ties-to-even",
              "native_quantizer": "same f32 inputs; scale 64; clamp [-32767,32767]; truncate toward zero",
              "native_reexport_all_bytes_equal": True, "non_ft_bytes_preserved": True,
              "non_ft_tail_offset": TAIL_OFFSET, "non_ft_tail_sha256": sha256(output[TAIL_OFFSET:]),
              "teacher_identity": args.teacher_identity, "adam_step": adam["step"],
              "inputs": {name: {"path": str(paths[name]), "sha256": sha256(data), "bytes": len(data)}
                         for name, data in snapshots.items()},
              "pinned_source_sha256": SOURCE_HASHES, "script_sha256": sha256(Path(__file__).read_bytes()),
              "ft_statistics": stats, "engine_verified": False}
    try:
        binary = destination / "nearest.bin"
        with binary.open("xb") as stream:
            stream.write(output)
            stream.flush()
            os.fsync(stream.fileno())
        if binary.read_bytes() != output:
            raise ValueError("output verification failed")
        sidecar = {"format": "sekirei-nnue-output-v1", "nnue_output": "absolute",
                   "checkpoint_hash": fnv1a(output), "baseline": None}
        write_json(destination / "nearest.meta.json", sidecar)
        for name, path in paths.items():
            if path.read_bytes() != snapshots[name]:
                raise ValueError("input changed while writing post-export outputs")
        recipe.update(status="complete", output_sha256=sha256(output), output_bytes=len(output),
                      checkpoint_hash=sidecar["checkpoint_hash"])
    except BaseException as error:
        recipe.update(status="failed", error=str(error))
        raise
    finally:
        write_json(destination / "recipe.json", recipe)
    print(json.dumps({"status": "complete", "output": str(destination),
                      "sha256": sha256(output), "ft_statistics": stats}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--adam", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--teacher-identity", required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory under an existing private parent")
    main(parser.parse_args())
