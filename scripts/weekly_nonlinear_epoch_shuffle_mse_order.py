"""Dedicated deterministic row order and actual consumption trace validator.

No data, model, process, filesystem or fitting operations are performed here.
Indices are zero-based original train-row ordinals; holdout is never passed.
"""
import hashlib
import json

SEED = 20261006
TRAIN_COUNT = 112681
EPOCHS = 3
MASK = (1 << 64) - 1
GAMMA = 0x9e3779b97f4a7c15
ALGORITHM = "splitmix64-fisher-yates-rejection-v1"
MIXING = "state=seed-xor-epoch-times-0x9e3779b97f4a7c15-mod2^64;next-adds-gamma"
ENCODING = "original-train-row-index-u64-le"
TRACE = "successful-update-indices-in-contiguous-1024-row-chunks-v1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


class SplitMix64:
    def __init__(self, state):
        require(type(state) is int and 0 <= state <= MASK, "typed u64 state required")
        self.state = state

    def next(self):
        self.state = (self.state + GAMMA) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & MASK
        z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & MASK
        return z ^ (z >> 31)

    def below(self, bound):
        require(type(bound) is int and 0 < bound <= MASK, "positive u64 bound required")
        # Equivalent to u64::wrapping_neg(bound) % bound; reject modulo bias.
        threshold = ((-bound) & MASK) % bound
        while True:
            value = self.next()
            if value >= threshold:
                return value % bound


def permutation(count, seed=SEED, epoch=1):
    require(type(count) is int and 0 < count <= MASK, "positive typed row count required")
    require(type(seed) is int and seed == SEED, "fixed distinct shuffle seed required")
    require(type(epoch) is int and 1 <= epoch <= EPOCHS, "fresh epoch 1..3 required")
    order = list(range(count))
    rng = SplitMix64(seed ^ ((epoch * GAMMA) & MASK))
    for i in range(count - 1, 0, -1):
        j = rng.below(i + 1)
        order[i], order[j] = order[j], order[i]
    return order


def order_identity(order):
    require(type(order) is list and all(type(i) is int and 0 <= i <= MASK for i in order),
            "typed original row indices required")
    raw = b"".join(i.to_bytes(8, "little") for i in order)
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def declaration():
    return {"algorithm": ALGORITHM, "seed_mixing": MIXING,
            "index_encoding": ENCODING, "consumption_trace": TRACE,
            "epochs": [{"epoch": e, "count": TRAIN_COUNT,
                        **order_identity(permutation(TRAIN_COUNT, epoch=e))}
                       for e in range(1, EPOCHS + 1)]}


def strict_json(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate trace JSON key")
            out[key] = value
        return out
    def reject(value):
        raise ValueError("nonfinite trace JSON: " + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)


def exact(actual, expected, message):
    # bool must never be accepted as int or an incomplete control as truthy.
    def same(a, b):
        if type(a) is not type(b): return False
        if type(a) is dict: return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
        if type(a) is list: return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
        return a == b
    require(same(actual, expected), message)


def observations(raw, recipe, *, count=TRAIN_COUNT):
    """Read the bound raw child log, requiring each actually consumed index.

    Recomputing a plan is not sufficient: missing, repeated, altered, unclosed
    or outside-epoch consumption records fail. Each chunk was emitted only
    after its updates succeeded and the Rust tracker checked each exact step.
    The production caller always uses TRAIN_COUNT; count is a fixture parameter.
    """
    require(type(raw) is bytes, "raw child log required")
    exact(recipe.get("schema"), "sekirei.white-view-paired-nonlinear-epoch-shuffle-mse-recipe.v1", "dedicated recipe schema differs")
    exact(recipe.get("mode"), "white-view-diverse-games-shuffle-seed42-e3-v1", "dedicated mode differs")
    exact(recipe.get("objective"), "absolute-cp-mse", "MSE objective differs")
    exact(recipe.get("seed"), 42, "fresh initialization seed differs")
    exact(recipe.get("shuffle_seed"), SEED, "fixed shuffle seed differs")
    exact(recipe.get("row_order"), declaration(), "frozen algorithm/permutation declaration differs")
    begun = None
    active = None
    done = []
    pending = None
    for line in raw.splitlines():
        if line.startswith(b"PAIRED_EPOCH_BEGIN "):
            require(begun is None and active is None and pending is None, "unclosed/duplicate epoch begin")
            value = strict_json(line[len(b"PAIRED_EPOCH_BEGIN "):])
            epoch = len(done) + 1
            require(set(value) == {"epoch", "start_step", "float"}, "epoch begin fields differ")
            exact(value["epoch"], epoch, "actual epoch order differs")
            exact(value["start_step"], (epoch-1)*count, "actual epoch start step differs")
            require(epoch <= EPOCHS, "extra actual epoch")
            begun = epoch
        elif line.startswith(b"PAIRED_SHUFFLE_BEGIN "):
            require(begun is not None and active is None and pending is None, "shuffle begin outside epoch or repeated")
            value = strict_json(line[len(b"PAIRED_SHUFFLE_BEGIN "):])
            expected_order = permutation(count, epoch=begun)
            expected = {"epoch": begun, "seed": SEED, "count": count,
                        "algorithm": ALGORITHM, "permutation": order_identity(expected_order)}
            exact(value, expected, "planned permutation trace differs")
            active = {"order": [], "expected": expected_order, "chunks": 0}
        elif line.startswith(b"PAIRED_SHUFFLE_ROWS "):
            require(active is not None and pending is None, "consumption outside active shuffle epoch")
            value = strict_json(line[len(b"PAIRED_SHUFFLE_ROWS "):])
            require(set(value) == {"epoch", "start_position", "start_step", "end_step", "indices"}, "consumption chunk fields differ")
            start = len(active["order"])
            indices = value["indices"]
            require(type(indices) is list and len(indices) == min(1024, count-start) and len(indices) > 0,
                    "missing/extra/partial consumption chunk")
            exact(value["epoch"], begun, "consumption epoch differs")
            exact(value["start_position"], start, "consumption chunk order/position differs")
            exact(value["start_step"], (begun-1)*count+start, "consumption start update step differs")
            exact(value["end_step"], (begun-1)*count+start+len(indices), "consumption end update step differs")
            require(all(type(i) is int and 0 <= i < count for i in indices), "consumed index outside original row domain")
            exact(indices, active["expected"][start:start+len(indices)], "actual consumed order differs from fixed permutation")
            active["order"].extend(indices)
            active["chunks"] += 1
        elif line.startswith(b"PAIRED_SHUFFLE_COMPLETE "):
            require(active is not None and pending is None, "shuffle completion without active consumption")
            value = strict_json(line[len(b"PAIRED_SHUFFLE_COMPLETE "):])
            order = active["order"]
            require(len(order) == count and len(set(order)) == count and set(order) == set(range(count)), "missing/duplicate actual consumed row")
            expected = {"epoch": begun, "consumed_count": count, "start_step": (begun-1)*count,
                        "end_step": begun*count, "consumed": order_identity(order)}
            exact(value, expected, "actual consumption digest/count/step differs")
            pending = {"epoch": begun, "positions": count, "start_step": (begun-1)*count,
                       "end_step": begun*count, "permutation": order_identity(active["expected"]),
                       "actual_consumed": order_identity(order), "chunks": active["chunks"],
                       "unique_indices": count, "exact_fixed_permutation_consumed": True}
            active = None
        elif line.startswith(b"PAIRED_EPOCH_COMPLETE "):
            require(begun is not None and active is None and pending is not None, "epoch completed without full consumption")
            value = strict_json(line[len(b"PAIRED_EPOCH_COMPLETE "):])
            require(set(value) == {"epoch", "positions", "end_step", "float"}, "epoch completion fields differ")
            exact(value["epoch"], begun, "epoch completion order differs")
            exact(value["positions"], count, "epoch completed count differs")
            exact(value["end_step"], begun*count, "epoch completed updates differ")
            done.append(pending); begun = active = pending = None
        elif line.startswith(b"PAIRED_SHUFFLE_"):
            raise ValueError("unknown shuffle record")
    require(begun is None and active is None and pending is None and len(done) == EPOCHS,
            "three fully consumed actual epochs required")
    return done
