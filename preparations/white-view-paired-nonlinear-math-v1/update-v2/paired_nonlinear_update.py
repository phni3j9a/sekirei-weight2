"""SOURCE ONLY: small-slot update reference; no model/dataset/file adapter.

This is not an activated recipe. The f32 Adam reference preserves the upstream
algebra/order, but its Python powi/sqrt endpoints are not a bit-exact Rust proof.
The future Rust adapter must reuse pinned adam_update_scalar and certify that
endpoint. Missing sparse gradients mean zero, not a skipped trainable update.
"""
from dataclasses import dataclass
import math
import struct

PROTOTYPE_ONLY = True
PENDING = {"learning_rate": None, "head_init_width": None,
           "output_native_l1_budget": None}
FT_Q_MAX = 797
FT_LIMIT = FT_Q_MAX / 64
FT_BIAS_Q = 64
MAX_PREFIX_FEATURES = 41
PAIR_COUNT = 14
NATIVE_SCALE = 64


def require(condition, message):
    if not condition:
        raise ValueError(message)


def f32(value):
    require(type(value) in (int, float) and math.isfinite(value), "finite numeric required")
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError("f32 overflow") from error
    require(math.isfinite(result), "finite f32 required")
    return result


def bits(value):
    return struct.pack("<f", f32(value))


def signed(value, sign):
    require(type(sign) is int and sign in (-1, 1), "strict mirror sign required")
    word = struct.unpack("<I", bits(value))[0]
    return struct.unpack("<f", struct.pack("<I", word ^ (0x80000000 if sign == -1 else 0)))[0]


@dataclass(frozen=True)
class Cell:
    param: float
    m: float = 0.0
    v: float = 0.0


@dataclass(frozen=True)
class Group:
    # members includes the independent positive master slot exactly once.
    master: str
    members: tuple
    kind: str


def cell_checked(cell):
    require(type(cell) is Cell, "strict Cell required")
    for value in (cell.param, cell.m, cell.v):
        require(type(value) is float and f32(value) == value, "stored f32 cell required")
    require(cell.v >= 0, "negative Adam variance")
    return cell


def mirrored(cell, sign):
    # Negative dependent moments transform as m -> -m, v -> v. They are
    # redundant checkpoint fields, never separately stepped by Adam.
    return Cell(signed(cell.param, sign), signed(cell.m, sign), cell.v)


def validate_partition(cells, groups, protected):
    require(type(cells) is dict and all(type(k) is str for k in cells), "named cells required")
    require(type(groups) is tuple and type(protected) is frozenset, "immutable partition required")
    for cell in cells.values():
        cell_checked(cell)
    used = set(protected)
    require(used <= cells.keys(), "unknown protected slot")
    masters = []
    for group in groups:
        require(type(group) is Group and group.kind in ("ft", "head", "bias", "output"), "strict group kind")
        require(type(group.members) is tuple and group.members, "nonempty members")
        require(group.members[0] == (group.master, 1), "positive master must be first")
        master = cell_checked(cells.get(group.master))
        for member in group.members:
            require(type(member) is tuple and len(member) == 2, "strict member shape")
            slot, sign = member
            require(type(slot) is str and slot in cells and slot not in used, "partition overlap/unknown")
            expected = mirrored(master, sign)
            require(all(bits(getattr(cells[slot], field)) == bits(getattr(expected, field))
                        for field in ("param", "m", "v")), "dependent state mirror mismatch")
            used.add(slot)
        if group.kind == "ft":
            require(abs(master.param) <= FT_LIMIT, "FT master outside safety box")
        masters.append(group.master)
    require(used == cells.keys(), "unclassified cell")
    require(masters == sorted(masters), "canonical master order required")


def aggregate_gradient(group, gradients):
    # Fixed member order; sum first, square only inside ONE master Adam call.
    total = 0.0
    for slot, sign in group.members:
        gradient = f32(gradients.get(slot, 0.0))
        total = f32(total + signed(gradient, sign))
    return total


def _powi_reference(base, exponent):
    # Pure repeated-square f32 reference. Rust powi endpoint parity pending.
    result = 1.0
    while exponent:
        if exponent & 1:
            result = f32(result * base)
        exponent >>= 1
        if exponent:
            base = f32(base * base)
    return result


def adam_reference(cell, gradient, lr, step):
    cell_checked(cell)
    lr = f32(lr)
    require(lr > 0, "positive synthetic LR required")
    require(type(step) is int and 1 <= step < 2**64, "strict positive global step")
    gradient = f32(gradient)
    b1, b2, eps = f32(0.9), f32(0.999), f32(1e-8)
    m = f32(f32(b1 * cell.m) + f32(f32(1 - b1) * gradient))
    v = f32(f32(b2 * cell.v) + f32(f32(f32(1 - b2) * gradient) * gradient))
    exponent = min(step, 2**31 - 1)
    mh = f32(m / f32(1 - _powi_reference(b1, exponent)))
    vh = f32(v / f32(1 - _powi_reference(b2, exponent)))
    denominator = f32(f32(math.sqrt(vh)) + eps)
    delta = f32(f32(-lr * mh) / denominator)
    return Cell(f32(cell.param + delta), m, v)


def update_once(cells, groups, protected, gradients, lr, step):
    """Pure tiny update, not export-ready; output budget intentionally pending.

    Protected params/m/v are copied without Adam. All trainable groups run,
    including groups absent from sparse gradients. FT projection changes only
    params, not moments. No arbitrary head/output clipping is introduced.
    """
    validate_partition(cells, groups, protected)
    require(type(gradients) is dict and gradients.keys() <= cells.keys(), "unknown gradient slot")
    for gradient in gradients.values():
        f32(gradient)
    result = dict(cells)
    updates = []
    for group in groups:
        gradient = aggregate_gradient(group, gradients)
        master = adam_reference(cells[group.master], gradient, lr, step)
        if group.kind == "ft":
            master = Cell(f32(max(-FT_LIMIT, min(FT_LIMIT, master.param))), master.m, master.v)
        for slot, sign in group.members:
            result[slot] = mirrored(master, sign)
        updates.append((group.master, gradient))
    validate_partition(result, groups, protected)
    require(all(result[slot] is cells[slot] for slot in protected), "protected state changed")
    return result, tuple(updates)


def nearest_aux_ft(value):
    value = f32(value)
    require(abs(value) <= FT_LIMIT, "unprojected FT cannot be exported")
    q = round(f32(value * 64))  # same nearest-even endpoint as public exporter
    require(type(q) is int and abs(q) <= FT_Q_MAX, "unsafe saved FT")
    return q


def prefix_bound():
    # Source order undo_capture: -current,+original,+captured,-hand.
    return FT_BIAS_Q - MAX_PREFIX_FEATURES * FT_Q_MAX, FT_BIAS_Q + MAX_PREFIX_FEATURES * FT_Q_MAX


def ft_alias_features(feature):
    """No full-table allocation. Board independent; hand donor0/1 ->3/2."""
    require(type(feature) is int and 0 <= feature < 2420, "strict flat feature index")
    if feature < 2268:
        return (feature,)
    bank, threshold = divmod(feature - 2268, 38)
    donor = bank if bank < 2 else 3-bank
    return (2268 + donor*38 + threshold, 2268 + (3-donor)*38 + threshold)


def head_index_bindings(pair, channel):
    """Native row-major ((view*256+channel)*32+output) slot aliases."""
    require(type(pair) is int and 0 <= pair < PAIR_COUNT, "strict pair index")
    require(type(channel) is int and 2 <= channel < 256, "strict auxiliary channel")
    plus, minus = 4 + 2*pair, 5 + 2*pair
    us, them = channel*32, (256+channel)*32
    return {"u": (us+plus, them+minus), "v": (them+plus, us+minus),
            "bias": (plus,minus), "output": ((plus,1),(minus,-1))}


def paired_head_gradient(a, b, u, v, bias, q, material, teacher):
    """Real-algebra tiny oracle, ONE pair. Native f32 parity is not claimed.

    q is saved native +/-output coefficient. g uses 2q, so this is
    (g(a,b)-g(b,a))/2. Bound and finite checks do not select initialization.
    """
    require(all(type(x) is tuple for x in (a, b, u, v)), "tuple vectors required")
    require(0 < len(a) == len(b) == len(u) == len(v), "paired dimensions")
    for value in (*a, *b, *u, *v, bias, q, material, teacher):
        require(type(value) in (int, float) and math.isfinite(value), "finite real oracle input")
    require(all(0 <= x <= 127 for x in (*a, *b)), "clipped FT input required")
    z = bias + sum(x * y for x, y in zip(a, u)) + sum(x * y for x, y in zip(b, v))
    zs = bias + sum(x * y for x, y in zip(b, u)) + sum(x * y for x, y in zip(a, v))
    require(math.isfinite(z) and math.isfinite(zs), "oracle preactivation overflow")
    h, hs = max(0.0, min(127.0, z)), max(0.0, min(127.0, zs))
    gate, gates = float(0 < z < 127), float(0 < zs < 127)
    residual = q * (h - hs) / NATIVE_SCALE
    error = material + residual - teacher
    factor = 2 * error / NATIVE_SCALE
    require(all(math.isfinite(x) for x in (residual, error, factor)), "oracle loss/residual overflow")
    result = {"residual": residual, "score": material + residual,
            "dq": factor * (h - hs),
            "du": tuple(factor * q * (gate*x - gates*y) for x,y in zip(a,b)),
            "dv": tuple(factor * q * (gate*y - gates*x) for x,y in zip(a,b)),
            "db": factor * q * (gate - gates),
            "da": tuple(factor * q * (gate*x - gates*y) for x,y in zip(u,v)),
            "db_input": tuple(factor * q * (gate*y - gates*x) for x,y in zip(u,v))}
    for value in result.values():
        require(all(math.isfinite(x) for x in value) if type(value) is tuple
                else math.isfinite(value), "oracle score/gradient overflow")
    return result


def runtime_guard():
    # Before path parsing, I/O, imports of numerical/data adapters or processes.
    raise RuntimeError("PROTOTYPE_ONLY: recipe pending; no train/export/runtime adapter")


if __name__ == "__main__":
    runtime_guard()
