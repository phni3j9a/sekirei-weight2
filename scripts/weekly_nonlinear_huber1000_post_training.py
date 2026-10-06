"""Shared post-training checks with an explicitly supplied pure validator.

The former parent loaded its validator inside main(), but its physical reader
and epoch parser looked for a module-global ``g``. Keep those checks unchanged
and pass the validator to each consumer before a long-running child starts.
This module does not launch training or turn a checkpoint into model adoption.
"""
from copy import deepcopy
from pathlib import Path


POSITIONS_PER_EPOCH = 112681
EPOCHS = 3


def require(ok, message):
    if not ok:
        raise ValueError(message)


def small(reference):
    return {key: reference[key] for key in ("bytes", "sha256")}


class PhysicalBytes:
    """Read only declared bytes; callers supply their canonical file checker.

    ``physical_ref`` must enforce the parent's canonical regular-file policy,
    and return a complete path/bytes/SHA-256 reference. The pure gate is an
    instance dependency, so a local import in a caller cannot become unbound.
    """

    def __init__(self, expected, *, gate, physical_ref):
        gate.identity_map(expected)
        require(callable(physical_ref), "canonical physical reference checker required")
        self.expected = deepcopy(expected)
        self.gate = gate
        self.physical_ref = physical_ref

    def read(self, reference):
        gate = self.gate
        gate.fullref(reference)
        require(gate.exact(self.expected.get(reference["path"]), small(reference)),
                "unbound physical reference")
        require(gate.exact(self.physical_ref(Path(reference["path"])), reference),
                "physical reference changed")
        raw = Path(reference["path"]).read_bytes()
        require(gate.exact(gate.digest(raw), small(reference)),
                "changed while reading physical reference")
        return raw

    def json(self, reference):
        return self.gate.strict_json(self.read(reference))

    def map(self, value):
        self.gate.identity_map(value)
        for path, identity in value.items():
            self.read({"path": path, **identity})

    def refs_in(self, value, required_map=None):
        if type(value) is dict:
            if set(value) == {"path", "bytes", "sha256"}:
                self.read(value)
                if required_map is not None:
                    require(self.gate.exact(required_map.get(value["path"]), small(value)),
                            "lost raw closure")
            else:
                for item in value.values():
                    self.refs_in(item, required_map)
        elif type(value) is list:
            for item in value:
                self.refs_in(item, required_map)


def epoch_observations(raw, *, gate, recipe):
    """Parse exactly three complete, ordered epochs using the supplied gate."""
    require(type(recipe) is dict, "strict dedicated Huber recipe required before numerical observations")
    for key, expected in {"schema": "sekirei.white-view-paired-nonlinear-huber1000-recipe.v1",
            "mode": "white-view-diverse-games-seed42-huber1000-e3-v1", "objective": "absolute-cp-twice-huber",
            "huber_delta_cp_f32_bits": "447a0000"}.items():
        require(type(recipe.get(key)) is str and recipe[key] == expected, "dedicated Huber observation context differs")
    require(type(raw) is bytes, "raw immutable epoch log bytes required")
    observed = []
    begun = None
    for line in raw.splitlines():
        if line.startswith(b"PAIRED_EPOCH_BEGIN "):
            require(begun is None, "duplicate/unclosed actual epoch begin")
            begun = gate.strict_json(line[len(b"PAIRED_EPOCH_BEGIN "):])
            epoch = len(observed) + 1
            gate.obj(begun, {"epoch", "start_step", "float"})
            gate.fixed(begun, {"epoch": epoch,
                               "start_step": (epoch - 1) * POSITIONS_PER_EPOCH})
            gate.float_snapshot(begun["float"])
        elif line.startswith(b"PAIRED_EPOCH_COMPLETE "):
            require(begun is not None, "actual epoch completed before begin")
            record = gate.strict_json(line[len(b"PAIRED_EPOCH_COMPLETE "):])
            epoch = len(observed) + 1
            gate.obj(record, {"epoch", "positions", "end_step", "float"})
            gate.fixed(record, {"epoch": epoch, "positions": POSITIONS_PER_EPOCH,
                                "end_step": epoch * POSITIONS_PER_EPOCH})
            gate.float_snapshot(record["float"])
            observed.append({"epoch": epoch, "start_step": begun["start_step"],
                             "positions": POSITIONS_PER_EPOCH,
                             "end_step": record["end_step"],
                             "before": begun["float"], "after": record["float"]})
            begun = None
    require(begun is None and len(observed) == EPOCHS,
            "three complete actual epochs required")
    return observed
