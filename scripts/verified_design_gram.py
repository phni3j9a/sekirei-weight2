"""Source-only algebraic Gram origin demonstration, prepared full NumPy producer; synthetic tests only.
No dataset/model/file driver, engine, runtime entry or deserialization.
The only source-file read is hashing this module before/after production.
The fixed producer invocation proves origin; SHA only binds content. Python
reflection or arbitrary code execution is outside this source-pinned contract.
"""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
import os
from pathlib import Path
import sys
import re
import struct

PROTOTYPE_ONLY = True
MAX_DIMENSION = 254
MAX_SAMPLES = 1_000_000
EXACT_FLOAT_INTEGER_LIMIT = 2**52
MAX_Z = 40
MAX_D = 55779
SCHEMA = 'sekirei.verified-design-algebraic-gram.source-prototype.v1'
PRODUCER = 'numpy-float64-exact-integer-gram-v1'


class InvalidOrigin(ValueError):
    pass


def runtime_entry(*args, **kwargs):
    raise RuntimeError('source prototype only; no actual fit or I/O entry')


def require(value, message):
    if not value:
        raise InvalidOrigin(message)


def _sha(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'strict SHA256 required')
    return value


def _hash(parts):
    h = hashlib.sha256()
    for part in parts:
        h.update(part)
    return h.hexdigest()


@dataclass(frozen=True)
class _VerifiedDesign:
    samples: int
    dimension: int
    z_bytes: bytes
    d_bytes: bytes
    design_sha256: str


@dataclass(frozen=True)
class _ProducedGram:
    design: _VerifiedDesign
    gram: tuple
    hz: tuple
    producer: str
    producer_source_sha256: str
    backend_metadata: tuple
    gram_sha256: str


@dataclass(frozen=True)
class AlgebraicProblem:
    gram: tuple
    K: tuple
    hz: tuple
    sample_count: int
    dimension: int
    lipschitz_exact: Fraction
    lipschitz_float: float


# Strong references prevent id reuse. Issued object identity and independent
# canonical digests reject dataclasses.replace/copy and altered frozen fields.
_design_origins = {}
_gram_origins = {}


def _design_hash(samples, dimension, z_bytes, d_bytes):
    return _hash([SCHEMA.encode(), b'\x00design\x00', struct.pack('<QQ', samples, dimension), z_bytes, d_bytes])


def verified_design_bytes(z_bytes, d_bytes, samples, dimension):
    """Consumer interface for a future SFEN/FT producer; canonical bytes only.

    Future driver must bind actual original TRAIN join and native FT/source SHA.
    This pure layer verifies numerical design content, not shogi provenance.
    """
    require(type(samples) is int and 1 <= samples <= MAX_SAMPLES
            and type(dimension) is int and 1 <= dimension <= MAX_DIMENSION,
            'strict count/dimension required, never bool')
    require(type(z_bytes) is bytes and len(z_bytes) == samples * dimension
            and type(d_bytes) is bytes and len(d_bytes) == 4 * samples,
            'canonical immutable int8/int32-LE design bytes required')
    allowed = {value & 255 for value in range(-MAX_Z, MAX_Z + 1)}
    require(set(z_bytes).issubset(allowed), 'Z byte outside signed integer bound')
    require(all(abs(value[0]) <= MAX_D for value in struct.iter_unpack('<i', d_bytes)),
            'd byte outside signed integer bound')
    require(samples * MAX_Z**2 < EXACT_FLOAT_INTEGER_LIMIT - 256
            and samples * MAX_Z * MAX_D < EXACT_FLOAT_INTEGER_LIMIT - 256,
            'integer floating accumulation bounds exceeded')
    digest = _design_hash(samples, dimension, z_bytes, d_bytes)
    result = _VerifiedDesign(samples, dimension, z_bytes, d_bytes, digest)
    _design_origins[id(result)] = (result, digest)
    return result


def verified_design(Z, d):
    require(type(Z) in (list, tuple) and type(d) in (list, tuple), 'sequences required')
    samples = len(Z)
    require(1 <= samples <= MAX_SAMPLES and len(d) == samples, 'sample count mismatch')
    require(type(Z[0]) in (list, tuple), 'row sequence required')
    dimension = len(Z[0])
    require(1 <= dimension <= MAX_DIMENSION, 'dimension out of range')
    flat = []
    for row in Z:
        require(type(row) in (list, tuple) and len(row) == dimension, 'rectangular strict rows required')
        for value in row:
            require(type(value) is int and abs(value) <= MAX_Z, 'Z strict bounded int, never bool/float')
            flat.append(value & 255)
    for value in d:
        require(type(value) is int and abs(value) <= MAX_D, 'd strict bounded int, never bool/float')
    return verified_design_bytes(bytes(flat), b''.join(struct.pack('<i', value) for value in d), samples, dimension)


def _verify_design(design):
    require(type(design) is _VerifiedDesign, 'exact issued design type required')
    record = _design_origins.get(id(design))
    require(record is not None and record[0] is design, 'design was not issued by verified_design')
    require(type(design.samples) is int and type(design.dimension) is int
            and 1 <= design.samples <= MAX_SAMPLES and 1 <= design.dimension <= MAX_DIMENSION,
            'strict design count/shape required')
    require(type(design.z_bytes) is bytes and len(design.z_bytes) == design.samples * design.dimension
            and type(design.d_bytes) is bytes and len(design.d_bytes) == 4 * design.samples,
            'immutable canonical design bytes required')
    digest = _design_hash(design.samples, design.dimension, design.z_bytes, design.d_bytes)
    require(digest == record[1] == design.design_sha256, 'design content/origin digest changed')
    return design


def _gram_hash(design, gram, hz, source_sha256, backend_metadata):
    return _hash([SCHEMA.encode(), b'\x00gram\x00', PRODUCER.encode(), b'\x00', bytes.fromhex(source_sha256),
                  repr(backend_metadata).encode('ascii'), b'\x00',
                  bytes.fromhex(design.design_sha256), struct.pack('<QQ', design.samples, design.dimension),
                  b''.join(struct.pack('<q', value) for row in gram for value in row),
                  b''.join(struct.pack('<q', value) for value in hz)])


_backend_first_import = None


def _source_sha256():
    source = Path(__file__).resolve(strict=True)
    require(source.is_file() and source.suffix == '.py', 'actual full producer source required')
    return hashlib.sha256(source.read_bytes()).hexdigest()


def _numpy():
    global _backend_first_import
    if _backend_first_import is None:
        _backend_first_import = 'numpy' in sys.modules
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    import numpy as np
    require(np.__version__ == '1.26.4', 'fixed NumPy1.26.4 required')
    return np


def verified_numpy_design(Z, d):
    """Strict int8/int32 C-array bridge, deep immutable bytes copy.

    Existing NumPy import needs actual thread-pool setting evidence in the
    future runtime driver; setting environment now alone does not prove that.
    """
    np = _numpy()
    require(type(Z) is np.ndarray and Z.ndim == 2 and Z.dtype == np.dtype('int8')
            and Z.flags.c_contiguous, 'exact C-order int8 ndarray required')
    require(type(d) is np.ndarray and d.ndim == 1 and d.dtype == np.dtype('<i4')
            and d.flags.c_contiguous, 'exact C-order int32-LE ndarray target required')
    n, p = Z.shape
    require(d.shape == (n,), 'NumPy design target count differs')
    return verified_design_bytes(Z.tobytes(order='C'), d.tobytes(order='C'), int(n), int(p))


def produce_gram(design, *, expected_producer_source_sha256, deadline_check):
    """Sole producer: validated immutable design, no caller-supplied Gram/hz."""
    expected_source = _sha(expected_producer_source_sha256)
    require(callable(deadline_check), 'internal deadline check required')
    deadline_check()
    design = _verify_design(design)
    before_source = _source_sha256()
    require(before_source == expected_source, 'producer full source differs from external binding')
    np = _numpy()
    deadline_check()
    n, p = design.samples, design.dimension
    # Bound every product and every partial sum, independent of BLAS order/FMA.
    gram_bound, hz_bound = n*MAX_Z**2, n*MAX_Z*MAX_D
    require(gram_bound < 2**53 and hz_bound < 2**53, 'binary64 integer partial-sum proof failed')
    z8 = np.frombuffer(design.z_bytes, dtype=np.int8).reshape((n,p), order='C')
    d32 = np.frombuffer(design.d_bytes, dtype='<i4')
    require(z8.shape == (n,p) and z8.flags.c_contiguous and not z8.flags.writeable
            and d32.shape == (n,) and d32.flags.c_contiguous and not d32.flags.writeable,
            'immutable design view/order/shape differs')
    Z = z8.astype(np.float64, order='C', copy=True)
    d = d32.astype(np.float64, order='C', copy=True)
    Z.flags.writeable = False
    d.flags.writeable = False
    deadline_check()
    require(Z.dtype == d.dtype == np.dtype(np.float64) and Z.shape == (n,p) and d.shape == (n,)
            and Z.flags.c_contiguous and d.flags.c_contiguous
            and np.array_equal(Z,z8) and np.array_equal(d,d32), 'binary64 cast/order changed design')
    # Precisely this transpose/order defines the two outputs.
    G = Z.T @ Z
    deadline_check()
    h = Z.T @ d
    deadline_check()
    require(type(G) is np.ndarray and type(h) is np.ndarray
            and G.dtype == h.dtype == np.dtype(np.float64)
            and G.shape == (p,p) and h.shape == (p,), 'Gram backend shape/dtype differs')
    require(bool(np.isfinite(G).all()) and bool(np.isfinite(h).all())
            and bool((np.abs(G) <= gram_bound).all()) and bool((np.abs(h) <= hz_bound).all())
            and np.array_equal(G,np.rint(G)) and np.array_equal(h,np.rint(h))
            and np.array_equal(G,G.T), 'Gram/rhs not exact finite bounded symmetric integers')
    gram = tuple(tuple(int(value) for value in row) for row in G.tolist())
    hz = tuple(int(value) for value in h.tolist())
    metadata = (('numpy_version',np.__version__), ('dtype','float64'), ('design_order','C'),
                ('transpose','Z.T@Z;Z.T@d'), ('first_numpy_preimported',_backend_first_import),
                ('environment_set_before_first_numpy_import',not _backend_first_import),
                ('existing_import_threadpool_evidence_not_claimed',True))
    digest = _gram_hash(design,gram,hz,before_source,metadata)
    result = _ProducedGram(design,gram,hz,PRODUCER,before_source,metadata,digest)
    _gram_origins[id(result)] = (result,id(design),design.design_sha256,digest,before_source,metadata)
    require(_source_sha256() == before_source and _verify_design(design) is design,
            'producer source/design changed during Gram production')
    deadline_check()
    return result


def consume_gram(artifact, *, expected_design_sha256, expected_gram_sha256, expected_producer_source_sha256):
    expected_design_sha256, expected_gram_sha256 = _sha(expected_design_sha256), _sha(expected_gram_sha256)
    require(type(artifact) is _ProducedGram, 'exact issued Gram type required; JSON/copy not accepted')
    origin = _gram_origins.get(id(artifact))
    require(origin is not None and origin[0] is artifact, 'Gram was not issued by the sole producer')
    design = _verify_design(artifact.design)
    expected_source = _sha(expected_producer_source_sha256)
    require(artifact.producer_source_sha256 == origin[4] == expected_source == _source_sha256()
            and type(artifact.backend_metadata) is tuple and artifact.backend_metadata == origin[5],
            'full producer source/backend binding changed')
    require(origin[1] == id(design) and origin[2] == design.design_sha256 == expected_design_sha256,
            'producer/consumer design binding changed')
    require(type(artifact.producer) is str and artifact.producer == PRODUCER
            and type(artifact.producer_source_sha256) is str
            and type(artifact.gram) is tuple and type(artifact.hz) is tuple,
            'immutable fixed producer output required')
    p, n = design.dimension, design.samples
    require(len(artifact.gram) == p and len(artifact.hz) == p, 'Gram/hz shape mismatch')
    for row in artifact.gram:
        require(type(row) is tuple and len(row) == p, 'immutable square Gram required')
        require(all(type(v) is int and abs(v) <= n*MAX_Z**2 for v in row), 'Gram strict integer/bound required')
    require(all(type(v) is int and abs(v) <= n*MAX_Z*MAX_D for v in artifact.hz), 'hz strict integer/bound required')
    require(all(artifact.gram[i][j] == artifact.gram[j][i] for i in range(p) for j in range(i)), 'Gram symmetry changed')
    digest = _gram_hash(design, artifact.gram, artifact.hz,artifact.producer_source_sha256,artifact.backend_metadata)
    require(digest == origin[3] == artifact.gram_sha256 == expected_gram_sha256,
            'Gram/hz content/producer digest changed')
    # PSD theorem: v^T G v = sum_i (Z_i v)^2 >= 0. The checked origin is
    # essential; a caller-supplied hash or success flag does not prove this.
    K = tuple(tuple(artifact.gram[i][j] + (256 if i == j else 0) for j in range(p)) for i in range(p))
    lipschitz = Fraction(max(sum(abs(v) for v in row) for row in K), 256)
    numeric = math.nextafter(float(lipschitz), math.inf)
    require(math.isfinite(numeric) and numeric > 0, 'invalid exact Gershgorin bound')
    return AlgebraicProblem(artifact.gram, K, artifact.hz, n, p, lipschitz, numeric)


if __name__ == '__main__':
    runtime_entry()
