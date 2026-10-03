"""Pure synthetic constrained-ridge prototype; no dataset or file APIs."""
from dataclasses import dataclass
from fractions import Fraction
import math
import os
import struct

PROTOTYPE_ONLY = True
RHO = Fraction(161791, 4096)  # 39.5 - 2**-12
SAVED_RADIUS = Fraction(79, 2)
GAP_PER_SAMPLE = Fraction(1, 1_000_000)
MAX_ITERATIONS = 20_000
MAX_DIMENSION = 254
MAX_SAMPLES = 1_000_000
EXACT_FLOAT_INTEGER_LIMIT = 2**52


class PrototypeOnly(RuntimeError):
    pass


class InvalidProblem(ValueError):
    pass


class ConvergenceFailure(RuntimeError):
    pass


def runtime_entry(*args, **kwargs):
    if PROTOTYPE_ONLY:
        raise PrototypeOnly("prototype only; actual entry is blocked before I/O")
    raise PrototypeOnly("this pure prototype has no actual input or output driver")


def _require(condition, message):
    if not condition:
        raise InvalidProblem(message)


def _sequence(value, name):
    _require(type(value) in (list, tuple), name + " must be a list or tuple")
    return value


def _integer(value, name):
    _require(type(value) is int and abs(value) <= EXACT_FLOAT_INTEGER_LIMIT - 256,
             name + " must be an exact bounded Python integer, not bool")
    return value


def _positive_radius(rho):
    _require(type(rho) is Fraction and rho > 0, "radius must be a positive Fraction")
    return rho


def _dyadic_vector(vector, dimension=None):
    values = tuple(_sequence(vector, "coefficient vector"))
    _require(0 < len(values) <= MAX_DIMENSION, "coefficient dimension out of bounds")
    if dimension is not None:
        _require(len(values) == dimension, "coefficient dimension mismatch")
    _require(all(type(v) is float and math.isfinite(v) for v in values),
             "coefficients must be finite Python floats, not bool")
    ratios = [v.as_integer_ratio() for v in values]
    denominator = max(den for _, den in ratios)
    numerators = tuple(num * (denominator // den) for num, den in ratios)
    return values, numerators, denominator


def _norm(vector):
    _, numerators, denominator = _dyadic_vector(vector)
    return Fraction(sum(abs(v) for v in numerators), denominator)


def _require_psd(gram):
    """Exact LDL/Schur PSD test, including singular zero pivots."""
    dimension = len(gram)
    schur = [[Fraction(v) for v in row] for row in gram]
    for k in range(dimension):
        pivot = schur[k][k]
        _require(pivot >= 0, "Gz is not positive semidefinite")
        if pivot == 0:
            _require(all(schur[i][k] == 0 for i in range(k + 1, dimension)),
                     "Gz has a nonzero column at a zero PSD pivot")
            continue
        for i in range(k + 1, dimension):
            for j in range(i, dimension):
                value = schur[i][j] - schur[i][k] * schur[j][k] / pivot
                schur[i][j] = value
                schur[j][i] = value


@dataclass(frozen=True)
class _Problem:
    gram: tuple
    K: tuple
    hz: tuple
    sample_count: int
    dimension: int
    lipschitz_exact: Fraction
    lipschitz_float: float


def prepare_problem(Gz, hz, N):
    _require(type(N) is int and 1 <= N <= MAX_SAMPLES, "N must be a bounded positive integer")
    rows = _sequence(Gz, "Gz")
    dimension = len(rows)
    _require(1 <= dimension <= MAX_DIMENSION, "matrix dimension out of bounds")
    gram = []
    for row in rows:
        row = _sequence(row, "Gz row")
        _require(len(row) == dimension, "Gz must be square")
        gram.append(tuple(_integer(v, "Gz entry") for v in row))
    _require(all(gram[i][j] == gram[j][i] for i in range(dimension) for j in range(i)),
             "Gz must be exactly symmetric")
    target = _sequence(hz, "hz")
    _require(len(target) == dimension, "hz dimension mismatch")
    target = tuple(_integer(v, "hz entry") for v in target)
    _require_psd(gram)
    K = tuple(tuple(gram[i][j] + (256 if i == j else 0) for j in range(dimension))
              for i in range(dimension))
    lipschitz = Fraction(max(sum(abs(v) for v in row) for row in K), 256)
    # Rounding upward makes this a certified bound for the exact H entries.
    numeric_lipschitz = math.nextafter(float(lipschitz), math.inf)
    _require(math.isfinite(numeric_lipschitz) and numeric_lipschitz > 0,
             "nonfinite Lipschitz bound")
    return _Problem(tuple(gram), K, target, N, dimension, lipschitz, numeric_lipschitz)


def _certificate(problem, vector, rho):
    rho = _positive_radius(rho)
    values, U, Q = _dyadic_vector(vector, problem.dimension)
    norm = Fraction(sum(abs(v) for v in U), Q)
    # g_i = gradient_numerator_i / (256*Q), with exact integer sums.
    KU = tuple(sum(k * u for k, u in zip(row, U)) for row in problem.K)
    numerator = tuple(v - 16 * Q * h for v, h in zip(KU, problem.hz))
    gradient_dot = Fraction(sum(g * u for g, u in zip(numerator, U)), 256 * Q * Q)
    maximum_gradient = Fraction(max(abs(g) for g in numerator), 256 * Q)
    gap = gradient_dot + rho * maximum_gradient
    objective_delta = (Fraction(sum(u * v for u, v in zip(U, KU)), 512 * Q * Q)
                       - Fraction(sum(h * u for h, u in zip(problem.hz, U)), 16 * Q))
    feasible = norm <= rho
    _require(not feasible or gap >= 0, "negative exact FW gap at feasible coefficients")
    per_sample = gap / problem.sample_count
    return {"norm": norm, "radius": rho, "feasible": feasible,
            "fw_gap": gap, "fw_gap_per_sample": per_sample,
            "solver_threshold_met": feasible and 0 <= per_sample <= GAP_PER_SAMPLE,
            "objective_delta_vs_zero": objective_delta,
            "gradient_numerators": numerator, "gradient_denominator": 256 * Q,
            "coefficient_values": values}


def exact_certificate(Gz, hz, N, coefficients, rho=RHO):
    """Exact dyadic certificate for the half objective J, including ridge."""
    return _certificate(prepare_problem(Gz, hz, N), coefficients, rho)


def project_l1(vector, rho=RHO):
    """Stable-sort numerical Euclidean projection; final feasibility is separate."""
    rho = _positive_radius(rho)
    values, _, _ = _dyadic_vector(vector)
    if _norm(values) <= rho:
        return tuple(0.0 if v == 0.0 else v for v in values)
    ordered = sorted(enumerate(map(abs, values)), key=lambda item: (-item[1], item[0]))
    prefix = []
    active = 0
    threshold = 0.0
    for k, (_, value) in enumerate(ordered, 1):
        prefix.append(value)
        candidate = (math.fsum(prefix) - float(rho)) / k
        if value > candidate:
            active = k
            threshold = candidate
    _require(active > 0, "projection has no active coordinate")
    projected = tuple(math.copysign(max(abs(v) - threshold, 0.0), v) if abs(v) > threshold else 0.0
                      for v in values)
    _require(_norm(projected) <= rho * (1 + Fraction(1, 2**40)),
             "numerical L1 projection exceeds the final one-correction allowance")
    return projected


def _numpy():
    # Lazy import: the actual-entry barrier runs without importing any numeric backend.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    import numpy
    return numpy


def _saved_f32(vector):
    values = []
    for value in vector:
        try:
            converted = struct.unpack("<f", struct.pack("<f", value))[0]
        except (OverflowError, struct.error) as error:
            raise InvalidProblem("f32 coefficient conversion failed") from error
        _require(math.isfinite(converted), "nonfinite saved f32 coefficient")
        values.append(0.0 if converted == 0.0 else converted)
    return tuple(values)


def finalize_f64(problem, vector):
    """Only one factor, only for an exact rho violation; always recertify."""
    values, _, _ = _dyadic_vector(vector, problem.dimension)
    correction_used = _norm(values) > RHO
    if correction_used:
        values = tuple(float(v * (1.0 - 2.0**-40)) for v in values)
    certificate = _certificate(problem, values, RHO)
    _require(certificate["feasible"], "one permitted final f64 correction did not make coefficients feasible")
    return values, certificate, correction_used


# New dedicated design origin entry. Generic prepare/PSD/certificate stay exact.
import hashlib as _hashlib
from pathlib import Path as _Path
import time as _time
import verified_design_gram as _design_gram

_problem_origins = {}


class DeadlineExceeded(ConvergenceFailure):
    pass


class _Deadline:
    def __init__(self, seconds=1200.0):
        _require(type(seconds) is float and math.isfinite(seconds) and 0 < seconds <= 1200.0,
                 'deadline must be a finite float in (0,1200]')
        self.started = _time.monotonic()
        self.seconds = seconds

    def check(self):
        elapsed = _time.monotonic() - self.started
        _require(math.isfinite(elapsed) and elapsed >= 0, 'invalid monotonic elapsed time')
        if elapsed >= self.seconds:
            raise DeadlineExceeded('fixed wall-clock resource deadline reached; no fallback')


def _check_deadline(deadline):
    if deadline is not None:
        _require(type(deadline) is _Deadline, 'internally issued deadline required')
        deadline.check()


def _full_solver_sha256():
    return _hashlib.sha256(_Path(__file__).resolve(strict=True).read_bytes()).hexdigest()


def _problem_digest(problem):
    payload = (problem.gram,problem.K,problem.hz,problem.sample_count,problem.dimension,
               problem.lipschitz_exact.numerator,problem.lipschitz_exact.denominator,
               problem.lipschitz_float.hex())
    return _hashlib.sha256(repr(payload).encode('ascii')).hexdigest()


def _issue_problem(problem, origin):
    _require(type(problem) is _Problem, 'exact internally constructed problem required')
    _problem_origins[id(problem)] = (problem,_problem_digest(problem),origin)
    return problem


def _validate_problem_origin(problem):
    _require(type(problem) is _Problem, 'exact internally constructed problem required')
    record = _problem_origins.get(id(problem))
    _require(record is not None and record[0] is problem and record[1] == _problem_digest(problem),
             'unissued or changed Problem; no free prepared-problem entry')


def solve_synthetic(Gz,hz,N,*,max_iterations=MAX_ITERATIONS):
    # Generic entry still executes the unmodified full exact PSD guard.
    problem = _issue_problem(prepare_problem(Gz,hz,N),'generic-exact-PSD')
    return _solve_prepared(problem,N,max_iterations=max_iterations)


def _solve_design(design,*,expected_producer_source_sha256,expected_solver_source_sha256,
                  max_iterations,deadline):
    before_solver = _full_solver_sha256()
    _require(before_solver == _design_gram._sha(expected_solver_source_sha256),
             'full solver source differs from external binding')
    artifact = _design_gram.produce_gram(design,expected_producer_source_sha256=expected_producer_source_sha256,
                                         deadline_check=deadline.check)
    certified = _design_gram.consume_gram(artifact,expected_design_sha256=design.design_sha256,
        expected_gram_sha256=artifact.gram_sha256,expected_producer_source_sha256=expected_producer_source_sha256)
    problem = _Problem(certified.gram,certified.K,certified.hz,certified.sample_count,
                       certified.dimension,certified.lipschitz_exact,certified.lipschitz_float)
    problem = _issue_problem(problem,('verified-design-algebraic-PSD',design.design_sha256,artifact.gram_sha256))
    deadline.check()
    result = _solve_prepared(problem,design.samples,max_iterations=max_iterations,_deadline=deadline)
    # Reconfirm immutable design/Gram and both full source files after the fit.
    _design_gram.consume_gram(artifact,expected_design_sha256=design.design_sha256,
        expected_gram_sha256=artifact.gram_sha256,expected_producer_source_sha256=expected_producer_source_sha256)
    _require(_full_solver_sha256() == before_solver,'full solver source changed during synthetic solve')
    deadline.check()
    result['verified_design_gram'] = {'schema':_design_gram.SCHEMA,'producer':artifact.producer,
        'design_sha256':design.design_sha256,'gram_sha256':artifact.gram_sha256,
        'producer_source_sha256':artifact.producer_source_sha256,'solver_source_sha256':before_solver,
        'backend_metadata':artifact.backend_metadata,'sample_count':design.samples,'dimension':design.dimension,
        'algebraic_psd_proven_from_issued_design':True,'runtime_threadpool_proof_claimed':False,
        'actual_dataset_provenance_claimed':False,'source_prototype_only':True}
    return result


def solve_verified_design(Z,d,*,expected_producer_source_sha256,expected_solver_source_sha256,
                          max_iterations=MAX_ITERATIONS,deadline_seconds=1200.0):
    deadline = _Deadline(deadline_seconds)
    deadline.check()
    design = _design_gram.verified_design(Z,d)
    deadline.check()
    return _solve_design(design,expected_producer_source_sha256=expected_producer_source_sha256,
        expected_solver_source_sha256=expected_solver_source_sha256,
        max_iterations=max_iterations,deadline=deadline)


def solve_verified_design_bytes(z_bytes,d_bytes,N,dimension,*,expected_producer_source_sha256,
                                expected_solver_source_sha256,max_iterations=MAX_ITERATIONS,
                                deadline_seconds=1200.0):
    deadline = _Deadline(deadline_seconds)
    deadline.check()
    design = _design_gram.verified_design_bytes(z_bytes,d_bytes,N,dimension)
    deadline.check()
    return _solve_design(design,expected_producer_source_sha256=expected_producer_source_sha256,
        expected_solver_source_sha256=expected_solver_source_sha256,
        max_iterations=max_iterations,deadline=deadline)


def _solve_prepared(problem, N, *, max_iterations=MAX_ITERATIONS, _deadline=None):
    """One bounded FISTA attempt. Failure never selects another solver/candidate."""
    _require(type(max_iterations) is int and 1 <= max_iterations <= MAX_ITERATIONS,
             "iteration resource cap must be between 1 and 20000")
    _validate_problem_origin(problem)
    _require(type(N) is int and N == problem.sample_count, 'internal sample count differs')
    _check_deadline(_deadline)
    np = _numpy()
    H = np.asarray(problem.K, dtype=np.float64) / 256.0
    b = np.asarray(problem.hz, dtype=np.float64) / 16.0
    L = problem.lipschitz_float
    x = np.zeros(problem.dimension, dtype=np.float64)
    y = x.copy()
    momentum = 1.0
    restart_count = 0
    last_certificate = None

    def gradient(v):
        result = H @ v - b
        _require(bool(np.isfinite(result).all()), "nonfinite numeric gradient")
        return result

    def objective(v):
        result = float(0.5 * (v @ (H @ v)) - b @ v)
        _require(math.isfinite(result), "nonfinite numeric objective")
        return result

    def projected_step(v):
        proposal = v - gradient(v) / L
        _require(bool(np.isfinite(proposal).all()), "nonfinite projected step")
        return np.asarray(project_l1(tuple(float(v) for v in proposal)), dtype=np.float64)

    for iteration in range(1, max_iterations + 1):
        _check_deadline(_deadline)
        candidate = projected_step(y)
        restarted = objective(candidate) > objective(x)
        if restarted:
            restart_count += 1
            candidate = projected_step(x)
            next_momentum = 1.0
            next_y = candidate.copy()
        else:
            next_momentum = (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum)) / 2.0
            next_y = candidate + ((momentum - 1.0) / next_momentum) * (candidate - x)
        x = candidate
        y = next_y
        momentum = next_momentum
        g = gradient(x)
        gap_proxy = float(g @ x + float(RHO) * np.max(np.abs(g))) / N
        _require(math.isfinite(gap_proxy), "nonfinite approximate FW gap")
        if gap_proxy <= 0.9e-6 or iteration == max_iterations:
            _check_deadline(_deadline)
            last_certificate = _certificate(problem, tuple(float(v) for v in x), RHO)
            correction_used = False
            if not last_certificate["feasible"]:
                final, last_certificate, correction_used = finalize_f64(problem, tuple(float(v) for v in x))
                if not last_certificate["solver_threshold_met"]:
                    raise ConvergenceFailure("one final feasibility correction failed the exact FW-gap certificate")
                x = np.asarray(final, dtype=np.float64)
            if last_certificate["solver_threshold_met"]:
                saved = _saved_f32(last_certificate["coefficient_values"])
                saved_certificate = _certificate(problem, saved, SAVED_RADIUS)
                _require(saved_certificate["feasible"], "saved f32 L1 exceeds 39.5")
                _require(saved_certificate["objective_delta_vs_zero"] <= 0,
                         "saved f32 objective is worse than zero/material")
                _check_deadline(_deadline)
                _validate_problem_origin(problem)
                return {"status": "synthetic-certified", "iterations": iteration,
                        "restart_count": restart_count, "coefficient_f64": tuple(float(v) for v in x),
                        "coefficient_f32": saved, "certificate_f64": last_certificate,
                        # This reported gap uses the SOLVER rho; no saved-gap threshold is imposed.
                        "certificate_f32_solver_radius": _certificate(problem, saved, RHO),
                        "certificate_f32_saved_radius": saved_certificate,
                        "saved_f32_gap_is_solver_stopping_criterion": False,
                        "lipschitz_exact": problem.lipschitz_exact,
                        "lipschitz_numeric_upper": L, "fallback_used": False,
                        "one_if_needed_f64_correction_used": correction_used,
                        "one_if_needed_f64_correction_factor": 1.0 - 2.0**-40,
                        "lambda": 1, "loss_is_half_unnormalized_sum": True}
    raise ConvergenceFailure("iteration cap reached without exact feasible FW-gap certificate")


if __name__ == "__main__":
    runtime_entry()
