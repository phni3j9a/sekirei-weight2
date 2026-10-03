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


def solve_synthetic(Gz, hz, N, *, max_iterations=MAX_ITERATIONS):
    """One bounded FISTA attempt. Failure never selects another solver/candidate."""
    _require(type(max_iterations) is int and 1 <= max_iterations <= MAX_ITERATIONS,
             "iteration resource cap must be between 1 and 20000")
    problem = prepare_problem(Gz, hz, N)
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
