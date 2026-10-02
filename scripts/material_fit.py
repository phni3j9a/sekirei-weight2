#!/usr/bin/env python3
"""Constrained 13-parameter material fit; import is inert.

No search or optimizer-based NNUE training. This module provides a small solver
and a candidate encoder, not an automatic adoption/selection runner. Do not call
encode_candidate until an explicitly controlled private experiment is started.
"""
import hashlib
import math
from pathlib import Path
import struct

PARAMETERS = ('pawn', 'lance', 'knight', 'silver', 'gold', 'bishop', 'rook',
              'pawn_promotion', 'lance_promotion', 'knight_promotion',
              'silver_promotion', 'bishop_promotion', 'rook_promotion')
PRIOR = (100., 430., 470., 640., 680., 890., 1040., 500., 170., 130., 0., 260., 260.)
LAMBDAS = (0.01, 0.1, 1., 10., 100.)
FIT_CAP = 16064.
ENCODED_CAP = 16128
# Each parameter belongs to exactly one of two disjoint weighted simplices.
GROUPS = (((0, 18.), (7, 18.)),
          ((1, 4.), (2, 4.), (3, 4.), (4, 4.), (5, 2.), (6, 2.),
           (8, 4.), (9, 4.), (10, 4.), (11, 2.), (12, 2.)))
BASE = (0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 5, 6)
PROMOTION_PARAMETER = {8: 7, 9: 8, 10: 9, 11: 10, 12: 11, 13: 12}
NATIVE_VALUES = (100, 430, 470, 640, 680, 890, 1040, 0, 600, 600, 600, 640, 1150, 1300)
MATERIAL_SEED42_SHA256 = 'bbe9fbea4c943d69d605190f9ef8c6e9c7a4b9aa7c3a6be970d93e3405334e40'
INPUT, L1, L2 = 2420, 256, 32
BOARD_INPUT, HAND_THRESHOLDS = 2268, 38
HAND_OFFSETS, HAND_MAX = (0, 18, 22, 26, 30, 34, 36), (18, 4, 4, 4, 4, 2, 2)
BINARY_BYTES = 1305356


class FitError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise FitError(message)


def finite_vector(values, count=13):
    values = tuple(values)
    require(len(values) == count, 'wrong vector dimension')
    require(all(type(v) in (int, float) and math.isfinite(v) for v in values), 'nonfinite or nonnumeric vector')
    return tuple(float(v) for v in values)


def dot(left, right):
    require(len(left) == len(right), 'dot-product dimension mismatch')
    return math.fsum(a * b for a, b in zip(left, right))


def group_totals(values):
    values = finite_vector(values)
    return tuple(math.fsum(values[i] * a for i, a in group) for group in GROUPS)


def require_feasible(values, cap=FIT_CAP, tolerance=1e-8):
    values = finite_vector(values)
    require(all(v >= 0 for v in values), 'negative material parameter')
    require(all(total <= cap + tolerance for total in group_totals(values)), 'material group exceeds cap')
    return values


def prices(parameters):
    """Seven base values plus six nonnegative promotion increments -> 14 kinds."""
    parameters = finite_vector(parameters)
    values = [*parameters[:7], 0.]
    values += [parameters[BASE[k]] + parameters[PROMOTION_PARAMETER[k]] for k in range(8, 14)]
    return tuple(values)


def features(position):
    """Use only a position already parsed/inventory-checked by material_init.parse_sfen."""
    require(position['stm'] in (0, 1), 'invalid side to move')
    result = [0] * 13
    for _, kind, color in position['pieces']:
        require(type(kind) is int and 0 <= kind < 14 and color in (0, 1), 'invalid parsed piece')
        if kind == 7:
            continue
        sign = 1 if color == position['stm'] else -1
        result[BASE[kind]] += sign
        if kind in PROMOTION_PARAMETER:
            result[PROMOTION_PARAMETER[kind]] += sign
    require(len(position['hands']) == 2, 'invalid parsed hands')
    for color, hand in enumerate(position['hands']):
        require(len(hand) == 7 and all(type(n) is int and n >= 0 for n in hand), 'invalid parsed hand counts')
        sign = 1 if color == position['stm'] else -1
        for kind, count in enumerate(hand):
            result[kind] += sign * count
    return tuple(result)


def project(values, cap=FIT_CAP):
    """Exact weighted-simplex projection formula, evaluated in Python float64.

    Projection onto z>=0, a.z<=cap has z_i=max(v_i-tau*a_i,0).
    The breakpoints v_i/a_i identify the active set without iterative root search.
    Groups are disjoint, so these two projections are the global Euclidean one.
    """
    values = finite_vector(values)
    require(type(cap) in (float, int) and math.isfinite(cap) and cap > 0, 'invalid projection cap')
    result = [0.] * 13
    for group in GROUPS:
        positive = sorted(((values[i] / a, i, a) for i, a in group if values[i] > 0), reverse=True)
        if math.fsum(values[i] * a for _, i, a in positive) <= cap:
            for _, i, _ in positive:
                result[i] = values[i]
            continue
        numerator_terms, denominator_terms = [], []
        found = False
        for k, (_, index, coefficient) in enumerate(positive):
            numerator_terms.append(coefficient * values[index])
            denominator_terms.append(coefficient * coefficient)
            tau = (math.fsum(numerator_terms) - cap) / math.fsum(denominator_terms)
            next_breakpoint = positive[k + 1][0] if k + 1 < len(positive) else -math.inf
            if tau >= next_breakpoint:
                for _, i, a in positive:
                    result[i] = max(0., values[i] - tau * a)
                found = True
                break
        require(found, 'weighted-simplex active set not found')
        # At most a few ulps of feasibility correction; never claim exact real arithmetic.
        total = math.fsum(result[i] * a for i, a in group)
        if total > cap:
            scale = cap / total
            for i, _ in group:
                result[i] *= scale
    return require_feasible(result, cap=cap)


def sufficient_statistics(rows, targets):
    """Uncentered 1/N X'X, 1/N X'y, and 1/N y'y. No intercept/feature centering."""
    rows, targets = list(rows), tuple(targets)
    require(len(rows) == len(targets) and len(rows) > 0, 'empty or mismatched observations')
    rows = [finite_vector(row) for row in rows]
    targets = finite_vector(targets, len(targets))
    n = len(rows)
    gram = [[math.fsum(row[i] * row[j] for row in rows) / n for j in range(13)] for i in range(13)]
    rhs = [math.fsum(row[i] * y for row, y in zip(rows, targets)) / n for i in range(13)]
    yy = math.fsum(y * y for y in targets) / n
    require(all(math.isfinite(v) for row in gram for v in row)
            and all(math.isfinite(v) for v in rhs) and math.isfinite(yy), 'nonfinite sufficient statistics')
    return {'count': n, 'gram': gram, 'rhs': rhs, 'mean_target_square': yy}


def fit(rows, targets, ridge, *, max_iterations=250000, distance_tolerance_cp=0.0001):
    return fit_statistics(sufficient_statistics(rows, targets), ridge,
                          max_iterations=max_iterations, distance_tolerance_cp=distance_tolerance_cp)


def fit_statistics(stats, ridge, *, max_iterations=250000, distance_tolerance_cp=0.0001):
    """Fixed-step projected gradient for a strictly convex quadratic.

    L=2*max absolute row sum(A), A=X'X/N+ridge*I, upper-bounds the gradient
    Lipschitz constant. mu=2*ridge is a conservative strong-convexity bound.
    For z=Proj(y-grad(y)/L), n=L(y-z)-grad(y) is a normal at z. Thus
    residual=grad(z)+n belongs to the objective's constrained subgradient.
    ||residual||/mu bounds distance to the unique optimum in exact arithmetic.
    Stopping records this float64 KKT certificate; no convergence -> exception.
    """
    require(type(ridge) in (float, int) and math.isfinite(ridge) and ridge > 0, 'ridge must be positive')
    require(type(max_iterations) is int and max_iterations > 0, 'invalid iteration limit')
    require(math.isfinite(distance_tolerance_cp) and 0 < distance_tolerance_cp <= 0.01, 'invalid convergence tolerance')
    gram = [finite_vector(row) for row in stats['gram']]
    rhs = finite_vector(stats['rhs'])
    require(len(gram) == 13 and stats['count'] > 0, 'invalid sufficient statistics shape/count')
    require(all(abs(gram[i][j] - gram[j][i]) <= 1e-10 for i in range(13) for j in range(13)), 'Gram matrix not symmetric')
    # Only statistics produced by sufficient_statistics are supported: X'X is PSD.
    require(all(gram[i][i] >= 0 for i in range(13)), 'negative Gram diagonal')
    a = [[gram[i][j] + (ridge if i == j else 0.) for j in range(13)] for i in range(13)]
    b = [rhs[i] + ridge * PRIOR[i] for i in range(13)]
    lipschitz = 2. * max(math.fsum(abs(v) for v in row) for row in a)
    require(math.isfinite(lipschitz) and lipschitz > 0, 'invalid gradient bound')
    strong_convexity = 2. * ridge
    current = require_feasible(PRIOR)
    for iteration in range(1, max_iterations + 1):
        gradient = [2. * (dot(a[i], current) - b[i]) for i in range(13)]
        candidate = project([current[i] - gradient[i] / lipschitz for i in range(13)])
        displacement = [current[i] - candidate[i] for i in range(13)]
        residual = [lipschitz * displacement[i] - 2. * dot(a[i], displacement) for i in range(13)]
        residual_norm = math.sqrt(math.fsum(r * r for r in residual))
        distance_bound = residual_norm / strong_convexity
        require(math.isfinite(distance_bound), 'nonfinite convergence certificate')
        if distance_bound <= distance_tolerance_cp:
            mse = dot(candidate, [dot(gram[i], candidate) for i in range(13)]) - 2. * dot(rhs, candidate) + stats['mean_target_square']
            penalty = ridge * math.fsum((v-p) ** 2 for v, p in zip(candidate, PRIOR))
            require(math.isfinite(mse) and mse >= -1e-5, 'invalid fitted MSE')
            return {'status': 'converged', 'method': 'fixed-step Euclidean projected gradient; float64 KKT certificate',
                    'ridge': float(ridge), 'parameters': list(candidate), 'prior': list(PRIOR),
                    'count': stats['count'], 'iterations': iteration, 'max_iterations': max_iterations,
                    'fit_cap_cp': FIT_CAP, 'group_maxima_cp': list(group_totals(candidate)),
                    'lipschitz_upper_bound': lipschitz, 'strong_convexity_lower_bound': strong_convexity,
                    'kkt_residual_l2': residual_norm, 'distance_bound_cp': distance_bound,
                    'distance_tolerance_cp': distance_tolerance_cp,
                    'objective_gap_bound': residual_norm ** 2 / (2. * strong_convexity),
                    'train_mse_cp2': max(0., mse), 'ridge_penalty': penalty,
                    'objective': max(0., mse) + penalty,
                    'certificate_caveat': 'mathematical bounds evaluated in float64; not an interval-arithmetic proof'}
        current = candidate
    raise FitError(f'projected gradient did not converge within {max_iterations} iterations; no candidate returned')


def round_parameters(fit_result):
    require(fit_result.get('status') == 'converged', 'cannot encode an unconverged fit')
    raw = require_feasible(fit_result['parameters'])
    require(fit_result['distance_bound_cp'] <= fit_result['distance_tolerance_cp'] <= 0.01,
            'fit convergence certificate missing or invalid')
    rounded = tuple(2 * round(value / 2.) for value in raw)
    require(all(type(v) is int and v >= 0 for v in rounded), 'rounding produced invalid parameters')
    require(all(abs(r-v) <= 1.000000001 for r, v in zip(rounded, raw)), 'unexpected even-cp rounding error')
    require_feasible(rounded, cap=ENCODED_CAP, tolerance=0.)
    require(all(v % 2 == 0 for v in rounded), 'parameters are not even cp')
    return rounded


def exact_predictions(rows, parameters):
    """Integer scores of the encoded material evaluator, not raw fit floats."""
    require(all(type(v) is int and v >= 0 and v % 2 == 0 for v in parameters), 'expected nonnegative even parameters')
    require_feasible(parameters, cap=ENCODED_CAP, tolerance=0.)
    result = []
    for row in rows:
        require(len(row) == 13 and all(type(v) is int for v in row), 'expected integer count features')
        result.append(sum(x * w for x, w in zip(row, parameters)))
    return result


def canonical_sha256(value):
    import json
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def static_metrics(predictions, targets):
    require(len(predictions) == len(targets) and predictions, 'metrics require nonempty equal counts')
    errors = [p-y for p, y in zip(predictions, targets)]
    absolute_sum = sum(abs(error) for error in errors)
    return {'count': len(targets), 'absolute_error_sum_cp': absolute_sum,
            'mae_cp': absolute_sum / len(targets), 'bias_cp': math.fsum(errors) / len(targets),
            'rmse_cp': math.sqrt(math.fsum(e*e for e in errors) / len(targets))}


def fit_grid(rows, targets, **solver_options):
    """Train only. Holdout must never enter rows/targets; caller uses verified_split."""
    stats = sufficient_statistics(rows, targets)
    results = []
    for ridge in LAMBDAS:
        try:
            results.append(fit_statistics(stats, ridge, **solver_options))
        except FitError as error:
            raise FitError(f'predeclared lambda {ridge} failed: {error}') from error
    return results


def rank_static_candidates(items):
    """Require all five predeclared candidates and exact core confirmation first."""
    require(len(items) == len(LAMBDAS) and {item['ridge'] for item in items} == set(LAMBDAS), 'incomplete fixed lambda grid')
    require(all(item.get('core_verified') is True and item.get('split') == 'holdout' for item in items),
            'cannot select before exact heldout core validation')
    bindings = {item['holdout_binding_sha256'] for item in items}
    require(len(bindings) == 1 and all(type(v) is str and len(v) == 64 for v in bindings),
            'candidate holdout identity/order/targets differ')
    counts = {item['holdout']['count'] for item in items}
    require(len(counts) == 1 and next(iter(counts)) > 0, 'different holdout denominators')
    for item in items:
        require(type(item['holdout']['absolute_error_sum_cp']) is int
                and item['holdout']['absolute_error_sum_cp'] >= 0, 'selection needs integer error sums')
    return sorted(items, key=lambda item: (item['holdout']['absolute_error_sum_cp'], -item['ridge']))


def ft_material_channels(feature, values):
    """Pure channel arithmetic, testable without generating a weight model."""
    require(type(feature) is int and 0 <= feature < INPUT, 'invalid flat feature index')
    require(len(values) == 14 and all(type(v) is int and v >= 0 and v % 2 == 0 for v in values),
            'expected fourteen nonnegative even piece prices')
    channels = [0, 0]
    if feature < BOARD_INPUT:
        kind, opponent = (feature % 28) // 2, feature % 2
        if opponent == 0 and kind != 7:
            channels[0 if kind in (0, 8) else 1] = values[kind] // 2
    else:
        bank, threshold = divmod(feature - BOARD_INPUT, HAND_THRESHOLDS)
        hand_color, perspective = divmod(bank, 2)
        kind = next(k for k in range(7) if HAND_OFFSETS[k] <= threshold < HAND_OFFSETS[k] + HAND_MAX[k])
        if hand_color == perspective:
            channels[0 if kind == 0 else 1] = values[kind] // 2
    return tuple(channels)


def encode_candidate(original, fit_result):
    """Only call during a controlled private run. No files are written here.

    Seed42 baseline must match exactly. Replace only the two material FT channels
    per feature. All auxiliary FT, FT bias, L2, output, and magic remain unchanged.
    """
    require(hashlib.sha256(original).hexdigest() == MATERIAL_SEED42_SHA256,
            'original material seed42 binary identity mismatch')
    require(len(original) == BINARY_BYTES and original[:8] == b'SEKIRW01', 'unexpected model layout')
    parameters = round_parameters(fit_result)
    values = tuple(int(v) for v in prices(parameters))
    require(all(v >= 0 and v % 2 == 0 for v in values), 'invalid encoded piece prices')
    output = bytearray(original)
    for feature in range(INPUT):
        offset = 8 + 2 * feature * L1
        channels = ft_material_channels(feature, values)
        struct.pack_into('<hh', output, offset, *channels)
        require(output[offset + 4:offset + 2 * L1] == original[offset + 4:offset + 2 * L1], 'auxiliary FT bytes changed')
    ft_end = 8 + 2 * INPUT * L1
    require(output[:8] == original[:8] and output[ft_end:] == original[ft_end:], 'nonmaterial model bytes changed')
    recipe = {'format': 'sekirei-constrained-material-candidate-v1', 'status': 'encoded',
              'parameters': dict(zip(PARAMETERS, parameters)), 'piece_values': values,
              'fit': fit_result, 'rounding': 'nearest even centipawn per parameter; half ties to even integer',
              'rounding_max_group_upward_cp': [36, 36], 'fit_cap_cp': FIT_CAP,
              'encoded_cap_cp': ENCODED_CAP, 'encoded_group_maxima_cp': group_totals(parameters),
              'original_sha256': MATERIAL_SEED42_SHA256, 'output_sha256': hashlib.sha256(output).hexdigest(),
              'output_bytes': len(output), 'ft_material_channels_only': True,
              'nonmaterial_bytes_preserved': True, 'engine_verified': False,
              'static_selection_verified': False, 'adoption': False}
    return bytes(output), recipe


def validate_probe_rows(rows, feature_rows, parameters):
    """Bind existing core_pair_probe output to exact linear predictions, in order.

    Runner must separately bind stdin SFEN order, probe/input hashes, zero exit,
    MXCSR log, timeout cleanup, and fixed-core build. This validates parsed rows.
    """
    expected_new = exact_predictions(feature_rows, parameters)
    expected_old = [int(dot(row, PRIOR)) for row in feature_rows]
    require(len(rows) == len(feature_rows) and len(rows) > 0, 'probe count mismatch')
    keys = {'index', 'native_core_cp', 'nearest_core_cp', 'native_quantized_float_cp',
            'nearest_quantized_float_cp', 'material_cp'}
    for index, (row, old, new) in enumerate(zip(rows, expected_old, expected_new)):
        require(isinstance(row, dict) and set(row) == keys, 'probe output schema mismatch')
        require(type(row['index']) is int and row['index'] == index, 'probe ordering mismatch')
        for key, expected in [('native_core_cp', old), ('material_cp', old), ('nearest_core_cp', new)]:
            require(type(row[key]) is int and row[key] == expected, 'core/material prediction is not exact')
        for key, expected in [('native_quantized_float_cp', old), ('nearest_quantized_float_cp', new)]:
            value = row[key]
            require(type(value) in (int, float) and math.isfinite(value), 'nonfinite probe float')
            require(struct.unpack('<f', struct.pack('<f', value))[0] == expected, 'quantized float is not exact integer material')
    return {'count': len(rows), 'native_core_exact': True, 'candidate_core_exact': True,
            'native_float_exact': True, 'candidate_float_exact': True}


def sha_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fit_verified_dataset(dataset, diagnostic_module, material_module, **solver_options):
    """Operational primitive; NOT called by this module's public tests.

    The caller must hold the existing runtime training/benchmark locks, enforce
    process wall time, and keep returned per-position information private. All
    five fits use train only. Frozen holdout yields unverified analytic metrics;
    rank_static_candidates refuses selection until the caller verifies all cores.
    """
    import json
    dataset = Path(dataset).resolve()
    manifest_path = dataset / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    expected_files = {f'{split}.{kind}.jsonl' for split in ('train', 'holdout') for kind in ('positions', 'labels')}
    require(set(manifest['files']) == expected_files, 'unexpected dataset file set')
    paths = [manifest_path] + [dataset / name for name in sorted(expected_files)]
    paths += [Path(__file__).resolve(), Path(diagnostic_module.__file__).resolve(), Path(material_module.__file__).resolve()]
    require(tuple(material_module.VALUES) == NATIVE_VALUES, 'material parser baseline mapping differs')
    before = {str(path): {'sha256': sha_file(path), 'bytes': path.stat().st_size} for path in paths}
    train_manifest, train_targets = diagnostic_module.verified_split(dataset, 'train')
    holdout_manifest, holdout_targets = diagnostic_module.verified_split(dataset, 'holdout')
    require(train_manifest == holdout_manifest == manifest, 'dataset changed while loading splits')
    train_positions = diagnostic_module.strict_rows(dataset / 'train.positions.jsonl')
    holdout_positions = diagnostic_module.strict_rows(dataset / 'holdout.positions.jsonl')
    train_rows = [features(material_module.parse_sfen(row['sfen'])) for row in train_positions]
    holdout_rows = [features(material_module.parse_sfen(row['sfen'])) for row in holdout_positions]
    require(len(train_rows) == len(train_targets) and len(holdout_rows) == len(holdout_targets), 'position order/count mismatch')
    fits = fit_grid(train_rows, train_targets, **solver_options)
    holdout_binding = canonical_sha256({'sfens': [row['sfen'] for row in holdout_positions],
                                       'targets': holdout_targets, 'features': holdout_rows})
    candidates = []
    for result in fits:
        result.update(fit_input_files=before, fit_split='train', teacher_identity=manifest['teacher_identity'],
                      feature_schema='seven-base-plus-six-nonnegative-promotion-increments-v1',
                      holdout_used_for_fitting=False)
        parameters = round_parameters(result)
        train_predictions = exact_predictions(train_rows, parameters)
        holdout_predictions = exact_predictions(holdout_rows, parameters)
        candidates.append({'ridge': result['ridge'], 'fit': result, 'rounded_parameters': list(parameters),
                           'split': 'holdout', 'core_verified': False, 'holdout_binding_sha256': holdout_binding,
                           'train': static_metrics(train_predictions, train_targets),
                           'holdout': static_metrics(holdout_predictions, holdout_targets)})
    require({str(path): {'sha256': sha_file(path), 'bytes': path.stat().st_size} for path in paths} == before,
            'dataset changed during fit')
    return {'status': 'fit_only_core_unverified', 'candidates': candidates,
            'teacher_identity': manifest['teacher_identity'], 'dataset_inputs': before,
            'holdout_features': holdout_rows,
            'holdout_sfens': [row['sfen'] for row in holdout_positions], 'holdout_targets': holdout_targets,
            'selection_rule': 'all five core-exact candidates; minimum exact holdout integer MAE, ties prefer higher lambda',
            'selected_ridge': None, 'adoption': False}


def write_candidate(original_path, fit_result, output, *, source, material_module,
                    input_paths, teacher_identity, dataset_manifest_path):
    """Operational writer; never called by the public tests.

    Input paths must include every file consumed by the external driver (dataset,
    original diagnostic/build identity and driver/helpers). Recorded hashes are
    checked before/after. The source validator uses unchanged upstream VALUES;
    no module globals are patched. No Adam checkpoint is manufactured.
    """
    import json
    import os
    import subprocess
    source, original_path, output = Path(source).resolve(), Path(original_path).resolve(), Path(output).resolve()
    dataset_manifest_path = Path(dataset_manifest_path).resolve()
    require(output.parent.is_dir() and not output.exists(), 'output parent must exist and output must be new')
    git = subprocess.run(['git', '-C', str(output.parent), 'rev-parse', '--is-inside-work-tree'],
                         capture_output=True, text=True, timeout=10)
    require(not (git.returncode == 0 and git.stdout.strip() == 'true'), 'weights/recipes must remain outside Git')
    require(isinstance(teacher_identity, str) and teacher_identity.startswith('external:') and len(teacher_identity) > 9,
            'expected external teacher identity')
    identity = material_module.verify_source(source)
    require(tuple(material_module.VALUES) == NATIVE_VALUES, 'material helper baseline values changed')
    require(Path(material_module.__file__).is_file(), 'material helper path unavailable')
    dataset_manifest = json.loads(dataset_manifest_path.read_text())
    require(dataset_manifest['teacher_identity'] == teacher_identity, 'teacher identity differs from dataset')
    expected_files = {f'{split}.{kind}.jsonl' for split in ('train', 'holdout') for kind in ('positions', 'labels')}
    require(set(dataset_manifest['files']) == expected_files, 'unexpected dataset manifest files')
    paths = {Path(p).resolve() for p in input_paths}
    paths |= {Path(__file__).resolve(), Path(material_module.__file__).resolve(), original_path, dataset_manifest_path}
    paths |= {source / relative for relative in material_module.SOURCE_HASHES}
    paths |= {dataset_manifest_path.parent / name for name in expected_files}
    before = {str(path): {'sha256': sha_file(path), 'bytes': path.stat().st_size} for path in sorted(paths)}
    for name, info in dataset_manifest['files'].items():
        current = before[str(dataset_manifest_path.parent / name)]
        require(current['sha256'] == info['sha256'] and current['bytes'] == info['bytes'], 'dataset file hash/size mismatch')
    require(fit_result.get('fit_split') == 'train' and fit_result.get('holdout_used_for_fitting') is False
            and fit_result.get('teacher_identity') == teacher_identity, 'fit provenance/split/teacher mismatch')
    fit_inputs = fit_result.get('fit_input_files')
    require(isinstance(fit_inputs, dict) and fit_inputs, 'fit input identity missing')
    for path, info in fit_inputs.items():
        require(path in before and before[path] == info, 'fit input identity differs from generation input')
    binary, recipe = encode_candidate(original_path.read_bytes(), fit_result)
    recipe.update(source_identity=identity, input_files=before, teacher_identity=teacher_identity,
                  dataset_manifest_sha256=before[str(dataset_manifest_path)]['sha256'], fit_split='train',
                  sidecar_format='sekirei-nnue-output-v1', lambda_grid=list(LAMBDAS))
    output.mkdir(mode=0o700)
    previous_umask = os.umask(0o077)
    try:
        weights_path = output / 'material-fit.bin'
        with weights_path.open('xb') as stream:
            stream.write(binary)
            stream.flush()
            os.fsync(stream.fileno())
        sidecar = {'format': 'sekirei-nnue-output-v1', 'nnue_output': 'absolute',
                   'checkpoint_hash': material_module.fnv1a(binary), 'baseline': None}
        (output / 'material-fit.meta.json').write_text(json.dumps(sidecar, indent=2, allow_nan=False) + '\n')
        require(weights_path.read_bytes() == binary, 'written model bytes differ')
        after = {str(path): {'sha256': sha_file(path), 'bytes': path.stat().st_size} for path in sorted(paths)}
        require(before == after, 'input changed during candidate encoding')
        recipe.update(status='encoded_core_unverified', inputs_unchanged=True)
        (output / 'recipe.json').write_text(json.dumps(recipe, indent=2, sort_keys=True, allow_nan=False) + '\n')
        return recipe
    except BaseException:
        recipe.update(status='failed', engine_verified=False, adoption=False)
        (output / 'recipe.json').write_text(json.dumps(recipe, indent=2, sort_keys=True, allow_nan=False) + '\n')
        raise
    finally:
        os.umask(previous_umask)
