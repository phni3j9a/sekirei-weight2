#!/usr/bin/env python3
"""Read-only validation/comparison of completed development MAE and Top3 runs.

Run with the pinned cshogi/NumPy venv. Only a new private output JSON is written.
Exit 0 means the comparison was valid (including rejected/held candidates), not
that adoption passed. Exit 2 means invalid/uncomparable evidence. No final data.
"""
import argparse
import copy
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
import benchmark as b
import benchmark_report as br
import top3
from prepare import DEFAULT_RUNTIME, sha256

BASE_MAE = DEFAULT_RUNTIME / 'runs/development-baseline-20261002-v039-v1'
BASE_TOP3 = DEFAULT_RUNTIME / 'runs/development-top3-baseline-20261002-v1'
TEACHER_FIELDS = (
    'status', 'position_type', 'position_classification', 'requested_nodes',
    'score_kind', 'score_bound_stm', 'score_bound_sente', 'score_cp_stm',
    'score_cp_sente', 'reported_cp_stm', 'reported_cp_sente', 'score_mate_stm',
    'mate_distance', 'mate_distance_known', 'mate_sign', 'winner', 'winner_stm',
    'winner_sente', 'bestmove', 'bestmove_kind', 'pv', 'pv_head',
)
OCCURRENCE_FIELDS = (
    'game_id', 'ply', 'occurrence_id', 'position', 'side_to_move',
    'position_type', 'classification',
)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read_json(path):
    return json.loads(Path(path).read_text())


def run_path(path):
    path = Path(path).expanduser().resolve(strict=True)
    b._validate_run_id(path.name)
    require(path.is_dir() and path.parent.name == 'runs', 'expected a runtime/runs directory')
    require('final' not in path.name.casefold(), 'final runs are forbidden')
    return path


def descriptor(config, mae, ranking, evaluation=None):
    mae, ranking = run_path(mae), run_path(ranking)
    require(mae.parent == ranking.parent, 'MAE and Top3 must share their runtime')
    return {'config': Path(config).expanduser().resolve(strict=True), 'mae': mae,
            'top3': ranking, 'runtime': mae.parent.parent, 'evaluation': evaluation}


def from_evaluation(directory):
    directory = directory.expanduser().resolve(strict=True)
    state_path = directory / 'evaluation.json'
    state = read_json(state_path)
    # Check the completion marker before reading any candidate run artifacts.
    require(state.get('status') == 'complete', 'evaluation is not complete; no candidate runs read')
    ids = state['run_ids']
    for key in ('mae-pilot', 'mae', 'top3-pilot', 'top3'):
        b._validate_run_id(ids[key])
        require('final' not in ids[key].casefold(), 'final runs are forbidden')
    result = descriptor(directory / 'candidate.json', DEFAULT_RUNTIME / 'runs' / ids['mae'],
                        DEFAULT_RUNTIME / 'runs' / ids['top3'], state_path)
    result['state'] = state
    return result


def inventory(paths, directories):
    files = set(Path(p).resolve(strict=True) for p in paths)
    directory_entries = {}
    for directory in directories:
        directory = Path(directory).resolve(strict=True)
        children = sorted(p for p in directory.rglob('*') if p.is_file())
        directory_entries[str(directory)] = [str(p.relative_to(directory)) for p in children]
        files.update(p.resolve(strict=True) for p in children)
    entries = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in sorted(files)]
    return {'files': entries, 'directories': directory_entries, 'sha256': b.json_sha256(entries)}


def prepare_inputs(desc):
    config = b.load_config(desc['config'])
    require(config.get('requested_nodes') == 1000000, 'MAE must use 1000000 nodes')
    require(config.get('split') == 'development', 'only development is allowed')
    gate = config['formal']['pilot_evidence']
    b._validate_run_id(gate['pilot_run_id'])
    mae_pilot = run_path(desc['runtime'] / 'runs' / gate['pilot_run_id'])
    manifest = read_json(desc['mae'] / 'manifest.json')
    require(manifest.get('status') == 'complete' and manifest.get('run_type') == 'formal',
            'MAE is not a completed formal run')
    ranking_manifest = read_json(desc['top3'] / 'manifest.json')
    require(ranking_manifest.get('run_type') == 'formal', 'Top3 must be formal')
    b._validate_run_id(ranking_manifest['pilot_run_id'])
    ranking_pilot = run_path(desc['runtime'] / 'runs' / ranking_manifest['pilot_run_id'])
    require(Path(ranking_manifest['mae_run']).resolve() == desc['mae'], 'Top3 parent path mismatch')
    if desc.get('state'):
        ids = desc['state']['run_ids']
        require(ids['mae-pilot'] == mae_pilot.name and ids['top3-pilot'] == ranking_pilot.name,
                'evaluation summary pilot IDs differ from evidence')
    paths = [desc['config'], desc['runtime'] / 'build-manifest.json']
    runtime_identity = manifest['runtime_identity']
    paths.extend(Path(value['path']) for value in runtime_identity['binaries'].values())
    if runtime_identity['teacher_weight']:
        paths.append(Path(runtime_identity['teacher_weight']['path']))
    if desc['evaluation']:
        paths.append(desc['evaluation'])
    model = config['candidate_model']
    if model['kind'] == 'nnue':
        paths.append(Path(model['path']))
    directories = [desc['mae'], desc['top3'], mae_pilot, ranking_pilot]
    return config, paths, directories


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def integer_cp(value):
    require(type(value) is int, 'exact cp must be the parser integer, not a rounded value')
    return value


def rational(value):
    return {'numerator': value.numerator, 'denominator': value.denominator,
            'float': float(value)}


def close_to_fraction(value, exact):
    return finite(value) and math.isclose(value, float(exact), rel_tol=2e-15, abs_tol=1e-12)


def normalized_execution(identity):
    """Only loaded-model identity/path and explicit default absolute mode may differ."""
    identity = copy.deepcopy(identity)
    identity.pop('candidate_model')
    identity['runtime'].pop('candidate_model')
    for options in (identity['resolved_options']['sekirei'], identity['engines']['sekirei']['options']):
        require(options.get('NnueOutput', 'absolute') == 'absolute', 'absolute output required')
        options.pop('EvalFile', None)
        options.pop('NnueOutput', None)
    identity['resolved_option_order']['sekirei'] = [
        key for key in identity['resolved_option_order']['sekirei'] if key not in ('EvalFile', 'NnueOutput')]
    return identity


def verify_run(desc, config):
    b.validate_runtime(desc['runtime'], config)
    b.validate_formal_gate(desc['runtime'], config)
    pilot_path = desc['runtime'] / 'runs' / config['formal']['pilot_evidence']['pilot_run_id']
    pilot = br.report_from_run(pilot_path, config=config)
    require(pilot['validity']['complete_evidence_valid'], 'MAE pilot invalid')
    require(set(pilot['repeatability']['engines']) == {'teacher', 'sekirei'}, 'pilot engine set mismatch')
    require(all(x['stable_position_count'] == 17 and x['unstable_position_count'] == 0
                for x in pilot['repeatability']['engines'].values()), 'MAE pilot unstable')
    manifest, plan, _, records, _, _ = b.validate_run_artifacts(
        desc['mae'], config, require_complete=True, expected_run_type='formal')
    report = br.report_from_run(desc['mae'], config=config)
    require(report['validity']['formal_run_valid'] and report['headline']['valid'], 'MAE report invalid')
    require(report['universe']['total_occurrences'] == 570, 'expected 570 MAE occurrences')
    require(report['e_exact_coverage']['teacher_e_count'] == 266 and
            report['e_exact_coverage']['candidate_exact_count'] == 266, 'expected complete Teacher-E 266')
    ctx = top3.context(desc['runtime'], config, desc['mae'])
    identity, eligible, teacher_reference, _, _ = ctx
    ranking = top3.inspect(desc['top3'], *ctx, config)
    require(ranking['valid'] and ranking['run_type'] == 'formal' and finite(ranking['top3_rate']),
            'Top3 report invalid/undefined')
    require(ranking['attempts'] == 551 and ranking['eligible_positions'] == 551, 'expected 551 Top3 occurrences')
    require(read_json(desc['top3'] / 'top3-report.json') == ranking,
            'stored Top3 report differs from strict fresh recomputation')
    game_ids = list(config['universe']['plies'])
    require(len(game_ids) == 5 and set(report['per_game']) == set(game_ids) and
            set(ranking['per_game']) == set(game_ids), 'five-game aggregation mismatch')
    teachers = {}
    candidates = {}
    for row in records:
        key = f"{row['game_id']}:{row['ply']}"
        target = teachers if row['engine_id'] == 'teacher' else candidates
        require(key not in target, 'duplicate occurrence')
        target[key] = row
    require(len(teachers) == len(candidates) == 570, 'engine occurrence sets incomplete')
    teacher_projection = {}
    e_cp = {}
    for key, row in teachers.items():
        result = row['result']
        require(all(field in result for field in TEACHER_FIELDS), 'teacher semantic field missing')
        teacher_projection[key] = {'side_to_move': row['side_to_move'], 'position_sha256': row['position_sha256'],
                                   **{field: result[field] for field in TEACHER_FIELDS}}
        if row['status'] == 'exact_cp':
            e_cp[key] = integer_cp(result['score_cp_sente'])
    require(len(e_cp) == 266, 'Teacher-E integer map is not 266 points')
    mae, rate = Fraction(0), Fraction(0)
    metrics = {}
    for game in game_ids:
        keys = [key for key in e_cp if teachers[key]['game_id'] == game]
        require(keys, 'empty Teacher-E game')
        error_sum = sum(abs(integer_cp(candidates[key]['result']['score_cp_sente']) - e_cp[key]) for key in keys)
        game_mae = Fraction(error_sum, len(keys))
        top = ranking['per_game'][game]
        require(type(top['hits']) is int and type(top['denominator']) is int and
                0 <= top['hits'] <= top['denominator'] and top['denominator'] > 0 and
                top['observed'] == top['denominator'], 'invalid Top3 count/denominator')
        game_rate = Fraction(top['hits'], top['denominator'])
        require(close_to_fraction(report['per_game'][game]['mae_cp'], game_mae) and
                close_to_fraction(top['rate'], game_rate), 'per-game metric inconsistency')
        mae += game_mae / 5
        rate += game_rate / 5
        metrics[game] = {'mae': rational(game_mae), 'e_count': len(keys), 'top3': top}
    require(close_to_fraction(report['headline']['mae_cp'], mae) and close_to_fraction(ranking['top3_rate'], rate),
            'headline differs from exact equal-game calculation')
    if desc.get('state'):
        state = desc['state']
        require(state['weight_sha256'] == manifest['runtime_identity']['candidate_model']['sha256'],
                'evaluation summary weight hash mismatch')
        require(state['mae_cp'] == report['headline']['mae_cp'] and state['top3_rate'] == ranking['top3_rate'],
                'evaluation summary metric mismatch')
        require(state.get('top3_report') == ranking, 'evaluation summary Top3 report mismatch')
    comparable = {
        'execution_except_model': normalized_execution(manifest['fingerprint_payload']['execution_identity']),
        'occurrences': [{field: p[field] for field in OCCURRENCE_FIELDS} for p in plan['positions']],
        'teacher_results': teacher_projection,
        'teacher_E_cp': e_cp,
        'top3_occurrences': identity['eligible_occurrences'],
        'top3_legal_moves_sha256': identity['legal_moves_sha256'],
        'top3_denominators': {game: ranking['per_game'][game]['denominator'] for game in game_ids},
        'top3_teacher_bestmoves': {p['occurrence_id']: teacher_reference[(p['game_id'], p['ply'])]['bestmove']
                                  for p in eligible},
    }
    summary = {'mae': rational(mae), 'top3': rational(rate), 'reported_mae_cp': report['headline']['mae_cp'],
               'reported_top3_rate': ranking['top3_rate'], 'per_game': metrics,
               'mae_fingerprint': manifest['fingerprint'], 'top3_fingerprint': ranking['fingerprint'],
               'model_identity': manifest['runtime_identity']['candidate_model'],
               'mae_report_valid': True, 'top3_report_valid': True,
               'teacher_count': 570, 'teacher_E_count': 266, 'top3_count': 551}
    return comparable, summary, mae, rate


def compare(left, right):
    checks = {}
    for field in left:
        a, c = left[field], right[field]
        item = {'equal': a == c, 'baseline_sha256': b.json_sha256(a), 'candidate_sha256': b.json_sha256(c)}
        if isinstance(a, dict) and isinstance(c, dict):
            item['mismatch_count'] = sum(a.get(key) != c.get(key) or key not in a or key not in c
                                         for key in a.keys() | c.keys())
        checks[field] = item
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate_evaluation', type=Path, nargs='?')
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--baseline-evaluation', type=Path)
    parser.add_argument('--candidate-config', type=Path)
    parser.add_argument('--candidate-mae-run', type=Path)
    parser.add_argument('--candidate-top3-run', type=Path)
    args = parser.parse_args()
    explicit = (args.candidate_config, args.candidate_mae_run, args.candidate_top3_run)
    require((args.candidate_evaluation is not None and not any(explicit)) or
            (args.candidate_evaluation is None and all(explicit)), 'choose evaluation directory OR all three explicit candidate paths')
    output = args.output_json.expanduser().resolve()
    require(not output.exists(), 'output already exists; no overwrites')
    forbidden = [REPO, Path('/home/server/projects/sekirei-weight2'), DEFAULT_RUNTIME / 'runs']
    require(not any(output.is_relative_to(path) for path in forbidden), 'output must be a new private file outside repositories/runs')
    result = {'schema': 'sekirei.formal-comparison.v1', 'status': 'validating', 'created_at': b.utc_now(),
              'adopt': False, 'decision': 'invalid', 'teacher_result_fields': list(TEACHER_FIELDS),
              'teacher_record_fields': ['side_to_move', 'position_sha256'],
              'occurrence_fields': list(OCCURRENCE_FIELDS),
              'comparison_exclusions': ['node counters, timing, raw score/log lines, diagnostic line counts',
                                       'candidate model identity/EvalFile; absent vs explicit absolute NnueOutput'],
              'policy': 'strictly lower five-game mean MAE and non-decreasing five-game mean Top3; exact rational comparison',
              'final_used': False, 'requested_nodes': 1000000, 'reasons': []}
    code = 2
    try:
        baseline = from_evaluation(args.baseline_evaluation) if args.baseline_evaluation else descriptor(b.CONFIG_PATH, BASE_MAE, BASE_TOP3)
        candidate = from_evaluation(args.candidate_evaluation) if args.candidate_evaluation else descriptor(*explicit)
        descriptors = {'baseline': baseline, 'candidate': candidate}
        prepared = {name: prepare_inputs(desc) for name, desc in descriptors.items()}
        paths, directories = [Path(__file__), b.BENCHMARK_MANIFEST_PATH], []
        paths.extend(REPO / 'scripts' / name for name in
                     ('benchmark.py', 'benchmark_report.py', 'top3.py', 'prepare.py', 'audit_pack.py', 'smoke.py'))
        paths.extend(REPO / name for name in ('config/toolchain.lock.json', 'config/top3-benchmark.json',
                                             'config/development-position-classifications.json', 'config/audit-requirements.txt'))
        for config, own_paths, own_dirs in prepared.values():
            paths.extend(own_paths)
            directories.extend(own_dirs)
            paths.extend(b._safe_development_path(game['path']) for game in config['games'])
        for desc in descriptors.values():
            if desc['evaluation']:
                require(not output.is_relative_to(desc['evaluation'].parent), 'output must not be inside an input evaluation directory')
        require(not any(output.is_relative_to(path) for path in directories), 'output must not be inside an input run')
        before = inventory(paths, directories)
        result['input_receipt'] = before
        result['sources'] = {name: {key: str(value) for key, value in desc.items()
                                    if key in ('config', 'mae', 'top3', 'runtime', 'evaluation')}
                             for name, desc in descriptors.items()}
        verified = {}
        for name, desc in descriptors.items():
            verified[name] = verify_run(desc, prepared[name][0])
            result[name] = verified[name][1]
        result['checks'] = compare(verified['baseline'][0], verified['candidate'][0])
        after = inventory(paths, directories)
        result['inputs_unchanged'] = before == after
        require(result['inputs_unchanged'], 'comparison inputs changed while being read')
        failed = [name for name, item in result['checks'].items() if not item['equal']]
        require(not failed, 'reference/settings mismatch: ' + ', '.join(failed))
        mae_improved = verified['candidate'][2] < verified['baseline'][2]
        top3_preserved = verified['candidate'][3] >= verified['baseline'][3]
        result.update(status='complete', comparison_valid=True,
                      mae_improved=mae_improved, top3_preserved=top3_preserved,
                      adopt=mae_improved and top3_preserved)
        if result['adopt']:
            result.update(decision='adopt', reasons=['MAE strictly improved and Top3 did not decrease'])
        elif (mae_improved and not top3_preserved) or (not mae_improved and verified['candidate'][3] > verified['baseline'][3]):
            result.update(decision='hold', reasons=['only one metric improved; adoption criteria not met'])
        else:
            result.update(decision='reject', reasons=['joint adoption criteria not met'])
        code = 0
    except Exception as error:
        result.update(status='invalid', comparison_valid=False, adopt=False, decision='invalid',
                      error_type=type(error).__name__, private_error=str(error))
        result['reasons'].append('evidence validation or reference equivalence failed; no adoption')
    result['finished_at'] = b.utc_now()
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    # Deliberately omit paths, raw positions/PVs, references and exception text.
    print(json.dumps({key: result.get(key) for key in
                      ('status', 'comparison_valid', 'decision', 'adopt', 'mae_improved', 'top3_preserved')}, ensure_ascii=False))
    return code


if __name__ == '__main__':
    sys.exit(main())
