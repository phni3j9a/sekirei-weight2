#!/usr/bin/env python3
"""Strictly compare two explicitly pinned NNUE models on development runs.

Both models need their own complete MAE pilot/formal and Top3 pilot/formal.
The existing parsers revalidate all raw evidence. No default fallback, engine
execution, model registration, or final-set access occurs in this command.
Run with the same pinned audit interpreter used to produce the measurements.
"""
import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_candidates as cc

b = cc.b
REPO = cc.REPO
CHECKS = frozenset((
    'execution_except_model', 'occurrences', 'teacher_results', 'teacher_E_cp',
    'top3_occurrences', 'top3_legal_moves_sha256', 'top3_denominators',
    'top3_teacher_bestmoves',
))
GAMES = {'development-01': 86, 'development-02': 126, 'development-03': 127,
         'development-04': 108, 'development-05': 123}


class OutputError(ValueError):
    """Unsafe output locations must never receive even an invalid report."""


def model_pin(config, expected_sha256):
    cc.require(type(expected_sha256) is str and
               re.fullmatch('[0-9a-f]{64}', expected_sha256),
               'an explicit lowercase SHA-256 model pin is required')
    cc.require(config.get('split') == 'development' and
               config.get('requested_nodes') == 1000000 and
               config.get('universe', {}).get('plies') == GAMES,
               'only the fixed five-game development comparison is allowed')
    identity = b._model_identity(config)
    cc.require(identity['kind'] == 'nnue' and identity['sha256'] == expected_sha256,
               'requested NNUE model differs from the explicit model pin')
    return identity


def validate_verified(value, expected_model):
    """Check the strict reader's contract before comparing its rational metrics."""
    comparable, summary, mae, rate = value
    cc.require(set(comparable) == CHECKS, 'all eight comparison identities are required')
    cc.require(summary['model_identity'] == expected_model,
               'recorded measurement model differs from the explicit model pin')
    cc.require(summary['mae_report_valid'] is True and summary['top3_report_valid'] is True,
               'both formal reports must be valid')
    cc.require((summary['teacher_count'], summary['teacher_E_count'], summary['top3_count']) ==
               (570, 266, 551), 'formal coverage is incomplete')
    teachers, e_cp = comparable['teacher_results'], comparable['teacher_E_cp']
    occurrences = comparable['occurrences']
    # Canonical plan IDs are padded (development-01:001); the legacy
    # comparison reader keys teacher results by game_id + integer ply
    # (development-01:1). Bind them through the explicit coordinates rather
    # than requiring those two different serializations to be equal.
    keys, teacher_keys, game_teacher_keys = [], [], {game: [] for game in GAMES}
    for p in occurrences:
        game, ply = p['game_id'], p['ply']
        cc.require(game in GAMES and type(ply) is int and 1 <= ply <= GAMES[game],
                   'invalid MAE occurrence coordinates')
        cc.require(p['occurrence_id'] == f'{game}:{ply:03d}',
                   'MAE occurrence ID differs from canonical game/ply coordinates')
        keys.append(p['occurrence_id'])
        teacher_key = f'{game}:{ply}'
        teacher_keys.append(teacher_key)
        game_teacher_keys[game].append(teacher_key)
    top_keys = comparable['top3_occurrences']
    cc.require(len(keys) == len(set(keys)) == len(set(teacher_keys)) == len(teachers) == 570 and
               set(teacher_keys) == set(teachers), 'MAE occurrence/teacher sets differ')
    cc.require(len(e_cp) == 266 and set(e_cp) <= set(teachers) and
               all(type(v) is int for v in e_cp.values()), 'Teacher-E set is incomplete or malformed')
    cc.require(len(top_keys) == len(set(top_keys)) == 551 and
               set(top_keys) <= set(keys) and
               set(comparable['top3_teacher_bestmoves']) == set(top_keys),
               'Top3 occurrence/reference sets differ')
    cc.require(set(summary['per_game']) == set(GAMES) == set(comparable['top3_denominators']),
               'five-game aggregation mismatch')
    exact_mae, exact_rate = Fraction(0), Fraction(0)
    for game in GAMES:
        item = summary['per_game'][game]
        n = sum(key in e_cp for key in game_teacher_keys[game])
        top = item['top3']
        denominator = sum(key.startswith(game + ':') for key in top_keys)
        cc.require(type(item['e_count']) is int and item['e_count'] == n and n > 0,
                   'per-game Teacher-E coverage differs')
        cc.require(type(top['hits']) is int and type(top['denominator']) is int and
                   0 <= top['hits'] <= top['denominator'] == denominator and denominator > 0 and
                   top['observed'] == denominator == comparable['top3_denominators'][game],
                   'per-game Top3 coverage differs')
        rational_mae = item['mae']
        cc.require(type(rational_mae['numerator']) is int and type(rational_mae['denominator']) is int
                   and rational_mae['numerator'] >= 0 and rational_mae['denominator'] > 0,
                   'invalid exact MAE')
        game_mae = Fraction(rational_mae['numerator'], rational_mae['denominator'])
        cc.require((game_mae * n).denominator == 1, 'MAE cannot represent an integer error sum')
        exact_mae += game_mae / 5
        exact_rate += Fraction(top['hits'], denominator) / 5
    cc.require(type(mae) is Fraction and type(rate) is Fraction and
               (mae, rate) == (exact_mae, exact_rate), 'exact equal-game metrics differ')
    return value


def verify(desc, config, expected_sha256):
    expected = model_pin(config, expected_sha256)
    # This reader validates the model activation response, raw SHA, USI
    # lifecycle, nodes, cleanup, complete pilots, and both formal reports.
    value = cc.verify_run(desc, config)
    return validate_verified(value, expected)


def compare_verified(incumbent, candidate):
    checks = cc.compare(incumbent[0], candidate[0])
    cc.require(set(checks) == CHECKS, 'all eight comparison identities are required')
    failed = [name for name, item in checks.items() if not item['equal']]
    cc.require(not failed, 'reference/settings mismatch: ' + ', '.join(sorted(failed)))
    # Full runtime/build/binary/teacher/source/options identities stay exact.
    # cc.normalized_execution removes only NNUE model identity, EvalFile and
    # an explicit default absolute NnueOutput; it does not weaken a binary.
    improved = candidate[2] < incumbent[2]
    preserved = candidate[3] >= incumbent[3]
    adopt = improved and preserved
    decision = 'adopt' if adopt else 'hold' if (
        improved or candidate[3] > incumbent[3]) else 'reject'
    return {'status': 'complete', 'comparison_valid': True, 'checks': checks,
            'incumbent': incumbent[1], 'candidate': candidate[1],
            'mae_improved': improved, 'top3_preserved': preserved,
            'adopt': adopt, 'decision': decision}


def protected_output(output, descriptors, prepared):
    if os.path.lexists(output):
        raise OutputError('output already exists; no overwrites')
    protected = [REPO, Path('/home/server/projects/sekirei-weight2')]
    for desc in descriptors.values():
        protected.append(desc['runtime'])
        protected.append(desc['config'].parent)
    for _config, paths, directories in prepared.values():
        protected.extend(directories)
        protected.extend(paths)
    if any(output == path or output.is_relative_to(path) for path in protected):
        raise OutputError('output must be a new private file outside input runtimes/config directories/repositories')


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for role in ('incumbent', 'candidate'):
        result.add_argument('--' + role + '-config', type=Path, required=True)
        result.add_argument('--' + role + '-mae-run', type=Path, required=True)
        result.add_argument('--' + role + '-top3-run', type=Path, required=True)
        result.add_argument('--' + role + '-model-sha256', required=True)
    result.add_argument('--output-json', type=Path, required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    output = args.output_json.expanduser().resolve()
    # Refuse an existing result even when input validation subsequently fails.
    cc.require(not os.path.lexists(output), 'output already exists; no overwrites')
    cc.require(not output.is_relative_to(REPO) and
               not output.is_relative_to(Path('/home/server/projects/sekirei-weight2')),
               'output must be outside repositories')
    # Do this before reading manifests: malformed/missing inputs must not let
    # the invalid-result path write back into either input runtime.
    for role in ('incumbent', 'candidate'):
        for suffix, parents in (('config', 1), ('mae_run', 2), ('top3_run', 2)):
            path = getattr(args, role + '_' + suffix).expanduser().resolve()
            protected = path.parents[parents - 1]
            if output.is_relative_to(protected):
                raise OutputError('output is inside an input config directory or runtime')
    result = {'schema': 'sekirei.incumbent-formal-comparison.v1',
              'status': 'invalid', 'comparison_valid': False, 'adopt': False,
              'decision': 'invalid', 'created_at': b.utc_now(), 'final_used': False,
              'requested_nodes': 1000000, 'no_engine_process_started': True,
              'policy': 'strictly lower five-game mean MAE and non-decreasing five-game mean Top3; exact rational comparison',
              'comparison_exclusions': ['candidate model identity/EvalFile; absent vs explicit absolute NnueOutput'],
              'model_pins': {role: getattr(args, role + '_model_sha256') for role in ('incumbent', 'candidate')}}
    code = 2
    try:
        descriptors = {role: cc.descriptor(getattr(args, role + '_config'),
                        getattr(args, role + '_mae_run'), getattr(args, role + '_top3_run'))
                       for role in ('incumbent', 'candidate')}
        prepared = {role: cc.prepare_inputs(desc) for role, desc in descriptors.items()}
        protected_output(output, descriptors, prepared)
        paths = [Path(__file__), b.BENCHMARK_MANIFEST_PATH]
        paths.extend(REPO / 'scripts' / name for name in
                     ('benchmark.py', 'benchmark_report.py', 'top3.py', 'compare_candidates.py',
                      'prepare.py', 'audit_pack.py', 'smoke.py'))
        paths.extend(REPO / name for name in
                     ('config/toolchain.lock.json', 'config/top3-benchmark.json',
                      'config/development-position-classifications.json', 'config/audit-requirements.txt'))
        directories = []
        for config, own_paths, own_dirs in prepared.values():
            paths.extend(own_paths)
            directories.extend(own_dirs)
            paths.extend(b._safe_development_path(game['path']) for game in config['games'])
        before = cc.inventory(paths, directories)
        result['input_receipt'] = before
        result['sources'] = {role: {key: str(value) for key, value in desc.items()
                                    if key in ('config', 'mae', 'top3', 'runtime')}
                             for role, desc in descriptors.items()}
        verified = {role: verify(desc, prepared[role][0], result['model_pins'][role])
                    for role, desc in descriptors.items()}
        after = cc.inventory(paths, directories)
        result['inputs_unchanged'] = before == after
        cc.require(result['inputs_unchanged'], 'comparison inputs changed while being read')
        result.update(compare_verified(verified['incumbent'], verified['candidate']))
        code = 0
    except OutputError:
        raise
    except Exception as error:
        result.update(status='invalid', comparison_valid=False, adopt=False, decision='invalid',
                      error_type=type(error).__name__, private_error=str(error))
    result['finished_at'] = b.utc_now()
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: result.get(key) for key in
                     ('status', 'comparison_valid', 'decision', 'adopt', 'mae_improved', 'top3_preserved')}, ensure_ascii=False))
    return code


if __name__ == '__main__':
    sys.exit(main())
