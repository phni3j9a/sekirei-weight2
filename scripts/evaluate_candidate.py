"""Evaluate a candidate with its own fixed development MAE and Top3 evidence.

Run with the pinned audit venv. Outputs must be new private directories outside
the checkout. This measurement command does not itself adopt a model.
"""
import argparse
import contextlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
import benchmark as b
import benchmark_report as br
import top3
from prepare import DEFAULT_RUNTIME, sha256


def typed_equal(actual, expected):
    if type(actual) is not type(expected):
        return False
    if type(actual) is dict:
        return actual.keys() == expected.keys() and all(typed_equal(actual[k], v) for k, v in expected.items())
    if type(actual) is list:
        return len(actual) == len(expected) and all(typed_equal(a, v) for a, v in zip(actual, expected))
    return actual == expected


def validate_base_config(config):
    """Keep the published five-game development universe and 1M policy."""
    fixed = b.load_config(b.CONFIG_PATH)
    if (config.get('split') != 'development' or type(config.get('requested_nodes')) is not int
            or config['requested_nodes'] != 1000000):
        raise ValueError('candidate evaluation requires development and exactly 1000000 nodes')
    for key in ('games', 'universe', 'hashes', 'position_classifications'):
        if not typed_equal(config.get(key), fixed[key]):
            raise ValueError('candidate evaluation requires the fixed five-game development ' + key)
    if not typed_equal(config.get('node_policy'), b.NODE_POLICY):
        raise ValueError('candidate evaluation requires the exact runner-owned node policy')
    for run_type in ('pilot', 'formal'):
        b.validate_run_policy(config, run_type)


def verify_physical_model(config, expected):
    actual = b._model_identity(config)
    if not typed_equal(actual, expected):
        raise ValueError('physical candidate model differs from the pinned path/bytes/SHA-256')
    return actual


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate run manifest JSON key')
            result[key] = value
        return result
    def reject(value):
        raise ValueError('nonfinite run manifest JSON constant: ' + value)
    return json.loads(path.read_bytes(), object_pairs_hook=pairs, parse_constant=reject)


def verify_stage_model(run_dir, stage_name, expected):
    manifest = strict_json(run_dir / 'manifest.json')
    run_type = 'pilot' if stage_name.endswith('pilot') else 'formal'
    if manifest.get('run_id') != run_dir.name or manifest.get('run_type') != run_type:
        raise ValueError('candidate stage run ID/type differs from the requested measurement')
    try:
        if stage_name.startswith('mae'):
            actual = manifest['runtime_identity']['candidate_model']
        else:
            actual = manifest['payload']['identity']['base_execution_identity']['runtime']['candidate_model']
    except (KeyError, TypeError) as error:
        raise ValueError('candidate stage model identity is missing') from error
    if not typed_equal(actual, expected):
        raise ValueError('recorded ' + stage_name + ' candidate model differs from the explicit SHA-256 pin')
    return actual


def write_config(path, config):
    # Engine and setoption order are part of the fixed execution contract.
    b.atomic_write(path, json.dumps(config, indent=2, ensure_ascii=False) + '\n')
    loaded = b.load_config(path)
    if loaded != config or list(loaded['engines']) != list(config['engines']):
        raise ValueError('config round trip changed semantics or engine order')
    for engine in config['engines']:
        if list(loaded['engines'][engine]['options'].items()) != list(config['engines'][engine]['options'].items()):
            raise ValueError('config round trip changed setoption order')


def evaluate(args):
    """Run all four stages with the same physically and semantically pinned model."""
    b._validate_run_id(args.prefix)
    output = args.output.resolve()
    weight = args.weight.resolve(strict=True)
    runtime = args.runtime.resolve(strict=True)
    weight_sha256 = sha256(weight)
    expected_sha256 = (weight_sha256 if args.expected_weight_sha256 is None else args.expected_weight_sha256)
    if (type(expected_sha256) is not str or not re.fullmatch('[0-9a-f]{64}', expected_sha256)
            or weight_sha256 != expected_sha256):
        raise ValueError('candidate weight differs from the explicit SHA-256 pin')
    config = b.load_config(args.base_config)
    validate_base_config(config)
    config['candidate_model'] = {'kind': 'nnue', 'path': str(weight)}
    # A base config may describe the incumbent; EvalFile is model identity.
    # Updating its value preserves the supplied option insertion order.
    options = config['engines']['sekirei']['options']
    if 'EvalFile' in options:
        options['EvalFile'] = str(weight)
    config['formal']['pilot_evidence'] = None
    expected_model = b._model_identity(config)
    if expected_model['sha256'] != expected_sha256:
        raise ValueError('candidate changed while preparing its pinned configuration')
    if output.exists() or output.is_relative_to(REPO):
        raise ValueError('use a new private output outside the repository')
    output.mkdir(parents=True)
    shutil.copyfile(__file__, output / 'evaluation-script.py')
    config_path = output / 'candidate.json'
    write_config(config_path, config)
    ids = {name: args.prefix + '-' + name for name in ('mae-pilot', 'mae', 'top3-pilot', 'top3')}
    started = time.monotonic()
    state = {'status': 'running', 'created_at': b.utc_now(), 'weight_sha256': weight_sha256,
             'expected_weight_sha256': expected_sha256, 'candidate_model': expected_model,
             'stage_model_identities': {}, 'runtime': str(runtime), 'base_config_sha256': sha256(args.base_config),
             'run_ids': ids, 'script_sha256': sha256(Path(__file__)), 'repo': b.repo_identity()}
    b.atomic_write_json(output / 'evaluation.json', state)
    def stage(name, fn):
        print(name, b.utc_now(), flush=True)
        state['stage'] = name
        b.atomic_write_json(output / 'evaluation.json', state)
        with (output / (name + '.log')).open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            evidence = state['stage_model_identities'][name] = {}
            evidence['before'] = verify_physical_model(config, expected_model)
            try:
                result = fn()
            finally:
                evidence['after'] = verify_physical_model(config, expected_model)
            evidence['recorded'] = verify_stage_model(runtime / 'runs' / ids[name], name, expected_model)
            return result
    try:
        stage('mae-pilot', lambda: b.execute_run(runtime, config, 'pilot', run_id=ids['mae-pilot']))
        pilot_dir = runtime / 'runs' / ids['mae-pilot']
        pilot_report = br.report_from_run(pilot_dir, output / 'pilot-report', config=config)
        if not pilot_report['validity']['complete_evidence_valid']:
            raise ValueError('incomplete/invalid MAE pilot')
        for engine in pilot_report['repeatability']['engines'].values():
            if engine['unstable_position_count'] or engine['stable_position_count'] != 17:
                raise ValueError('MAE pilot is not stable at all 17 positions')
        manifest, _, _, records, _, _ = b.validate_run_artifacts(pilot_dir, config, require_complete=True)
        verify_stage_model(pilot_dir, 'mae-pilot', expected_model)
        config['formal']['pilot_evidence'] = {
            'pilot_run_id': ids['mae-pilot'], 'pilot_fingerprint': manifest['fingerprint'],
            'observed_max_reported_nodes': b.observed_max_reported_nodes(records, strict=True),
            'max_reported_nodes': b.node_reporting_limit(config['requested_nodes'])}
        write_config(config_path, config)
        b.validate_formal_gate(runtime, config)
        stage('mae', lambda: b.execute_run(runtime, config, 'formal', run_id=ids['mae']))
        mae_dir = runtime / 'runs' / ids['mae']
        report = br.report_from_run(mae_dir, output / 'mae-report', config=config)
        if not report['validity']['formal_run_valid'] or not report['headline']['valid']:
            raise ValueError('formal MAE undefined or invalid')
        br.export_public(output / 'public-mae', report)
        common = dict(runtime=runtime, mae_config=config_path, mae_run=mae_dir, resume=False)
        stage('top3-pilot', lambda: top3.run(SimpleNamespace(**common, action='pilot', run_id=ids['top3-pilot'], pilot_run_id=None)))
        stage('top3', lambda: top3.run(SimpleNamespace(**common, action='formal', run_id=ids['top3'], pilot_run_id=ids['top3-pilot'])))
        top = strict_json(runtime / 'runs' / ids['top3'] / 'top3-report.json')
        if not top['valid']:
            raise ValueError('Top3 undefined or invalid')
        verify_physical_model(config, expected_model)
        for name in ids:
            verify_stage_model(runtime / 'runs' / ids[name], name, expected_model)
        state.update(status='complete', mae_cp=report['headline']['mae_cp'], top3_rate=top['top3_rate'],
                     per_game_mae={k:v['mae_cp'] for k,v in report['per_game'].items()}, top3_report=top)
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        state.update(finished_at=b.utc_now(), wall_seconds=time.monotonic()-started)
        b.atomic_write_json(output / 'evaluation.json', state)
        print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=Path, required=True)
    parser.add_argument('--expected-weight-sha256')
    parser.add_argument('--runtime', type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument('--base-config', type=Path, default=b.CONFIG_PATH)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    evaluate(parser.parse_args())


if __name__ == '__main__':
    main()
