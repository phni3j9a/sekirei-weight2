"""Evaluate a candidate with its own fixed development MAE and Top3 evidence.

Run with the pinned audit venv. Outputs must be new private directories outside
the checkout. This measurement command does not itself adopt a model.
"""
import argparse
import contextlib
import json
from pathlib import Path
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


def write_config(path, config):
    # Engine and setoption order are part of the fixed execution contract.
    b.atomic_write(path, json.dumps(config, indent=2, ensure_ascii=False) + '\n')
    loaded = b.load_config(path)
    if loaded != config or list(loaded['engines']) != list(config['engines']):
        raise ValueError('config round trip changed semantics or engine order')
    for engine in config['engines']:
        if list(loaded['engines'][engine]['options'].items()) != list(config['engines'][engine]['options'].items()):
            raise ValueError('config round trip changed setoption order')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    b._validate_run_id(args.prefix)
    output = args.output.resolve()
    weight = args.weight.resolve(strict=True)
    if output.exists() or output.is_relative_to(REPO):
        raise ValueError('use a new private output outside the repository')
    output.mkdir(parents=True)
    shutil.copyfile(__file__, output / 'evaluation-script.py')
    config = b.load_config()
    config['candidate_model'] = {'kind': 'nnue', 'path': str(weight)}
    config['formal']['pilot_evidence'] = None
    config_path = output / 'candidate.json'
    write_config(config_path, config)
    ids = {name: args.prefix + '-' + name for name in ('mae-pilot', 'mae', 'top3-pilot', 'top3')}
    started = time.monotonic()
    state = {'status': 'running', 'created_at': b.utc_now(), 'weight_sha256': sha256(weight),
             'run_ids': ids, 'script_sha256': sha256(Path(__file__)), 'repo': b.repo_identity()}
    b.atomic_write_json(output / 'evaluation.json', state)
    def stage(name, fn):
        print(name, b.utc_now(), flush=True)
        state['stage'] = name
        b.atomic_write_json(output / 'evaluation.json', state)
        with (output / (name + '.log')).open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            return fn()
    try:
        stage('mae-pilot', lambda: b.execute_run(DEFAULT_RUNTIME, config, 'pilot', run_id=ids['mae-pilot']))
        pilot_dir = DEFAULT_RUNTIME / 'runs' / ids['mae-pilot']
        pilot_report = br.report_from_run(pilot_dir, output / 'pilot-report', config=config)
        if not pilot_report['validity']['complete_evidence_valid']:
            raise ValueError('incomplete/invalid MAE pilot')
        for engine in pilot_report['repeatability']['engines'].values():
            if engine['unstable_position_count'] or engine['stable_position_count'] != 17:
                raise ValueError('MAE pilot is not stable at all 17 positions')
        manifest, _, _, records, _, _ = b.validate_run_artifacts(pilot_dir, config, require_complete=True)
        config['formal']['pilot_evidence'] = {
            'pilot_run_id': ids['mae-pilot'], 'pilot_fingerprint': manifest['fingerprint'],
            'observed_max_reported_nodes': b.observed_max_reported_nodes(records, strict=True),
            'max_reported_nodes': b.node_reporting_limit(config['requested_nodes'])}
        write_config(config_path, config)
        b.validate_formal_gate(DEFAULT_RUNTIME, config)
        stage('mae', lambda: b.execute_run(DEFAULT_RUNTIME, config, 'formal', run_id=ids['mae']))
        mae_dir = DEFAULT_RUNTIME / 'runs' / ids['mae']
        report = br.report_from_run(mae_dir, output / 'mae-report', config=config)
        if not report['validity']['formal_run_valid'] or not report['headline']['valid']:
            raise ValueError('formal MAE undefined or invalid')
        br.export_public(output / 'public-mae', report)
        common = dict(runtime=DEFAULT_RUNTIME, mae_config=config_path, mae_run=mae_dir, resume=False)
        stage('top3-pilot', lambda: top3.run(SimpleNamespace(**common, action='pilot', run_id=ids['top3-pilot'], pilot_run_id=None)))
        stage('top3', lambda: top3.run(SimpleNamespace(**common, action='formal', run_id=ids['top3'], pilot_run_id=ids['top3-pilot'])))
        top = json.loads((DEFAULT_RUNTIME / 'runs' / ids['top3'] / 'top3-report.json').read_text())
        if not top['valid']:
            raise ValueError('Top3 undefined or invalid')
        state.update(status='complete', mae_cp=report['headline']['mae_cp'], top3_rate=top['top3_rate'],
                     per_game_mae={k:v['mae_cp'] for k,v in report['per_game'].items()},
                     top3_report=top)
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        state.update(finished_at=b.utc_now(), wall_seconds=time.monotonic()-started)
        b.atomic_write_json(output / 'evaluation.json', state)
        print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
