"""Disabled SOURCE implementation of real explicit-runtime measurement ports.

All reads/writes/legacy engine calls are behind PROTOTYPE_ONLY. No production
path is opened on import. Root freezes an enabled version and external JSON
preflight only after actual build/fit/new model proof contracts are complete.
"""
from contextlib import ExitStack, contextmanager, redirect_stdout, redirect_stderr
import copy
import importlib.util
import os
from pathlib import Path
import sys
import white_view_evaluation as w
import white_view_runtime_io as io

PROTOTYPE_ONLY = True
R = Path('/home/server/worktrees/sekirei-weight2/issue-17-autonomous-weight-improvement')
C = Path('/home/server/.local/share/sekirei-weight2/campaign-17-autonomous-v1')
N = Path(w.RUNTIME)
B = C.parent/'suisho11beta-sekirei-v0.3.39-v1'
T = C.parent/'training-15-v1'
F = C.parent/'training-17-v1'
Q = F/'white-view-paired-linear-constrained-ridge1-v1/run-v1'
OLD_FAN = F/'bounded-material-fanin509-trainer-v1'
LOCKS = ((T/'.training.lock', True), (B/'.prepare.lock', False),
         (Q.parent/'.fit.lock', True), (OLD_FAN/'.build.lock', True),
         (OLD_FAN/'.training.lock', True), (N/'.build.lock', False), (N/'.prepare.lock', False))
PUBLIC_NAMES = ('prepare', 'benchmark', 'benchmark_report', 'top3', 'compare_candidates', 'audit_pack')


def barrier():
    w.require(not PROTOTYPE_ONLY and not w.PROTOTYPE_ONLY and not io.PROTOTYPE_ONLY,
              'SOURCE ONLY actual ports disabled before read/write/engine/lock')


def load_source(reader, path, expected, name):
    """Execute the exact pinned bytes, never a second unverified path read."""
    raw = reader.read(path, expected)
    if name in sys.modules:
        module = sys.modules[name]
        w.require(Path(module.__file__).resolve(strict=True) == Path(path)
                  and w.exact(io.info(path), expected)
                  and (not name.startswith('_white_view_')
                       or w.exact(getattr(module, '_white_source_identity', None), expected)),
                  'preimported module from another source/bytes')
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    w.require(spec is not None and spec.loader is not None, 'canonical source loader required')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), 'exec'), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None); raise
    w.require(w.exact(io.info(path), expected), 'module changed during import')
    module._white_source_identity = copy.deepcopy(expected)
    return module


@contextmanager
def runtime_validator(reader, binding, frozen, label):
    """Keep builder's actual adjacent contract under its exact source guard.

    Source execution alone temporarily binds the builder's import name; the
    verified worker keeps that real adjacent object in module.c afterward.
    Restore an unrelated public module even when compile/import throws.
    Root's operational contract is serial and forbids concurrent loaders.
    """
    source = binding['validator_source']['path']
    adjacent = str(Path(source).with_name('white_view_build_contract.py'))
    w.require(adjacent in frozen and source in frozen, 'both builder source files must be separately frozen')
    contract = load_source(reader, adjacent, frozen[adjacent], '_white_view_adjacent_contract_'+label)
    existed = 'white_view_build_contract' in sys.modules
    old = sys.modules.get('white_view_build_contract')
    try:
        sys.modules['white_view_build_contract'] = contract
        worker = load_source(reader, source, frozen[source], '_white_view_actual_builder_'+label)
        w.require(getattr(worker, 'c', None) is contract
            and Path(contract.__file__).resolve(strict=True) == Path(adjacent), 'actual adjacent contract object differs')
    finally:
        if existed: sys.modules['white_view_build_contract'] = old
        else: sys.modules.pop('white_view_build_contract', None)
    try: yield worker, contract
    finally:
        reader.pin(adjacent, frozen[adjacent]); reader.pin(source, frozen[source])


@contextmanager
def candidate_validator(reader, binding, source):
    """Pin the exact private proof common object during gate-worker import.

    Preserve any other cached public/common module. The producer's source and
    canonical paths are not rewritten, and the worker retains the real common
    object after its import cache name is restored. Root runs loaders serially.
    """
    common_ref = binding['common']; w.fullref(common_ref); w.fullref(source)
    expected_common = {k: common_ref[k] for k in ('bytes', 'sha256')}
    expected_source = {k: source[k] for k in ('bytes', 'sha256')}
    common = load_source(reader, common_ref['path'], expected_common, '_white_view_gate_actual_common')
    existed = 'white_view_proof_common_v2' in sys.modules
    old = sys.modules.get('white_view_proof_common_v2')
    try:
        sys.modules['white_view_proof_common_v2'] = common
        worker = load_source(reader, source['path'], expected_source, '_white_view_new03_candidate_gate')
        w.require(getattr(worker, 'p', None) is common
            and Path(common.__file__).resolve(strict=True) == Path(common_ref['path']), 'actual candidate proof common object differs')
    finally:
        if existed: sys.modules['white_view_proof_common_v2'] = old
        else: sys.modules.pop('white_view_proof_common_v2', None)
    try: yield worker
    finally:
        reader.pin(common_ref['path'], expected_common)
        reader.pin(source['path'], expected_source)


def validate_launch_preflight(value):
    require_keys = {'schema', 'status', 'spec', 'config_source', 'new_build_binding',
        'repo_identity', 'decoder_versions', 'preregistration', 'candidate_binding', 'candidate_validator'}
    w.require(type(value) is dict and set(value) == require_keys
        and value['schema'] == 'sekirei.white-view-evaluation-launch-preflight.v1'
        and value['status'] == 'frozen-before-first-write', 'new typed launch preflight required')
    s = w.validate_spec(value['spec'])
    w.fullref(value['config_source'])
    w.require(value['config_source']['path'] == str(R/'config/development-benchmark.json'), 'fixed public development config required')
    binding = value['new_build_binding']
    w.require(type(binding) is dict and set(binding) == {'runtime', 'manifest', 'identity', 'validator_source'}
        and binding['runtime'] == w.RUNTIME and w.exact(binding['manifest'], s['build_manifest'])
        and w.exact(binding['identity'], s['build_identity']), 'new build binding differs from evaluation spec')
    w.fullref(binding['validator_source'])
    for record in (value['config_source'], binding['validator_source']):
        w.require(w.exact(s['inputs'].get(record['path']), {k: record[k] for k in ('bytes', 'sha256')}), 'preflight prerequisite absent from input map')
    repo = value['repo_identity']
    w.require(type(repo) is dict and set(repo) == {'commit', 'dirty', 'status_porcelain'}
        and type(repo['commit']) is str and len(repo['commit']) == 40
        and all(x in '0123456789abcdef' for x in repo['commit'])
        and repo['dirty'] is False and repo['status_porcelain'] == '', 'frozen clean Git source identity required')
    w.require(w.exact(value['decoder_versions'], {'numpy': '1.26.4', 'cshogi': '1.0.4'}), 'fixed decoder versions required')
    for name in PUBLIC_NAMES:
        path = str(R/'scripts'/(name+'.py'))
        w.require(path in s['source_helpers'] and path in s['inputs'], 'fixed legacy public helper missing')
    if s['role'] == 'fallback':
        w.require(all(value[k] is None for k in ('preregistration', 'candidate_binding', 'candidate_validator')), 'fallback cannot depend on fitted model or unverified gate')
    else:
        # New proof producer source is externally frozen by Root after schema
        # review. Absence/pending source never falls back to old Adam/E3 guard.
        w.fullref(value['preregistration']); w.fullref(value['candidate_validator'])
        w.require(value['candidate_validator']['path'] == str(C/'white-view-candidate-model-gate-worker-v2.py'), 'dedicated new03 proof consumer required')
        for record in (value['preregistration'], value['candidate_validator']):
            w.require(w.exact(s['inputs'].get(record['path']), {k: record[k] for k in ('bytes', 'sha256')}), 'new model gate source/ref not frozen')
        w.validate_candidate_binding(value['candidate_binding'], s['inputs'])
        w.require(w.exact(value['candidate_binding']['preregistration'], value['preregistration'])
            and w.exact(value['candidate_binding']['gate_worker'], value['candidate_validator'])
            and w.exact(value['candidate_binding']['gate'], s['candidate_gate']), 'candidate gate/control binding differs')
    return value


class ProductionPorts:
    def __init__(self, preflight_ref, inherited_locks=None):
        barrier()
        self.reader = io.Reader()
        self.pf_ref = copy.deepcopy(w.fullref(preflight_ref))
        self.pf = validate_launch_preflight(self.reader.json_ref(preflight_ref))
        self.spec = self.pf['spec']
        self.writer = io.OwnedOutput(self.spec['evaluation_path'])
        self.modules, self.stage_evidence = {}, {}
        self.locked = False
        self.inherited_locks = inherited_locks
        self.state = {'schema': 'sekirei.white-view-evaluation-progress.v1', 'status': 'running',
            'role': self.spec['role'], 'candidate': w.MODE, 'run_ids': self.spec['run_ids'],
            'candidate_model': self.spec['model'], 'runtime': w.RUNTIME, 'final_used': False,
            'adoption_verified': False}

    @contextmanager
    def coordinating_locks(self, spec):
        barrier(); w.validate_spec(spec)
        w.require(w.exact(spec, self.spec), 'caller spec differs from pinned parsed preflight')
        with ExitStack() as stack:
            if self.inherited_locks is None:
                for path, exclusive in LOCKS:
                    stack.enter_context(io.existing_lock(path, exclusive))
            else:
                validate_inherited_locks(self.inherited_locks)
            self.locked = True
            try: yield
            finally: self.locked = False

    def verify_sources(self):
        for path, expected in self.spec['inputs'].items(): self.reader.pin(path, expected)
        w.require(sys.executable == w.AUD and sys.prefix == str(Path(w.AUD).parent.parent), 'invoke exact audit venv before imports/engine')
        resolved = str(Path(sys.executable).resolve(strict=True))
        w.require(resolved in self.spec['inputs'], 'resolved Python executable not preflight-bound')
        for path, expected_sha in self.spec['source_helpers'].items():
            w.require(self.reader.files[path]['sha256'] == expected_sha, 'all source helper bytes must be externally frozen')
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(R/'scripts'))
        for name in PUBLIC_NAMES:
            path = str(R/'scripts'/(name+'.py'))
            self.modules[name] = load_source(self.reader, path, self.spec['inputs'][path], name)
        for name, module in list(sys.modules.items()):
            path = getattr(module, '__file__', None)
            if path and Path(path).is_relative_to(R/'scripts'):
                path = str(Path(path).resolve(strict=True))
                w.require(path in self.spec['source_helpers'] and path in self.reader.files
                    and io.info(path)['sha256'] == self.spec['source_helpers'][path], 'transitive local helper import not frozen')
        w.require(w.exact(self.modules['benchmark'].repo_identity(), self.pf['repo_identity']), 'public source Git state changed')
        _, _, versions = self.modules['audit_pack'].require_decoder()
        w.require(w.exact(versions, self.pf['decoder_versions']), 'actual decoder dependency versions differ')
        return True

    def verify_new_runtime(self, config):
        binding = self.pf['new_build_binding']
        with runtime_validator(self.reader, binding, self.spec['inputs'], 'measurement') as (validator, contract):
            w.require(callable(getattr(validator, 'verify_runtime', None)), 'new runtime verifier API missing')
            result = validator.verify_runtime(N, binding['manifest']['sha256'], binding['identity']['sha256'])
            actual_inventory = contract.immutable_inputs_for_runtime(result, binding['manifest'])
            for path, rec in actual_inventory.items():
                w.require(w.exact(self.spec['inputs'].get(path), rec), 'complete actual builder inventory absent from external launch map')
                self.reader.pin(path, rec)
        w.require(type(result) is dict and set(result) == {'manifest', 'identity'}, 'new verified runtime bundle required')
        for key in ('manifest', 'identity'):
            w.require(w.exact(self.reader.json_ref(binding[key]), result[key]), 'verified runtime differs from externally pinned raw bytes')
        w.validate_project_cargo_dependencies(result['manifest'], result['identity'])
        # The dedicated manifest validator owns all compiler/source/current
        # checks. Old runtime verification is retained as an additional gate.
        self.modules['benchmark'].validate_runtime(N, config)
        return result

    def verify_candidate(self):
        binding = self.pf['candidate_binding']
        source = self.pf['candidate_validator']
        w.validate_candidate_binding(binding, self.spec['inputs'])
        with candidate_validator(self.reader, binding, source) as module:
            w.require(callable(getattr(module, 'verify_candidate_inputs', None)), 'dedicated proof consumer missing; actual candidate remains blocked')
            gate = module.verify_candidate_inputs(self.reader, binding, self.spec)
        expected = self.reader.json_ref(self.spec['candidate_gate'])
        w.require(w.exact(gate, expected), 'candidate gate bytes/result differ')
        w.validate_native_gate(gate, self.spec['model'], self.pf['new_build_binding'], binding, self.spec['inputs'])
        for record in gate['proofs'].values():
            self.reader.pin(record['path'], {k: record[k] for k in ('bytes', 'sha256')})
        bridge = read_verified_bridge(self.reader, self.spec['fallback_bridge'])
        w.require(w.exact(bridge['new_build'], self.spec['build_manifest']), 'bridge new build differs')
        return True

    def preflight(self, spec, config):
        barrier(); w.require(self.locked, 'all coordinating locks required before reads/write')
        # Locked reparse ties parsed bytes, expected SHA and current file.
        w.require(w.exact(self.reader.json_ref(self.pf_ref), self.pf), 'launch preflight changed before locks')
        self.verify_sources(); self.verify_new_runtime(config)
        base = self.modules['benchmark'].load_config(self.pf['config_source']['path'])
        w.require(w.exact(w.config_for_role(base, spec), config), 'planned config differs from fixed default except model/own pilot')
        if spec['role'] == 'candidate': self.verify_candidate()
        for path, rec in self.reader.files.items():
            if path != self.pf_ref['path']:
                w.require(w.exact(spec['inputs'].get(path), rec), 'all transitive build/model/proof inputs must be launch-bound')
        protected = [str(R), '/home/server/projects/sekirei-weight2', *spec['inputs']]
        io.private_output(spec['evaluation_path'], protected)
        w.require(all(not os.path.lexists(N/'runs'/run_id) for run_id in spec['run_ids'].values()), 'all four run IDs must be fresh')
        w.require(w.exact(self.reader.current(), io.union_exact(spec['inputs'], {
            self.pf_ref['path']: {k: self.pf_ref[k] for k in ('bytes', 'sha256')}})), 'preflight input union differs')
        return {'schema': 'sekirei.white-view-evaluation-preflight-envelope.v1', 'status': 'verified-before-first-write',
            'role': spec['role'], 'runtime': w.RUNTIME, 'interpreter': w.AUD, 'runtime_verified': True,
            'original_input_verified': True, 'source_and_build_unchanged': True,
            'externally_frozen_bindings_verified': True, 'output_and_four_run_paths_absent': True,
            'inputs': copy.deepcopy(spec['inputs']), 'build_manifest': spec['build_manifest'],
            'build_identity': spec['build_identity'], 'candidate_gate': spec['candidate_gate'],
            'fallback_bridge': spec['fallback_bridge'], 'native_gate_verified': spec['role'] == 'candidate',
            'fallback_bridge_verified': spec['role'] == 'candidate', 'final_used': False, 'adoption_verified': False}

    def new_output(self, spec, config):
        w.require(self.locked, 'lock release before first write')
        self.writer.create(); self.write_config(spec, config)
        self.writer.write('evaluation.json', self.state, mutable=True)
        for path in (Path(w.__file__).resolve(), Path(__file__).resolve()):
            w.require(str(path) in spec['inputs'], 'evaluation source provenance not frozen')
            io.write_new(self.writer.root/(path.name+'.source-snapshot'), self.reader.read(path, spec['inputs'][str(path)]))

    def write_config(self, spec, config):
        self.writer.write('candidate.json', config, mutable=True)
        loaded = self.modules['benchmark'].load_config(self.writer.root/'candidate.json')
        w.require(w.exact(loaded, config) and list(loaded['engines']) == list(config['engines']), 'config serialization changed typed semantics/order')
        for engine in config['engines']:
            w.require(list(loaded['engines'][engine]['options'].items()) == list(config['engines'][engine]['options'].items()), 'setoption order changed')

    def run_stage(self, stage, spec, config):
        self.state.update(stage=stage)
        self.writer.write('evaluation.json', self.state, mutable=True)
        try:
            with (self.writer.root/(stage+'.log')).open('x') as log, redirect_stdout(log), redirect_stderr(log):
                w.legacy_stage_call(stage, spec, config, self.writer.root/'candidate.json', self.modules['benchmark'], self.modules['top3'])
        except BaseException as error:
            self.state.update(status='failed', error_type=type(error).__name__, error=str(error),
                              complete=False, executor_reaped=False, cleanup_verified=False)
            self.writer.write('evaluation.json', self.state, mutable=True)
            raise

    def audit_stage(self, stage, spec, config):
        return reparse_stage(stage, N, spec['run_ids'], self.writer.root/'candidate.json', config, self.modules)

    def current_inputs(self, spec):
        self.verify_new_runtime(self.modules['benchmark'].load_config(self.writer.root/'candidate.json'))
        w.require(w.exact(self.modules['benchmark'].repo_identity(), self.pf['repo_identity']), 'public source changed during four stages')
        current = self.reader.current()
        w.require(all(w.exact(current.get(p), rec) for p, rec in spec['inputs'].items()), 'original source/build/model inputs changed')
        return copy.deepcopy(spec['inputs'])


def reparse_stage(stage, runtime, ids, config_path, config, modules):
    """All attempt raw SHA/options/node/lifecycle are legacy parser gates."""
    barrier()
    b, br, top3 = (modules[k] for k in ('benchmark', 'benchmark_report', 'top3'))
    directory = Path(runtime)/'runs'/ids[stage]
    expected_count = w.STAGES[stage]
    if stage in ('mae-pilot', 'mae'):
        manifest, _, _, rows, _, _ = b.validate_run_artifacts(directory, config,
            require_complete=True, expected_run_type='pilot' if stage == 'mae-pilot' else 'formal')
        w.require(len(rows) == expected_count and all(row['status'] not in b.TECHNICAL_FAILURE_STATUSES for row in rows), 'MAE raw coverage/technical outcome invalid')
        report = br.report_from_run(directory, config=config)
        if stage == 'mae-pilot':
            w.require(report['validity']['complete_evidence_valid'] is True, 'MAE pilot validity failed')
            stable = {k: v['stable_position_count'] for k, v in report['repeatability']['engines'].items()}
            unstable = {k: v['unstable_position_count'] for k, v in report['repeatability']['engines'].items()}
        else:
            w.require(report['validity']['formal_run_valid'] is True and report['headline']['valid'] is True
                and report['universe']['total_occurrences'] == 570
                and report['e_exact_coverage']['teacher_e_count'] == 266
                and report['e_exact_coverage']['candidate_exact_count'] == 266, 'formal MAE coverage invalid')
        maximum = b.observed_max_reported_nodes(rows, strict=True)
        fingerprint = manifest['fingerprint']
    else:
        ctx = top3.context(Path(runtime), config, Path(runtime)/'runs'/ids['mae'])
        report = top3.inspect(directory, *ctx, config)
        w.require(report['valid'] is True and report['attempts'] == expected_count
            and report['eligible_positions'] == (12 if stage == 'top3-pilot' else 551), 'Top3 raw/ranks/stability/coverage invalid')
        maximum = 0
        for path in (directory/'logs').glob('*.jsonl'):
            nodes = b._node_evidence_summary(b._read_raw_event_log(path))
            w.require(not nodes['errors'] and not nodes['score_errors'] and nodes['valid_values'], 'Top3 node evidence invalid')
            maximum = max(maximum, max(nodes['valid_values']))
        w.require(len(list((directory/'logs').glob('*.jsonl'))) == expected_count, 'Top3 raw file set differs')
        fingerprint = report['fingerprint']
        stable, unstable = 12, 0
    evidence = {'stage': stage, 'attempts': expected_count, 'raw_sha_verified': expected_count,
        'lifecycle_verified': expected_count, 'cleanup_ok': expected_count, 'runner_exit_zero': expected_count,
        'technical_failures': 0, 'timeout_failures': 0, 'complete': True, 'final_used': False,
        'max_reported_nodes': maximum, 'fingerprint': fingerprint}
    if stage in ('mae-pilot', 'top3-pilot'):
        evidence.update(stable_positions=stable, unstable_positions=unstable)
    return w.validate_stage(stage, evidence)


def produce_raw_bundle(descriptor, config, role, model, runtime, modules, build_ref, terminal_ref,
                       source_inputs, stage_ids):
    """Root calls after externally observed normal terminal; no engine start."""
    barrier(); w.fullref(build_ref); w.fullref(terminal_ref)
    reader = io.Reader()
    terminal = reader.json_ref(terminal_ref)
    validate_audit_terminal(terminal, role, model, stage_ids)
    b, br, top3, cc = (modules[k] for k in ('benchmark', 'benchmark_report', 'top3', 'compare_candidates'))
    directories = [Path(runtime)/'runs'/stage_ids[s] for s in w.STAGES]
    paths = [descriptor['config'], build_ref['path'], terminal_ref['path'], *source_inputs]
    if descriptor.get('evaluation'): paths.append(descriptor['evaluation'])
    before = io.inventory(directories, paths)
    for path, expected in source_inputs.items():
        w.require(w.exact(before['files'].get(path), expected), 'audit external input pin differs')
    parsed = w.legacy_reparse_descriptor(descriptor, config, b, br, top3, cc)
    stages = {name: reparse_stage(name, runtime, stage_ids, descriptor['config'], config, modules) for name in w.STAGES}
    after = io.inventory(directories, paths)
    w.require(w.exact(before, after), 'raw/config/source/model/terminal inventory changed during strict reparse')
    bundle = {'schema': 'sekirei.white-view-reparsed-role-bundle.v1', 'status': 'complete', 'role': role,
        'runtime': str(runtime), 'model': copy.deepcopy(model), 'raw_reparsed': True, 'inputs_unchanged': True,
        'cleanup_verified': True, 'executor_reaped': True, 'final_used': False, 'adoption_verified': False,
        'stages': stages, 'comparable': parsed['comparable'], 'per_game': parsed['per_game'],
        'run_ids': copy.deepcopy(stage_ids),
        'semantic_projections': parsed['semantic_projections'], 'projection_policy': parsed['projection_policy'],
        'build_manifest': copy.deepcopy(build_ref),
        'terminal_ref': copy.deepcopy(terminal_ref), 'audit_inputs_before': before['files'],
        'audit_inputs_after': after['files'], 'directory_inventory': before['directories']}
    # No own receipt SHA in payload. The small receipt points one way to this
    # separately written payload; consumers attach its external ref in memory.
    return w.validate_raw_bundle(bundle, role, model, str(runtime))


def validate_inherited_locks(records):
    """Parent passes its flock-open descriptions; child never drops them."""
    import fcntl
    w.require(type(records) is list and len(records) == len(LOCKS), 'all inherited coordinating locks required')
    for row, (path, exclusive) in zip(records, LOCKS):
        w.require(type(row) is dict and set(row) == {'path', 'exclusive', 'fd'}
            and row['path'] == str(path) and row['exclusive'] is exclusive
            and type(row['fd']) is int and row['fd'] >= 3, 'fixed parent lock descriptor required')
        w.require(path.resolve(strict=True) == path and not path.is_symlink(), 'lock path canonical')
        actual, expected = os.fstat(row['fd']), path.stat()
        w.require((actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino), 'inherited lock file differs')
        fcntl.flock(row['fd'], (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)|fcntl.LOCK_NB)
    w.require(len({r['fd'] for r in records}) == len(records), 'lock descriptors must be distinct')


def validate_audit_terminal(value, role, model, ids):
    fixed = {'schema': 'sekirei.white-view-evaluation-terminal.v1', 'status': 'observed-stopped',
        'exit_code': 0, 'session_closed': True, 'tool_observed_reaped': True,
        'all_related_groups_observed_stopped': True, 'role': role, 'model': model, 'run_ids': ids}
    w.require(type(value) is dict and all(w.exact(value.get(k), v) for k, v in fixed.items()),
        'normal external tool-reaped terminal required before full role audit')
    w.require(type(value.get('executor_session_id')) is int and value['executor_session_id'] > 0,
        'actual normal executor session required')
    return value


def write_audit_receipt(root, payload):
    """Source writer: immutable payload then one-way complete envelope."""
    barrier(); root = Path(root)
    w.require(not os.path.lexists(root), 'fresh audit root required')
    io.private_output(root, [*payload['audit_inputs_before']])
    root.mkdir(mode=0o700)
    payload_ref = io.write_new(root/'reparsed-role-payload.json', io.json_bytes(payload))
    receipt = {'schema': 'sekirei.white-view-full-role-audit.v1', 'status': 'complete',
        'role': payload['role'], 'runtime': payload['runtime'], 'model': payload['model'],
        'payload': payload_ref, 'build_manifest': payload['build_manifest'], 'terminal_ref': payload['terminal_ref'],
        'attempts': 1829, 'raw_reparsed': True, 'inputs_unchanged': True,
        'cleanup_verified': True, 'executor_reaped': True, 'final_used': False, 'adoption_verified': False}
    ref = io.write_new(root/'receipt.json', io.json_bytes(receipt))
    return ref


def validate_audit_request(value):
    """External source-only contract for either old fallback or new role."""
    keys = {'schema', 'status', 'role', 'model', 'runtime', 'descriptor', 'run_ids', 'config',
        'build_manifest', 'terminal', 'source_inputs', 'source_helpers', 'repo_identity', 'decoder_versions',
        'new_build_binding', 'output'}
    w.require(type(value) is dict and set(value) == keys
        and value['schema'] == 'sekirei.white-view-full-role-audit-request.v1'
        and value['status'] == 'frozen-before-raw-reparse', 'typed external full raw audit request required')
    w.model_identity(value['role'], value['model']); w.absolute(value['runtime']); w.absolute(value['output'])
    w.require(value['runtime'] in (str(B), str(N)), 'only fixed old-stock or new-white-view runtime allowed')
    if value['runtime'] == str(B):
        w.require(value['role'] == 'fallback' and value['new_build_binding'] is None, 'old stock can only provide frozen fallback bridge subject')
    else:
        w.require(type(value['new_build_binding']) is dict, 'new runtime requires dedicated verifier binding')
        for key in ('manifest', 'identity', 'validator_source'): w.fullref(value['new_build_binding'][key])
        w.require(w.exact(value['new_build_binding']['manifest'], value['build_manifest'])
            and value['new_build_binding']['runtime'] == str(N), 'new audit build binding differs')
    for key in ('config', 'build_manifest', 'terminal'): w.fullref(value[key])
    w.require(value['config']['path'] == value['descriptor']['config']
        and value['descriptor']['runtime'] == value['runtime'], 'audit descriptor subject differs')
    w.require(type(value['run_ids']) is dict and set(value['run_ids']) == set(w.STAGES), 'all four audit run IDs required')
    w.require(value['descriptor']['mae'] == str(Path(value['runtime'])/'runs'/value['run_ids']['mae'])
        and value['descriptor']['top3'] == str(Path(value['runtime'])/'runs'/value['run_ids']['top3']), 'formal audit paths differ')
    w.require(type(value['source_inputs']) is dict and value['source_inputs'], 'full external input map required')
    for path, rec in value['source_inputs'].items(): w.absolute(path); w.file_identity(rec)
    for ref in (value['config'], value['build_manifest'], value['terminal']):
        w.require(w.exact(value['source_inputs'].get(ref['path']), {k: ref[k] for k in ('bytes', 'sha256')}), 'audit primary input not externally pinned')
    if value['new_build_binding'] is not None:
        for key in ('manifest', 'identity', 'validator_source'):
            ref = value['new_build_binding'][key]
            w.require(w.exact(value['source_inputs'].get(ref['path']), {k: ref[k] for k in ('bytes', 'sha256')}), 'new raw-audit build fullref not externally pinned')
    w.require(type(value['source_helpers']) is dict and value['source_helpers'], 'all audit parser sources required')
    for path, sha in value['source_helpers'].items():
        w.absolute(path); w.sha(sha)
        w.require(path in value['source_inputs'] and value['source_inputs'][path]['sha256'] == sha, 'audit helper absent from actual map')
    w.require(w.exact(value['decoder_versions'], {'numpy': '1.26.4', 'cshogi': '1.0.4'}), 'fixed audit decoder required')
    return value


def audit_role(reader, request):
    """Actual reader/reparser/writer implementation; no engine execution."""
    barrier(); validate_audit_request(request)
    with ExitStack() as locks:
        for path, exclusive in LOCKS: locks.enter_context(io.existing_lock(path, exclusive))
        w.require(sys.executable == w.AUD and sys.prefix == str(Path(w.AUD).parent.parent), 'invoke actual audit venv')
        for path, rec in request['source_inputs'].items(): reader.pin(path, rec)
        sys.dont_write_bytecode = True; sys.path.insert(0, str(R/'scripts'))
        modules = {}
        for name in PUBLIC_NAMES:
            path = str(R/'scripts'/(name+'.py'))
            w.require(path in request['source_helpers'], 'complete strict parser inventory missing')
            modules[name] = load_source(reader, path, request['source_inputs'][path], name)
        b = modules['benchmark']; config = b.load_config(request['config']['path'])
        if request['runtime'] == str(N):
            state_path = request['descriptor'].get('evaluation')
            w.require(type(state_path) is str and state_path in request['source_inputs'], 'complete new typed state must be externally pinned')
            state_ref = {'path': state_path, **request['source_inputs'][state_path]}
            state = reader.json_ref(state_ref)
            actual_descriptor = w.descriptor(state, str(Path(state_path).parent), request['config']['path'], request['role'], request['model'])
            w.require(w.exact(actual_descriptor, request['descriptor']) and w.exact(state['run_ids'], request['run_ids'])
                and w.exact(state['terminal'], request['terminal']), 'complete new state/config/run/terminal subject differs')
        w.require(w.exact(b.repo_identity(), request['repo_identity']), 'audit Git source identity changed')
        _, _, versions = modules['audit_pack'].require_decoder()
        w.require(w.exact(versions, request['decoder_versions']), 'actual audit decoder versions differ')
        for module in list(sys.modules.values()):
            path = getattr(module, '__file__', None)
            if path and Path(path).is_relative_to(R/'scripts'):
                canonical = str(Path(path).resolve(strict=True))
                w.require(canonical in request['source_helpers']
                    and io.info(canonical)['sha256'] == request['source_helpers'][canonical], 'transitive raw parser not frozen')
        expected_model = request['model']
        if request['role'] == 'fallback':
            w.require(w.exact(config.get('candidate_model', {'kind': 'material_fallback', 'path': None}),
                {'kind': 'material_fallback', 'path': None}), 'fallback raw config must remain unloaded')
        else:
            w.require(w.exact(config['candidate_model'], {'kind': 'nnue', 'path': expected_model['path']}), 'candidate raw config model differs')
        w.require(w.exact(b._model_identity(config), expected_model), 'actual candidate bytes/fallback identity differs')
        b.validate_runtime(Path(request['runtime']), config)
        binding = request['new_build_binding']
        if binding is not None:
            with runtime_validator(reader, binding, request['source_inputs'], 'audit') as (validator, contract):
                verified = validator.verify_runtime(N, binding['manifest']['sha256'], binding['identity']['sha256'])
                for path, rec in contract.immutable_inputs_for_runtime(verified, binding['manifest']).items():
                    w.require(w.exact(request['source_inputs'].get(path), rec), 'complete new build inventory missing from raw audit input map')
                    reader.pin(path, rec)
            w.require(w.exact(verified['manifest'], reader.json_ref(binding['manifest']))
                and w.exact(verified['identity'], reader.json_ref(binding['identity'])), 'actual new raw-audit runtime differs')
            w.validate_project_cargo_dependencies(verified['manifest'], verified['identity'])
        payload = produce_raw_bundle(request['descriptor'], config, request['role'], request['model'],
            request['runtime'], modules, request['build_manifest'], request['terminal'],
            io.union_exact(request['source_inputs'], reader.files), request['run_ids'])
        b.validate_runtime(Path(request['runtime']), config)
        if binding is not None:
            w.require(w.exact(validator.verify_runtime(N, binding['manifest']['sha256'], binding['identity']['sha256']), verified), 'new build changed during raw reparse')
        reader.current()
        return write_audit_receipt(request['output'], payload)


def read_audit_bundle(reader, reference):
    barrier()
    receipt = reader.json_ref(reference)
    fixed = {'schema': 'sekirei.white-view-full-role-audit.v1', 'status': 'complete',
        'attempts': 1829, 'raw_reparsed': True, 'inputs_unchanged': True, 'cleanup_verified': True,
        'executor_reaped': True, 'final_used': False, 'adoption_verified': False}
    w.require(type(receipt) is dict and all(w.exact(receipt.get(k), v) for k, v in fixed.items()), 'complete one-way role audit receipt required')
    payload = reader.json_ref(receipt['payload'])
    w.require('audit_ref' not in payload, 'payload must not contain own receipt hash')
    for key in ('role', 'runtime', 'model', 'build_manifest', 'terminal_ref'):
        w.require(w.exact(payload[key], receipt[key]), 'audit envelope subject differs')
    w.validate_raw_bundle(payload, receipt['role'], receipt['model'], receipt['runtime'])
    w.require(type(payload['audit_inputs_before']) is dict and w.exact(payload['audit_inputs_before'], payload['audit_inputs_after']), 'full raw audit inputs changed')
    for path, rec in payload['audit_inputs_after'].items(): reader.pin(path, rec)
    current = io.inventory(list(payload['directory_inventory']), list(payload['audit_inputs_after']))
    w.require(w.exact(current['files'], payload['audit_inputs_after'])
        and w.exact(current['directories'], payload['directory_inventory']), 'raw audit file membership/bytes changed')
    terminal = reader.json_ref(payload['terminal_ref'])
    w.require(type(payload.get('run_ids')) is dict and set(payload['run_ids']) == set(w.STAGES), 'explicit four stage IDs required')
    w.require(all(str(Path(payload['runtime'])/'runs'/run_id) in payload['directory_inventory']
        for run_id in payload['run_ids'].values()), 'all four directories absent from raw membership map')
    validate_audit_terminal(terminal, payload['role'], payload['model'], payload['run_ids'])
    result = copy.deepcopy(payload); result['audit_ref'] = copy.deepcopy(reference)
    return result


def read_verified_bridge(reader, reference):
    barrier()
    value = reader.json_ref(reference)
    w.require(type(value) is dict, 'bridge JSON dictionary required')
    old = read_audit_bundle(reader, value['old_audit'])
    fresh = read_audit_bundle(reader, value['new_audit'])
    pins = {'old_audit': value['old_audit'], 'new_audit': value['new_audit'],
        'old_runtime': old['runtime'], 'old_build': value['old_build'], 'new_build': value['new_build']}
    w.require(w.exact(w.fallback_bridge(old, fresh, pins), value), 'bridge does not reproduce all semantic/exact raw subjects')
    return value


def main():
    raise RuntimeError('SOURCE ONLY production ports entry is disabled')


if __name__ == '__main__': main()
