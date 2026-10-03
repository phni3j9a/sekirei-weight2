"""Synthetic bytes/control/semantic fixtures only; no real input or engines."""
import ast
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest import mock
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'preparations'/'white-view-evaluation-runtime-v3'))
import white_view_evaluation as w


def identity(char='a', size=4):
    return {'bytes': size, 'sha256': char*64}


def ref(path, char='a', size=4):
    return {'path': path, **identity(char, size)}


UNLOADED = {'kind': 'material_fallback', 'path': None, 'bytes': None, 'sha256': None}
NATIVE = {'kind': 'nnue', **ref('/tmp/public-native03.bin', 'b', 1305356)}


def spec(role='fallback'):
    manifest = ref(w.RUNTIME+'/build-manifest.json', 'c')
    build_id = ref('/tmp/public-build-identity.json', 'd')
    native_gate = None if role == 'fallback' else ref('/tmp/public-native-gate.json')
    bridge = None if role == 'fallback' else ref('/tmp/public-fallback-bridge.json')
    inputs = {'/tmp/public-input.json': identity('f'), '/tmp/public-helper.py': identity('e')}
    for record in (manifest, build_id):
        inputs[record['path']] = {k: record[k] for k in ('bytes', 'sha256')}
    if role == 'candidate':
        for record in (native_gate, bridge, NATIVE):
            inputs[record['path']] = {k: record[k] for k in ('bytes', 'sha256')}
    return w.make_spec(role, 'development-white-'+role+'-v1', '/tmp/public-white-'+role,
        UNLOADED if role == 'fallback' else NATIVE,
        manifest, build_id, {'/tmp/public-helper.py': 'e'*64}, inputs, native_gate, bridge)


def config():
    return {'split': 'development', 'requested_nodes': 1000000, 'universe': {'plies': dict(w.GAMES)},
        'engines': {'teacher': {'options': {'Threads': '1', 'USI_Hash': '128', 'MultiPV': '1'}},
                    'sekirei': {'options': {'Threads': '1', 'Hash': '128', 'MultiPV': '1'}}},
        'candidate_model': {'kind': 'material_fallback', 'path': None},
        'formal': {'max_reported_nodes': 1010000, 'pilot_evidence': {'old': 'not reused'}},
        'pilot': {'repetitions': 3}}


def stage(name):
    count = w.STAGES[name]
    value = {'stage': name, 'attempts': count, 'raw_sha_verified': count,
        'lifecycle_verified': count, 'cleanup_ok': count, 'runner_exit_zero': count,
        'technical_failures': 0, 'timeout_failures': 0, 'complete': True, 'final_used': False,
        'max_reported_nodes': 1000001, 'fingerprint': 'a'*64}
    if name == 'mae-pilot':
        value.update(stable_positions={'teacher': 17, 'sekirei': 17}, unstable_positions={'teacher': 0, 'sekirei': 0})
    if name == 'top3-pilot':
        value.update(stable_positions=12, unstable_positions=0)
    return value


def draft():
    s = spec()
    ports = w.SyntheticPorts(s['inputs'], {name: stage(name) for name in w.STAGES})
    return w.four_stage_call_sequence(s, config(), ports), ports


def parent_docs(d):
    termref = ref('/tmp/public-success-terminal.json', '9')
    terminal = {'schema': 'sekirei.white-view-evaluation-terminal.v1', 'status': 'observed-stopped',
        'exit_code': 0, 'session_closed': True, 'tool_observed_reaped': True,
        'all_related_groups_observed_stopped': True, 'role': d['role'], 'run_ids': d['run_ids'],
        'model': d['candidate_model'], 'executor_session_id': 123}
    op = {'schema': 'sekirei.white-view-evaluation-operational-verification.v1', 'status': 'complete',
        'role': d['role'], 'inputs_unchanged': True, 'source_and_build_unchanged': True,
        'cleanup_verified': True, 'final_used': False, 'inputs_before': d['inputs_before'],
        'inputs_after': d['inputs_after'], 'terminal': termref}
    return terminal, op, termref


def build_fixture():
    names = set(w.CHANGED)|{'Cargo.lock', 'crates/sekirei-core/src/search.rs', 'vendor/example/Cargo.toml'}
    before = {p: identity() for p in names}
    before.update({'synthetic-src/%03d.rs' % i: identity() for i in range(526-len(names))})
    after = copy.deepcopy(before)
    for p in w.CHANGED:
        after[p] = identity('b')
    db = {p: copy.deepcopy(before[p]) for p in w.CARGO|{'Cargo.lock'}}
    db['vendor/example/Cargo.toml'] = copy.deepcopy(before['vendor/example/Cargo.toml'])
    da = copy.deepcopy(db)
    for p in w.CARGO:
        da[p] = copy.deepcopy(after[p])
    return {'source_files_before': before, 'source_files_after': after,
        'dependency_files_before': db, 'dependency_files_after': da,
        'build_source_files_before': copy.deepcopy(after), 'build_source_files_after': copy.deepcopy(after),
        'build_dependency_files_before': copy.deepcopy(da), 'build_dependency_files_after': copy.deepcopy(da),
        'compiler_files': {'/tmp/public-rustc': identity('8')}, 'compiler_unchanged': True,
        'inputs_before': {'/tmp/public-rustc': identity('8')},
        'inputs_after': {'/tmp/public-rustc': identity('8')}}


def bundle(role='fallback', runtime=w.RUNTIME, error_delta=0, hits_delta=0):
    teacher = {str(i): {'side_to_move': 'b', 'score_cp_sente': i} for i in range(570)}
    comparable = {'execution_except_model': {'runtime': {'binaries': {'sekirei': runtime+'-binary'},
        'build_manifest': {'runtime': runtime}}}, 'occurrences': list(range(570)), 'teacher_results': teacher,
        'teacher_E_cp': {str(i): i for i in range(266)}, 'top3_occurrences': [str(i) for i in range(551)],
        'top3_legal_moves_sha256': 'a'*64, 'top3_denominators': {'synthetic': 551},
        'top3_teacher_bestmoves': {str(i): '1a1b' for i in range(551)}}
    ec, td = (40, 60, 60, 40, 66), (83, 122, 124, 103, 119)
    rows = {g: {'absolute_error_sum_cp': ec[i]*10+error_delta, 'e_count': ec[i],
        'top3_hits': td[i]//2+hits_delta, 'top3_denominator': td[i]} for i, g in enumerate(w.GAMES)}
    return {'schema': 'sekirei.white-view-reparsed-role-bundle.v1', 'status': 'complete', 'role': role,
        'runtime': runtime, 'model': copy.deepcopy(UNLOADED if role == 'fallback' else NATIVE),
        'raw_reparsed': True, 'inputs_unchanged': True, 'cleanup_verified': True, 'executor_reaped': True,
        'final_used': False, 'adoption_verified': False, 'stages': {name: stage(name) for name in w.STAGES},
        'comparable': comparable, 'per_game': rows,
        'semantic_projections': {'teacher570': copy.deepcopy(teacher), 'sekirei570': copy.deepcopy(teacher),
            'mae_pilot_102': {str(i): {'engine_id': 'sekirei', 'repetition': 1, 'semantic': {'score_cp_sente': i}} for i in range(102)},
            'top3_pilot_36': {str(i): {'bestmove': '1a1b', 'ranks': [1, 2, 3]} for i in range(36)},
            'top3_551': {str(i): {'bestmove': '1a1b', 'ranks': [1, 2, 3]} for i in range(551)}},
        'projection_policy': copy.deepcopy(w.PROJECTION_POLICY),
        'audit_ref': ref('/tmp/public-'+role+'-audit.json'),
        'build_manifest': ref(runtime+'/build-manifest.json', 'c')}


def bridge_fixture():
    old = bundle(runtime='/tmp/public-old-runtime')
    fresh = bundle()
    old['audit_ref'] = ref('/tmp/public-old-audit.json', '1')
    pins = {'old_audit': old['audit_ref'], 'new_audit': fresh['audit_ref'], 'old_runtime': old['runtime'],
            'old_build': old['build_manifest'], 'new_build': fresh['build_manifest']}
    return old, fresh, pins


class Tests(unittest.TestCase):
    def test_project_cargo_duplicate_source_and_compile_stage(self):
        self.assertTrue(w.validate_project_cargo_dependencies(build_fixture()))

    def test_unrelated_change_in_allowed_cargo_rejected(self):
        value = build_fixture()
        p = next(iter(w.CARGO))
        value['dependency_files_after'][p] = identity('c')
        value['build_dependency_files_before'][p] = identity('c')
        value['build_dependency_files_after'][p] = identity('c')
        with self.assertRaises(ValueError): w.validate_project_cargo_dependencies(value)

    def test_cargo_before_duplicate_disagreement_rejected(self):
        value = build_fixture(); value['dependency_files_before'][next(iter(w.CARGO))] = identity('c')
        with self.assertRaises(ValueError): w.validate_project_cargo_dependencies(value)

    def test_dependency_compiler_or_compile_mutation_rejected(self):
        for field, key in (('dependency_files_after', 'vendor/example/Cargo.toml'),
                           ('build_source_files_after', 'crates/sekirei-core/src/search.rs'),
                           ('compiler_files', '/tmp/public-rustc')):
            value = build_fixture(); value[field][key] = identity('d')
            with self.assertRaises(ValueError): w.validate_project_cargo_dependencies(value)

    def test_four_stages_own_gate_explicit_runtime_draft(self):
        d, p = draft()
        self.assertEqual(p.calls, ['preflight', 'new-output', 'mae-pilot', 'write-own-pilot-gate', 'mae', 'top3-pilot', 'top3', 'post-inputs'])
        self.assertEqual(d['runtime'], w.RUNTIME)
        self.assertFalse(d['executor_reaped']); self.assertFalse(d['cleanup_verified'])
        self.assertEqual(d['status'], 'measured-pending-parent-terminal')
        self.assertFalse(p.locked)

    def test_candidate_predecessor_refs_required(self):
        s = spec('candidate')
        self.assertIsNotNone(s['candidate_gate']); self.assertIsNotNone(s['fallback_bridge'])
        with self.assertRaises(ValueError):
            w.make_spec('candidate', s['prefix'], s['evaluation_path'], NATIVE, s['build_manifest'],
                s['build_identity'], s['source_helpers'], s['inputs'], None, None)

    def test_mutated_runtime_source_or_bridge_binding_rejected(self):
        for key, value in (('runtime', '/tmp/old-default'), ('resume', True), ('plan_sha256', '0'*64)):
            s = spec(); s[key] = value
            with self.assertRaises(ValueError): w.validate_spec(s)
        s = spec('candidate'); s['inputs'][s['fallback_bridge']['path']]['sha256'] = 'd'*64
        with self.assertRaises(ValueError): w.validate_spec(s)
        s = spec(); s['inputs'].pop('/tmp/public-helper.py')
        with self.assertRaises(ValueError): w.validate_spec(s)

    def test_stage_timeout_cleanup_missing_status_or_bool_rejected(self):
        for fields in ({'cleanup_ok': 1139}, {'timeout_failures': 1}, {'runner_exit_zero': True},
                       {'complete': False}, {'max_reported_nodes': 1010001}):
            value = stage('mae'); value.update(fields)
            with self.assertRaises(ValueError): w.validate_stage('mae', value)

    def test_pilot_failure_prevents_formal(self):
        s = spec(); outcomes = {name: stage(name) for name in w.STAGES}
        outcomes['mae-pilot']['stable_positions']['sekirei'] = 16
        p = w.SyntheticPorts(s['inputs'], outcomes)
        with self.assertRaises(ValueError): w.four_stage_call_sequence(s, config(), p)
        self.assertNotIn('mae', p.calls)
        self.assertFalse(p.locked)

    def test_parent_completion_and_typed_terminal_rejection(self):
        d, _ = draft(); t, op, pin = parent_docs(d)
        state = w.finalize_parent(d, t, op, pin)
        self.assertEqual(state['status'], 'complete')
        desc = w.descriptor(state, '/tmp/public-white-fallback', '/tmp/public-white-fallback/candidate.json', 'fallback', UNLOADED)
        self.assertTrue(desc['mae'].startswith(w.RUNTIME+'/runs/'))
        for fields in ({'exit_code': True}, {'tool_observed_reaped': False}, {'all_related_groups_observed_stopped': False}):
            bad = copy.deepcopy(t); bad.update(fields)
            with self.assertRaises(ValueError): w.finalize_parent(d, bad, op, pin)

    def test_incomplete_descriptor_before_raw_read(self):
        d, _ = draft()
        with self.assertRaises(ValueError): w.descriptor(d, '/tmp/public-white-fallback', '/tmp/public-white-fallback/candidate.json', 'fallback', UNLOADED)

    def test_fallback_bridge_requires_all1829_and_semantics(self):
        old, fresh, pins = bridge_fixture()
        result = w.fallback_bridge(old, fresh, pins)
        self.assertFalse(result['execution_identities_equal']); self.assertFalse(result['adopt'])
        for edit in (lambda x: x['stages']['top3'].update(cleanup_ok=550),
                     lambda x: x['semantic_projections']['sekirei570']['0'].update(score_cp_sente=999),
                     lambda x: x['per_game']['development-01'].update(absolute_error_sum_cp=401)):
            bad = copy.deepcopy(fresh); edit(bad)
            with self.assertRaises(ValueError): w.fallback_bridge(old, bad, pins)

    def test_bridge_same_binary_is_not_adoption(self):
        old, fresh, pins = bridge_fixture()
        old['comparable']['execution_except_model'] = copy.deepcopy(fresh['comparable']['execution_except_model'])
        with self.assertRaises(ValueError): w.fallback_bridge(old, fresh, pins)

    def test_joint_exact_criteria_same_newbinary(self):
        old, fresh, pins = bridge_fixture(); bridge = w.fallback_bridge(old, fresh, pins)
        candidate = bundle('candidate', error_delta=-1)
        result = w.compare_same_runtime(fresh, candidate, bridge, ref('/tmp/public-bridge.json'), NATIVE)
        self.assertTrue(result['adopt']); self.assertEqual(len(result['checks']), 8)
        candidate = bundle('candidate', error_delta=-1, hits_delta=-1)
        result = w.compare_same_runtime(fresh, candidate, bridge, ref('/tmp/public-bridge.json'), NATIVE)
        self.assertFalse(result['adopt']); self.assertTrue(result['mae_improved'])

    def test_unknown_binary_identity_never_removed(self):
        old, fresh, pins = bridge_fixture(); bridge = w.fallback_bridge(old, fresh, pins)
        candidate = bundle('candidate', error_delta=-1)
        candidate['comparable']['execution_except_model']['runtime']['binaries']['sekirei'] = 'different'
        with self.assertRaises(ValueError): w.compare_same_runtime(fresh, candidate, bridge, ref('/tmp/public-bridge.json'), NATIVE)

    def test_integer_only_metric_and_rounding_cannot_adopt(self):
        rows = bundle()['per_game']
        m0, t0 = w.equal_game_metrics(rows)
        changed = copy.deepcopy(rows); changed['development-05']['absolute_error_sum_cp'] -= 1
        m1, t1 = w.equal_game_metrics(changed)
        self.assertLess(m1, m0); self.assertEqual(t1, t0)
        for value in (True, 1.0):
            changed['development-05']['absolute_error_sum_cp'] = value
            with self.assertRaises(ValueError): w.equal_game_metrics(changed)

    def test_no_engine_or_actual_input_entry(self):
        self.assertTrue(w.PROTOTYPE_ONLY)
        for fn in (w.actual_read, w.actual_execute, w.main):
            with self.assertRaisesRegex(RuntimeError, 'SOURCE ONLY'): fn()
        with self.assertRaisesRegex(ValueError, 'SOURCE ONLY'):
            w.four_stage_call_sequence(spec(), config(), object())
        with self.assertRaisesRegex(ValueError, 'SOURCE ONLY'):
            w.legacy_stage_call('mae', spec(), config(), None, None, None)

    def test_legacy_call_api_is_explicit_runtime_no_resume(self):
        calls = []
        class B:
            def validate_formal_gate(self, runtime, cfg): calls.append(('formal-gate', str(runtime)))
            def execute_run(self, runtime, cfg, run_type, **kwargs): calls.append((run_type, str(runtime), kwargs))
        class Top:
            def run(self, args): calls.append((args.action, str(args.runtime), args.run_id, args.pilot_run_id, args.resume))
        s = spec()
        with mock.patch.object(w, 'PROTOTYPE_ONLY', False):
            for name in w.STAGES:
                w.legacy_stage_call(name, s, config(), '/tmp/public-config.json', B(), Top())
        self.assertTrue(all(row[1] == w.RUNTIME for row in calls))
        self.assertEqual(calls[0][2], {'run_id': s['run_ids']['mae-pilot'], 'resume': False})
        self.assertEqual(calls[3][3:], (None, False))
        self.assertEqual(calls[4][3:], (s['run_ids']['top3-pilot'], False))

    def test_native_gate_dedicated_full_producer_contract_and_external_refs(self):
        # Compile only the allowed source's pure document constructor. Every
        # reference/value below is synthetic; no artifact path is followed.
        source = Path(__file__).resolve().parents[1]/'preparations'/'white-view-model-proof-v1'/'white_view_proof_common.py'
        tree = ast.parse(source.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'gate_document')
        paths = {k: '/tmp/synthetic-'+k+'.json' for k in w.GATE_BINDING_NAMES}
        paths.update(metadata='/tmp/synthetic-Q/weights.meta.json', core='/tmp/synthetic-core/receipt.json',
                     incremental='/tmp/synthetic-incremental/receipt.json', native=NATIVE['path'])
        binding = {'schema': 'sekirei.white-view-candidate-gate-binding.v1', **{k: ref(v) for k,v in paths.items()}}
        binding['native'] = {k: NATIVE[k] for k in ('path','bytes','sha256')}
        refs = {v['path']:v for k,v in binding.items() if k != 'schema'}
        inputs = {p:{k:r[k] for k in ('bytes','sha256')} for p,r in refs.items() if p != binding['gate']['path']}
        expected_inputs = copy.deepcopy(inputs)
        expected_inputs[binding['gate']['path']] = {k:binding['gate'][k] for k in ('bytes','sha256')}
        build = {'runtime':w.RUNTIME,'manifest':ref('/tmp/synthetic-build.json'),
            'identity':ref('/tmp/synthetic-build-identity.json'),'validator_source':ref('/tmp/synthetic-builder.py')}
        feature = {'schema':'sekirei.white-view-paired-linear-feature-binding.v1','mode':w.MODE,
            'plan_sha256':w.PLAN_SHA256,'feature_schema':w.FEATURE_SCHEMA,'compile_feature':w.FEATURE,
            'native_magic':w.MAGIC,'dimensions':{'input':2420,'l1':256,'l2':32},
            'stock_initializer_sha256':'bbe9fbea4c943d69d605190f9ef8c6e9c7a4b9aa7c3a6be970d93e3405334e40',
            'feature_source_sha256':copy.deepcopy(w.GATE_FEATURE_SOURCES),'core_binary_sha256':'c'*64,
            'helper_source_sha256':'79845db6f8ceec2e0c73488b705db526813da52bcc601ab6b10eb3f52f0ce6eb'}
        ctx = {'binding':{'fit_receipt':binding['fit_run']},'native_ref':binding['native'],
            'reference_ref':binding['reference03'],'numeric_ref':binding['numeric_audit'],
            'preregistration':{'new_build_binding':build},'feature_binding':feature}
        args = SimpleNamespace(**{'expected_'+k+'_sha256':binding[v]['sha256'] for k,v in
            (('metadata','metadata'),('core','core'),('incremental','incremental'),
             ('preregistration','preregistration'),('activation','activation'),('source_preflight','source_preflight'),
             ('worker','gate_worker'),('common','common'))})
        def bound_ref(_reader,path,expected):
            result=refs[str(path)];self.assertEqual(result['sha256'],expected);return copy.deepcopy(result)
        env={'bound_ref':bound_ref,'Q':Path('/tmp/synthetic-Q'),'CORE_OUT':Path('/tmp/synthetic-core'),
             'INCR_OUT':Path('/tmp/synthetic-incremental'),'CANDIDATE':w.MODE,'PLAN_SHA':w.PLAN_SHA256}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),env)
        gate=env['gate_document'](args,None,ctx,{},inputs)
        self.assertTrue(w.validate_native_gate(gate,NATIVE,build,binding,expected_inputs))
        for change in ({'native_magic':'SEKIRW01'},{'epochs':False},{'adam_used':0},
                       {'canonical_new_abi_rebuild_equal':False},{'observed_core_integer_residual_cap_cp':101},
                       {'core_total_rows':True},{'no_child_process_started':True},
                       {'universal_integer_bound_proven':True},{'common_sha256':'0'*64}):
            bad=copy.deepcopy(gate);bad.update(change)
            with self.assertRaises(ValueError):w.validate_native_gate(bad,NATIVE,build,binding,expected_inputs)
        for key in ('reference_weight','proofs','feature_binding','new_build_binding'):
            bad=copy.deepcopy(gate)
            if key=='reference_weight':bad[key]['sha256']='0'*64
            elif key=='proofs':bad[key]['core']['sha256']='0'*64
            elif key=='feature_binding':bad[key]['dimensions']['l1']=True
            else:bad[key]['identity']['sha256']='0'*64
            with self.assertRaises(ValueError):w.validate_native_gate(bad,NATIVE,build,binding,expected_inputs)
        bad=copy.deepcopy(expected_inputs);bad.pop(binding['core_worker']['path'])
        with self.assertRaises(ValueError):w.validate_native_gate(gate,NATIVE,build,binding,bad)
        bad=copy.deepcopy(gate);bad['inputs_after'][binding['native']['path']]['sha256']='0'*64
        with self.assertRaises(ValueError):w.validate_native_gate(bad,NATIVE,build,binding,expected_inputs)
        bad=copy.deepcopy(gate);bad['inputs_before']['/tmp/not-frozen']=ref('/tmp/not-frozen')
        with self.assertRaises(ValueError):w.validate_native_gate(bad,NATIVE,build,binding,expected_inputs)

    def test_field_policy_matches_unchanged_compare_source(self):
        path = Path(__file__).resolve().parents[1]/'scripts'/'compare_candidates.py'
        tree = ast.parse(path.read_text())
        value = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'TEACHER_FIELDS' for t in node.targets))
        self.assertEqual(ast.literal_eval(value), w.SEMANTIC_FIELDS)

    def test_raw_parsed_bytes_pin_duplicate_nonfinite(self):
        raw = b'{"complete":true}'
        self.assertEqual(w.read_pinned_json_bytes(raw, hashlib.sha256(raw).hexdigest()), {'complete': True})
        for bad in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e1000}'):
            with self.assertRaises(ValueError): w.read_pinned_json_bytes(bad, hashlib.sha256(bad).hexdigest())
        with self.assertRaises(ValueError): w.read_pinned_json_bytes(raw, '0'*64)

    def test_schema_ast_parse_only(self):
        for name in ('white_view_evaluation.py', 'test_white_view_evaluation.py'):
            ast.parse(((Path(__file__).parent if name.startswith('test_') else Path(__file__).resolve().parents[1]/'preparations'/'white-view-evaluation-runtime-v3')/name).read_text())

    def test_projection_policy_freezes_fields_not_clock_frequency(self):
        old, fresh, pins = bridge_fixture()
        bad = copy.deepcopy(fresh)
        bad['projection_policy']['excluded_from_semantic_equality'].append('bestmove')
        with self.assertRaises(ValueError): w.fallback_bridge(old, bad, pins)
        bad = copy.deepcopy(fresh)
        bad['semantic_projections']['top3_pilot_36']['0']['ranks'] = [2, 1, 3]
        with self.assertRaises(ValueError): w.fallback_bridge(old, bad, pins)
        old['stages']['mae']['max_reported_nodes'] = 1000200
        fresh['stages']['mae']['max_reported_nodes'] = 1000300
        self.assertTrue(w.fallback_bridge(old, fresh, pins)['fallback_equivalent'])


if __name__ == '__main__':
    unittest.main()
