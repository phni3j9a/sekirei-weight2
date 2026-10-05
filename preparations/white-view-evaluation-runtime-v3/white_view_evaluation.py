"""SOURCE ONLY explicit-runtime evaluation/comparison adapter.

The production entry points stop before I/O. Pure consumers and the legacy
call sequence are testable in memory. Root must supply frozen source/build,
native/sidecar, original-input and terminal guards before enabling execution.
No DEFAULT_RUNTIME monkeypatch or cross-runtime identity weakening is used.
"""
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import PurePosixPath
import re
from types import SimpleNamespace

PROTOTYPE_ONLY = True
MODE = 'white-view-paired-linear-constrained-ridge1-l1-39p5-v1'
FEATURE = 'nnue_white_view_aux_tied'
FEATURE_SCHEMA = 'flat_white_view_aux_tied_v1'
MAGIC = 'SEKIRW03'
PLAN_SHA256 = '1f274b187a96b16674814c51f77cb402ca8a80e6a7e5d738a5caf9354127e836'
RUNTIME = '/home/server/.local/share/sekirei-weight2/suisho11beta-sekirei-v0.3.39-white-view-v1'
AUD = '/home/server/.local/share/sekirei-weight2/suisho11beta-v1/venv/bin/python'
STAGES = {'mae-pilot': 102, 'mae': 1140, 'top3-pilot': 36, 'top3': 551}
GAMES = {'development-01': 86, 'development-02': 126, 'development-03': 127,
         'development-04': 108, 'development-05': 123}
CHANGED = {'crates/sekirei-core/src/nnue.rs', 'crates/sekirei-core/Cargo.toml', 'crates/sekirei-usi/Cargo.toml'}
CARGO = {'crates/sekirei-core/Cargo.toml', 'crates/sekirei-usi/Cargo.toml'}
CHECKS = {'execution_except_model', 'occurrences', 'teacher_results', 'teacher_E_cp',
          'top3_occurrences', 'top3_legal_moves_sha256', 'top3_denominators', 'top3_teacher_bestmoves'}
DICT_CHECKS = {'execution_except_model', 'teacher_results', 'teacher_E_cp', 'top3_denominators', 'top3_teacher_bestmoves'}
SEVEN = CHECKS - {'execution_except_model'}
SEMANTIC_FIELDS = (
    'status', 'position_type', 'position_classification', 'requested_nodes',
    'score_kind', 'score_bound_stm', 'score_bound_sente', 'score_cp_stm',
    'score_cp_sente', 'reported_cp_stm', 'reported_cp_sente', 'score_mate_stm',
    'mate_distance', 'mate_distance_known', 'mate_sign', 'winner', 'winner_stm',
    'winner_sente', 'bestmove', 'bestmove_kind', 'pv', 'pv_head',
)
PILOT_METADATA = ('engine_id', 'game_id', 'ply', 'repetition', 'side_to_move', 'position_sha256')
TOP3_RANK_FIELDS = ('multipv', 'score_kind', 'score_raw', 'bound_stm', 'pv')
PROJECTION_POLICY = {
    'schema': 'sekirei.white-view-fallback-semantic-projection-policy.v1',
    'mae_semantic_fields': list(SEMANTIC_FIELDS), 'mae_pilot_metadata': list(PILOT_METADATA),
    'top3_rank_fields': list(TOP3_RANK_FIELDS),
    'top3_iteration': 'latest complete ranks 1,2,3 accepted by unchanged top3.parse_top3',
    'position_identity': 'exact engine-position-repetition set; formal occurrence keys and STM',
    'node_policy': {'requested_nodes': 1000000, 'max_reported_nodes': 1010000},
    'all_node_evidence_and_lifecycle': 'strictly reparsed separately on all 1829 attempts per runtime',
    'excluded_from_semantic_equality': ['runtime paths', 'binary identity', 'build identity', 'fingerprint',
        'depth', 'actual reported node counts', 'clock', 'time', 'nps', 'intermediate info frequency'],
    'retained_separately': ['both full runtime/build/binary identities', 'all raw SHA',
        'all node-policy/lifecycle/cleanup verification results', 'actual node maxima'],
}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def exact(a, b):
    if type(a) is not type(b):
        return False
    if type(a) is dict:
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if type(a) in (list, tuple):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


def sha(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'strict SHA256 required')
    return value


def absolute(value):
    require(type(value) is str and PurePosixPath(value).is_absolute()
            and str(PurePosixPath(value)) == value and '..' not in PurePosixPath(value).parts,
            'canonical absolute path text required')
    return PurePosixPath(value)


def file_identity(value):
    require(type(value) is dict and set(value) == {'bytes', 'sha256'}
            and type(value['bytes']) is int and value['bytes'] >= 0, 'strict file identity required')
    sha(value['sha256'])
    return value


def fullref(value):
    require(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'}, 'exact fullref required')
    absolute(value['path'])
    file_identity({k: value[k] for k in ('bytes', 'sha256')})
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_pinned_json_bytes(raw, expected):
    require(type(raw) is bytes and hashlib.sha256(raw).hexdigest() == sha(expected), 'externally pinned parsed bytes required')
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def nonfinite(_):
        raise ValueError('nonfinite JSON constant')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite,
                      parse_float=lambda x: nonfinite(x) if abs(float(x)) == float('inf') else float(x))


def validate_project_cargo_dependencies(manifest, identity=None):
    """Original -> patched maps; only Cargo2 changes, duplicated source exact.

    Externally pinned manifest/source identity and current on-disk verification
    remain the dedicated builder's responsibility. Compile-stage maps are
    independently required to stay unchanged; they are not patch-stage maps.
    """
    require(type(manifest) is dict, 'manifest dictionary required')
    source_before, source_after = manifest['source_files_before'], manifest['source_files_after']
    deps_before, deps_after = manifest['dependency_files_before'], manifest['dependency_files_after']
    for mapping in (source_before, source_after, deps_before, deps_after):
        require(type(mapping) is dict and mapping, 'full file inventory required')
        for path, record in mapping.items():
            require(type(path) is str and path and not PurePosixPath(path).is_absolute()
                    and str(PurePosixPath(path)) == path and '..' not in PurePosixPath(path).parts,
                    'canonical relative inventory key required')
            file_identity(record)
    require(len(source_before) == 526 and source_before.keys() == source_after.keys(), 'full tracked 526 set differs')
    require({p for p in source_before if not exact(source_before[p], source_after[p])} == CHANGED, 'three-file source patch required')
    require(deps_before.keys() == deps_after.keys() and CARGO <= deps_before.keys()
            and 'Cargo.lock' in deps_before, 'full same dependency set with Cargo2/lock required')
    require({p for p in deps_before if not exact(deps_before[p], deps_after[p])} == CARGO, 'only permitted project Cargo2 dependency changes')
    for path in deps_before:
        require(exact(deps_before[path], source_before[path]) and exact(deps_after[path], source_after[path]),
                'dependency duplicate records differ from tracked before/after')
    for kind, expected in (('source', source_after), ('dependency', deps_after)):
        before = manifest['build_' + kind + '_files_before']
        after = manifest['build_' + kind + '_files_after']
        require(exact(before, after) and exact(before, expected), 'compile stage changed source/dependencies')
    compiler = manifest.get('compiler_files')
    require(type(compiler) is dict and compiler and manifest.get('compiler_unchanged') is True
            and exact(manifest.get('inputs_before'), manifest.get('inputs_after')),
            'compiler/current immutable input maps changed')
    for path, record in compiler.items():
        absolute(path)
        file_identity(record)
        require(exact(manifest['inputs_before'].get(path), record), 'compiler input absent from build inventory')
    if identity is not None:
        require(type(identity) is dict and exact(identity.get('compiler_files'), compiler)
                and exact(identity.get('source_files'), source_before)
                and exact(identity.get('dependency_files'), deps_before), 'identity/source/dependency/compiler binding differs')
    return True


def model_identity(role, model):
    require(role in ('fallback', 'candidate'), 'explicit evaluation role required')
    fallback = {'kind': 'material_fallback', 'path': None, 'bytes': None, 'sha256': None}
    if role == 'fallback':
        require(exact(model, fallback), 'fallback must be unloaded')
    else:
        require(type(model) is dict and set(model) == {'kind', 'path', 'bytes', 'sha256'}
                and model['kind'] == 'nnue', 'explicit new native model required')
        fullref({k: model[k] for k in ('path', 'bytes', 'sha256')})
        require(model['bytes'] == 1305356, 'native03 size differs')
    return copy.deepcopy(model)


GATE_BINDING_NAMES = {'common', 'preregistration', 'activation', 'source_preflight', 'fit_run',
    'native', 'metadata', 'coefficients_f32', 'coefficients_f64', 'solver_certificate', 'reference03',
    'numeric_audit', 'numeric_worker', 'core', 'core_worker', 'incremental', 'incremental_worker', 'gate', 'gate_worker'}
GATE_FEATURE_SOURCES = {
    'crates/sekirei-core/src/nnue.rs': '083c99ed681108d1643fa897aa2a4c3f452a170a73219fa3dc2cf16cdf68de50',
    'crates/sekirei-core/Cargo.toml': 'c6c0d7a2e60b988e2975c5bc5b7dff620fe74b4426ff1ffc69474a0fa5cc2fc8',
    'crates/sekirei-usi/Cargo.toml': 'f0f9689226fc59c10b7d1adc10a101f9f31914bc6a3e342b29b68917ef73ffea'}


def validate_candidate_binding(binding, input_map):
    require(type(binding) is dict and set(binding) == {'schema'} | GATE_BINDING_NAMES
        and binding['schema'] == 'sekirei.white-view-candidate-gate-binding.v1', 'exact dedicated candidate gate binding required')
    require(type(input_map) is dict and input_map, 'externally frozen candidate inventory required')
    for key in GATE_BINDING_NAMES:
        rec = fullref(binding[key])
        require(exact(input_map.get(rec['path']), {k: rec[k] for k in ('bytes', 'sha256')}), 'candidate proof ref absent from launch inputs: '+key)
    return binding


def validate_native_gate(gate, expected_model, expected_build_binding, expected_binding, expected_inputs):
    """Full typed envelope from the dedicated new03 technical gate consumer.

    The source-pinned gate worker reconstructs native/fit/numeric/core/raw
    proofs. This second pure check binds the complete envelope to external
    launch controls; it never accepts self-reported references as expectations.
    """
    model_identity('candidate', expected_model)
    validate_candidate_binding(expected_binding, expected_inputs)
    require(type(expected_build_binding) is dict and set(expected_build_binding) ==
        {'runtime', 'manifest', 'identity', 'validator_source'} and expected_build_binding['runtime'] == RUNTIME,
        'exact dedicated build binding required')
    for key in ('manifest', 'identity', 'validator_source'):
        fullref(expected_build_binding[key])
    require(exact({k: expected_model[k] for k in ('path', 'bytes', 'sha256')}, expected_binding['native']), 'native model differs from proof binding')
    require(type(gate) is dict and type(gate.get('feature_binding')) is dict, 'complete new03 technical gate required')
    feature = gate['feature_binding']
    fixed_feature = {'schema': 'sekirei.white-view-paired-linear-feature-binding.v1', 'mode': MODE,
        'plan_sha256': PLAN_SHA256, 'feature_schema': FEATURE_SCHEMA, 'compile_feature': FEATURE,
        'native_magic': MAGIC, 'dimensions': {'input': 2420, 'l1': 256, 'l2': 32},
        'stock_initializer_sha256': 'bbe9fbea4c943d69d605190f9ef8c6e9c7a4b9aa7c3a6be970d93e3405334e40',
        'feature_source_sha256': GATE_FEATURE_SOURCES,
        'helper_source_sha256': '79845db6f8ceec2e0c73488b705db526813da52bcc601ab6b10eb3f52f0ce6eb',
        'core_binary_sha256': sha(feature.get('core_binary_sha256'))}
    require(exact(feature, fixed_feature), 'technical gate feature/source/helper declaration differs')
    proofs = {key: expected_binding[value] for key, value in
        (('fit_run', 'fit_run'), ('sidecar', 'metadata'), ('numeric_audit', 'numeric_audit'), ('core', 'core'), ('incremental', 'incremental'))}
    inputs = gate.get('inputs_before')
    require(type(inputs) is dict and inputs and exact(inputs, gate.get('inputs_after')), 'technical gate inputs changed/missing')
    for path, rec in inputs.items():
        absolute(path); file_identity(rec)
        require(exact(expected_inputs.get(path), rec), 'technical gate transitive input not launch-bound')
    for key in GATE_BINDING_NAMES - {'gate'}:
        rec = expected_binding[key]
        require(exact(inputs.get(rec['path']), {k: rec[k] for k in ('bytes', 'sha256')}), 'technical gate input missing proof/control: '+key)
    fixed = {'schema': 'sekirei.white-view-model-technical-gate.v1', 'status': 'complete',
        'candidate': MODE, 'mode': MODE, 'plan_sha256': PLAN_SHA256,
        'preregistration_sha256': expected_binding['preregistration']['sha256'],
        'activation_sha256': expected_binding['activation']['sha256'],
        'source_preflight_sha256': expected_binding['source_preflight']['sha256'],
        'feature_schema': FEATURE_SCHEMA, 'native_magic': MAGIC, 'model': expected_model,
        'reference_weight': expected_binding['reference03'], 'build_manifest': expected_build_binding['manifest'],
        'feature_binding': fixed_feature, 'new_build_binding': expected_build_binding, 'proofs': proofs,
        'adam_used': False, 'epochs': 0, 'resume_used': False, 'absolute_output': True,
        'canonical_new_abi_rebuild_equal': True, 'hand_full_row_ties_verified': True,
        'board_ft_bytes_preserved': True, 'ft_bias_bytes_preserved': True, 'material_bytes_preserved': True,
        'all_three_numeric_certificates_independently_verified': True, 'core_total_rows': 118591,
        'incremental_observations': 8185, 'observed_core_integer_residual_cap_cp': 100,
        'actual_reference03_core_material_verified': True, 'observed_float_core_bridge_lt_1p001': True,
        'all_observed_float_intermediates_finite': True, 'mxcsr_policy': 'x86-ftz-daz',
        'inputs_before': inputs, 'inputs_after': inputs, 'inputs_unchanged': True, 'source_unchanged': True,
        'stock_build_unchanged': True, 'new_build_unchanged': True, 'source_and_build_unchanged': True,
        'cleanup_verified': True, 'fit_executor_reaped': True, 'no_child_process_started': False,
        'no_engine_or_fit_process_started': True, 'runtime_validator_may_invoke_readonly_source_compiler_checks': True,
        'worker_sha256': expected_binding['gate_worker']['sha256'], 'common_sha256': expected_binding['common']['sha256'],
        'universal_integer_bound_proven': False, 'native_bitexact_covariance_proven': False,
        'python_native_forward_exactness_proven': False, 'search_benchmark_verified': False,
        'final_used': False, 'adoption_verified': False}
    require(exact(gate, fixed), 'full typed new-mode technical gate differs')
    return True


def make_spec(role, prefix, evaluation_path, model, build_manifest, build_identity,
              source_helpers, input_map, candidate_gate=None, bridge=None):
    model = model_identity(role, model)
    root = absolute(evaluation_path)
    require(type(prefix) is str and re.fullmatch('[a-z0-9][a-z0-9_.-]{0,127}', prefix)
            and 'final' not in prefix, 'development-only safe prefix required')
    fullref(build_manifest); fullref(build_identity)
    require(type(source_helpers) is dict and source_helpers, 'externally frozen helper set required')
    for path, value in source_helpers.items():
        absolute(path); sha(value)
    require(type(input_map) is dict and input_map, 'externally fixed input inventory required')
    for path, value in input_map.items():
        absolute(path); file_identity(value)
    for path, expected_sha in source_helpers.items():
        require(path in input_map and input_map[path]['sha256'] == expected_sha, 'helper inventory missing externally frozen source')
    for record in (build_manifest, build_identity):
        require(exact(input_map.get(record['path']), {k: record[k] for k in ('bytes', 'sha256')}), 'build pin not in immutable inputs')
    if role == 'candidate':
        fullref(candidate_gate); fullref(bridge)
        for record in (candidate_gate, bridge, {k: model[k] for k in ('path', 'bytes', 'sha256')}):
            require(exact(input_map.get(record['path']), {k: record[k] for k in ('bytes', 'sha256')}), 'candidate prerequisite pin not in immutable inputs')
    else:
        require(candidate_gate is None and bridge is None, 'fallback has no fitted-model or completed bridge dependency')
    return {'schema': 'sekirei.white-view-evaluation-spec.v1', 'role': role, 'mode': MODE, 'plan_sha256': PLAN_SHA256,
        'runtime': RUNTIME, 'evaluation_path': str(root), 'prefix': prefix,
        'run_ids': {k: prefix + '-' + k for k in STAGES}, 'model': model,
        'build_manifest': copy.deepcopy(build_manifest), 'build_identity': copy.deepcopy(build_identity),
        'source_helpers': copy.deepcopy(source_helpers), 'inputs': copy.deepcopy(input_map),
        'candidate_gate': copy.deepcopy(candidate_gate), 'fallback_bridge': copy.deepcopy(bridge),
        'interpreter': AUD, 'resume': False, 'stage_results_reused': False, 'final_used': False}


def validate_spec(spec):
    require(type(spec) is dict, 'fixed spec dictionary required')
    expected = make_spec(spec['role'], spec['prefix'], spec['evaluation_path'], spec['model'],
        spec['build_manifest'], spec['build_identity'], spec['source_helpers'], spec['inputs'],
        spec['candidate_gate'], spec['fallback_bridge'])
    require(exact(spec, expected), 'evaluation spec changed after creation')
    return spec


def config_for_role(base, spec):
    validate_spec(spec)
    require(type(base) is dict and base.get('split') == 'development'
            and exact(base.get('requested_nodes'), 1000000)
            and exact(base['universe']['plies'], GAMES), 'fixed development config required')
    result = copy.deepcopy(base)
    for engine, hash_name in (('teacher', 'USI_Hash'), ('sekirei', 'Hash')):
        options = result['engines'][engine]['options']
        require(options.get('Threads') == '1' and options.get(hash_name) == '128'
                and options.get('MultiPV') == '1', 'fixed Threads/Hash/MultiPV required')
    require('EvalFile' not in result['engines']['sekirei']['options']
            and 'NnueOutput' not in result['engines']['sekirei']['options'], 'base config contains model overrides')
    require(exact(result['formal']['max_reported_nodes'], 1010000)
            and exact(result['pilot']['repetitions'], 3), 'fixed node cap and repeats required')
    result['candidate_model'] = {'kind': spec['model']['kind'], 'path': spec['model']['path']}
    result['formal']['pilot_evidence'] = None
    return result


def validate_stage(stage, value):
    count = STAGES[stage]
    fixed = {'stage': stage, 'attempts': count, 'raw_sha_verified': count,
        'lifecycle_verified': count, 'cleanup_ok': count, 'runner_exit_zero': count,
        'technical_failures': 0, 'timeout_failures': 0, 'complete': True, 'final_used': False}
    require(type(value) is dict and all(exact(value.get(k), v) for k, v in fixed.items()), 'stage evidence incomplete/invalid')
    require(type(value.get('max_reported_nodes')) is int and 1 <= value['max_reported_nodes'] <= 1010000, 'strict node cap failed')
    sha(value.get('fingerprint'))
    if stage == 'mae-pilot':
        require(exact(value.get('stable_positions'), {'teacher': 17, 'sekirei': 17})
                and exact(value.get('unstable_positions'), {'teacher': 0, 'sekirei': 0}), 'own MAE pilot unstable')
    if stage == 'top3-pilot':
        require(exact(value.get('stable_positions'), 12) and exact(value.get('unstable_positions'), 0), 'own Top3 pilot unstable')
    return value


def validate_preflight_envelope(value, spec):
    fixed = {'schema': 'sekirei.white-view-evaluation-preflight-envelope.v1',
        'status': 'verified-before-first-write', 'role': spec['role'], 'runtime': RUNTIME,
        'interpreter': AUD, 'runtime_verified': True, 'original_input_verified': True,
        'source_and_build_unchanged': True, 'externally_frozen_bindings_verified': True,
        'output_and_four_run_paths_absent': True, 'inputs': spec['inputs'],
        'build_manifest': spec['build_manifest'], 'build_identity': spec['build_identity'],
        'candidate_gate': spec['candidate_gate'], 'fallback_bridge': spec['fallback_bridge'],
        'native_gate_verified': spec['role'] == 'candidate',
        'fallback_bridge_verified': spec['role'] == 'candidate',
        'final_used': False, 'adoption_verified': False}
    require(type(value) is dict and set(value) == set(fixed) and exact(value, fixed),
            'dedicated locked preflight/native/bridge verification required before write')
    return value['inputs']


def four_stage_call_sequence(spec, config, ports):
    """Explicit runtime normal4stage sequence, no outer benchmark lock.

    ports must be Root's verified production adapter or the exact synthetic
    test port. Production methods own source/current input/binary/native pins,
    new build+prepare SH locks and Root's global serial locks until cleanup.
    Each stage strictly reparses all raw/lifecycle evidence before returning.
    This function emits a draft; parent wait/reap is not inferred from return.
    """
    require(not PROTOTYPE_ONLY or type(ports) is SyntheticPorts, 'SOURCE ONLY: actual execution disabled before I/O')
    validate_spec(spec)
    config = config_for_role(config, spec)
    evidence = {}
    with ports.coordinating_locks(spec):
        before = validate_preflight_envelope(ports.preflight(spec, config), spec)
        require(exact(before, spec['inputs']), 'current preflight input map differs')
        ports.new_output(spec, config)
        for stage in STAGES:
            if stage == 'mae':
                pilot = evidence['mae-pilot']
                config['formal']['pilot_evidence'] = {'pilot_run_id': spec['run_ids']['mae-pilot'],
                    'pilot_fingerprint': pilot['fingerprint'], 'observed_max_reported_nodes': pilot['max_reported_nodes'],
                    'max_reported_nodes': 1010000}
                ports.write_config(spec, config)
            ports.run_stage(stage, spec, config)
            evidence[stage] = validate_stage(stage, ports.audit_stage(stage, spec, config))
        after = ports.current_inputs(spec)
        require(exact(before, after), 'input/source/build/model changed during stages')
    return {'schema': 'sekirei.white-view-evaluation-draft.v1', 'status': 'measured-pending-parent-terminal',
        'candidate': MODE, 'role': spec['role'], 'runtime': RUNTIME,
        'feature_schema': FEATURE_SCHEMA, 'native_magic': MAGIC, 'prefix': spec['prefix'],
        'run_ids': copy.deepcopy(spec['run_ids']), 'candidate_model': copy.deepcopy(spec['model']),
        'build_manifest': copy.deepcopy(spec['build_manifest']), 'build_identity': copy.deepcopy(spec['build_identity']),
        'stage_evidence': evidence, 'inputs_before': before, 'inputs_after': after,
        'stage_results_reused': False, 'executor_reaped': False, 'cleanup_verified': False,
        'final_used': False, 'adoption_verified': False}


def legacy_stage_call(stage, spec, config, config_path, b, top3):
    """The unchanged helper APIs. Only Root's enabled port may call this."""
    require(not PROTOTYPE_ONLY, 'SOURCE ONLY: legacy engines disabled before I/O')
    from pathlib import Path
    runtime = Path(spec['runtime'])
    ids = spec['run_ids']
    if stage in ('mae-pilot', 'mae'):
        if stage == 'mae':
            b.validate_formal_gate(runtime, config)
        return b.execute_run(runtime, config, 'pilot' if stage == 'mae-pilot' else 'formal',
                             run_id=ids[stage], resume=False)
    return top3.run(SimpleNamespace(runtime=runtime, mae_config=config_path,
        mae_run=runtime/'runs'/ids['mae'], resume=False, action='pilot' if stage == 'top3-pilot' else 'formal',
        run_id=ids[stage], pilot_run_id=None if stage == 'top3-pilot' else ids['top3-pilot']))


def finalize_parent(draft, terminal, operational, expected_terminal):
    """Parent alone publishes complete after terminal bytes and full cleanup."""
    fullref(expected_terminal)
    fixed = {'schema': 'sekirei.white-view-evaluation-terminal.v1', 'status': 'observed-stopped',
        'exit_code': 0, 'session_closed': True, 'tool_observed_reaped': True,
        'all_related_groups_observed_stopped': True, 'role': draft['role'],
        'run_ids': draft['run_ids'], 'model': draft['candidate_model']}
    require(type(terminal) is dict and all(exact(terminal.get(k), v) for k, v in fixed.items()), 'successful external normal terminal required')
    require(type(terminal.get('executor_session_id')) is int and terminal['executor_session_id'] > 0, 'strict actual executor session required')
    require(type(draft) is dict and draft.get('status') == 'measured-pending-parent-terminal'
            and draft.get('executor_reaped') is False and draft.get('cleanup_verified') is False, 'draft cannot claim parent completion')
    for stage in STAGES:
        validate_stage(stage, draft['stage_evidence'][stage])
    require(type(operational) is dict and operational.get('schema') == 'sekirei.white-view-evaluation-operational-verification.v1'
            and operational.get('status') == 'complete' and operational.get('role') == draft['role']
            and operational.get('inputs_unchanged') is True and operational.get('source_and_build_unchanged') is True
            and operational.get('cleanup_verified') is True and operational.get('final_used') is False
            and exact(operational.get('inputs_before'), draft['inputs_before'])
            and exact(operational.get('inputs_after'), draft['inputs_after'])
            and exact(operational.get('terminal'), expected_terminal), 'complete externally bound operational receipt required')
    require(exact(draft['inputs_before'], draft['inputs_after']), 'draft inputs changed')
    state = copy.deepcopy(draft)
    state.update(schema='sekirei.white-view-evaluation.v1', status='complete', executor_reaped=True,
                 cleanup_verified=True, terminal=copy.deepcopy(expected_terminal))
    return state


def descriptor(state, evaluation_path, config_path, role, expected_model):
    model_identity(role, expected_model)
    root = absolute(evaluation_path)
    require(absolute(config_path) == root/'candidate.json', 'descriptor config path differs')
    fixed = {'schema': 'sekirei.white-view-evaluation.v1', 'status': 'complete', 'candidate': MODE,
        'role': role, 'runtime': RUNTIME, 'feature_schema': FEATURE_SCHEMA, 'native_magic': MAGIC,
        'candidate_model': expected_model, 'executor_reaped': True, 'cleanup_verified': True,
        'stage_results_reused': False, 'final_used': False, 'adoption_verified': False}
    require(type(state) is dict and all(exact(state.get(k), v) for k, v in fixed.items()), 'completed new role descriptor required')
    prefix = state.get('prefix')
    require(type(prefix) is str and re.fullmatch('[a-z0-9][a-z0-9_.-]{0,127}', prefix)
            and 'final' not in prefix, 'development-only run prefix required')
    require(exact(state.get('run_ids'), {s: prefix+'-'+s for s in STAGES}), 'four role run IDs differ')
    for stage in STAGES:
        validate_stage(stage, state['stage_evidence'][stage])
    return {'config': config_path, 'evaluation': str(root/'evaluation.json'), 'runtime': RUNTIME,
        'mae': str(PurePosixPath(RUNTIME)/'runs'/state['run_ids']['mae']),
        'top3': str(PurePosixPath(RUNTIME)/'runs'/state['run_ids']['top3'])}


def equal_game_metrics(rows):
    require(type(rows) is dict and rows.keys() == GAMES.keys(), 'exact five games required')
    total_mae = total_top3 = Fraction(0)
    e_total = t_total = 0
    for game, row in rows.items():
        require(type(row) is dict and set(row) == {'absolute_error_sum_cp', 'e_count', 'top3_hits', 'top3_denominator'}, 'exact integer metric fields required')
        require(all(type(x) is int for x in row.values()) and row['absolute_error_sum_cp'] >= 0
                and 0 < row['e_count'] <= GAMES[game] and 0 < row['top3_denominator'] <= GAMES[game]
                and 0 <= row['top3_hits'] <= row['top3_denominator'], 'invalid integer metric counts')
        total_mae += Fraction(row['absolute_error_sum_cp'], row['e_count'])/5
        total_top3 += Fraction(row['top3_hits'], row['top3_denominator'])/5
        e_total += row['e_count']; t_total += row['top3_denominator']
    require(e_total == 266 and t_total == 551, 'Teacher-E266/Top3551 coverage differs')
    return total_mae, total_top3


def validate_raw_bundle(bundle, role, expected_model, runtime):
    model_identity(role, expected_model)
    fixed = {'schema': 'sekirei.white-view-reparsed-role-bundle.v1', 'status': 'complete',
        'role': role, 'runtime': runtime, 'model': expected_model, 'raw_reparsed': True,
        'inputs_unchanged': True, 'cleanup_verified': True, 'executor_reaped': True,
        'final_used': False, 'adoption_verified': False}
    require(type(bundle) is dict and all(exact(bundle.get(k), v) for k, v in fixed.items()), 'complete independent raw role bundle required')
    require(type(bundle.get('stages')) is dict and bundle['stages'].keys() == STAGES.keys(), 'all four stages required')
    for stage in STAGES:
        validate_stage(stage, bundle['stages'][stage])
    comparable = bundle.get('comparable')
    require(type(comparable) is dict and set(comparable) == CHECKS, 'legacy eight identity fields required')
    require(len(comparable['occurrences']) == 570 and len(comparable['teacher_results']) == 570
            and len(comparable['teacher_E_cp']) == 266 and len(comparable['top3_occurrences']) == 551,
            'full formal comparable coverage differs')
    projections = bundle.get('semantic_projections')
    require(type(projections) is dict and set(projections) == {'mae_pilot_102', 'teacher570', 'sekirei570', 'top3_pilot_36', 'top3_551'}, 'full1829 semantic outcomes required')
    require(type(projections['teacher570']) is dict and type(projections['sekirei570']) is dict
            and len(projections['teacher570']) == len(projections['sekirei570']) == 570
            and projections['teacher570'].keys() == projections['sekirei570'].keys()
            and exact(projections['teacher570'], comparable['teacher_results'])
            and type(projections['top3_551']) is dict and len(projections['top3_551']) == 551,
            'semantic key sets/teacher origin differ')
    require(set(projections['top3_551']) == set(comparable['top3_occurrences']), 'Top3 semantic occurrence set differs')
    require(type(projections['mae_pilot_102']) is dict and len(projections['mae_pilot_102']) == 102
            and type(projections['top3_pilot_36']) is dict and len(projections['top3_pilot_36']) == 36,
            'own pilot semantic coverage incomplete')
    require(exact(bundle.get('projection_policy'), PROJECTION_POLICY), 'pre-frozen semantic policy differs')
    equal_game_metrics(bundle['per_game'])
    return bundle


def fallback_bridge(old, fresh, pins):
    require(type(pins) is dict and set(pins) == {'old_audit', 'new_audit', 'old_runtime', 'old_build', 'new_build'}, 'exact bridge pins required')
    for key in ('old_audit', 'new_audit', 'old_build', 'new_build'):
        fullref(pins[key])
    require(pins['old_runtime'] != RUNTIME, 'cross-runtime bridge must record different runtime')
    absolute(pins['old_runtime'])
    fallback = {'kind': 'material_fallback', 'path': None, 'bytes': None, 'sha256': None}
    validate_raw_bundle(old, 'fallback', fallback, pins['old_runtime'])
    validate_raw_bundle(fresh, 'fallback', fallback, RUNTIME)
    require(exact(old['audit_ref'], pins['old_audit']) and exact(fresh['audit_ref'], pins['new_audit']), 'external complete raw audit pins differ')
    require(exact(old['build_manifest'], pins['old_build']) and exact(fresh['build_manifest'], pins['new_build']), 'two frozen build references differ')
    require(not exact(old['comparable']['execution_except_model'], fresh['comparable']['execution_except_model']), 'new/old execution differences must remain visible')
    for field in SEVEN:
        require(exact(old['comparable'][field], fresh['comparable'][field]), 'common identity differs: '+field)
    require(exact(old['semantic_projections'], fresh['semantic_projections']) and exact(old['per_game'], fresh['per_game']), 'fallback semantics/exact integer metrics changed')
    return {'schema': 'sekirei.white-view-fallback-semantic-bridge.v1', 'status': 'complete',
        'comparison_valid': True, 'fallback_equivalent': True, 'old_audit': copy.deepcopy(pins['old_audit']),
        'new_audit': copy.deepcopy(pins['new_audit']), 'old_build': copy.deepcopy(pins['old_build']),
        'new_build': copy.deepcopy(pins['new_build']), 'old_execution_sha256': digest(old['comparable']['execution_except_model']),
        'new_execution_sha256': digest(fresh['comparable']['execution_except_model']),
        'semantic_sha256': digest(old['semantic_projections']), 'per_game': copy.deepcopy(old['per_game']),
        'projection_policy': copy.deepcopy(PROJECTION_POLICY),
        'execution_identities_equal': False, 'old_results_reused_as_new': False, 'final_used': False, 'adopt': False}


def compare_same_runtime(fallback, candidate, bridge, expected_bridge_ref, model):
    fullref(expected_bridge_ref)
    require(type(bridge) is dict and bridge.get('schema') == 'sekirei.white-view-fallback-semantic-bridge.v1'
            and bridge.get('status') == 'complete' and bridge.get('comparison_valid') is True
            and bridge.get('fallback_equivalent') is True and bridge.get('execution_identities_equal') is False
            and bridge.get('old_results_reused_as_new') is False and bridge.get('adopt') is False
            and bridge.get('final_used') is False, 'externally pinned completed fallback bridge required')
    unloaded = {'kind': 'material_fallback', 'path': None, 'bytes': None, 'sha256': None}
    validate_raw_bundle(fallback, 'fallback', unloaded, RUNTIME)
    validate_raw_bundle(candidate, 'candidate', model, RUNTIME)
    require(exact(bridge['new_audit'], fallback['audit_ref']) and exact(bridge['new_build'], fallback['build_manifest'])
            and bridge['new_execution_sha256'] == digest(fallback['comparable']['execution_except_model'])
            and exact(bridge['per_game'], fallback['per_game']), 'bridge subject is not current new fallback')
    require(exact(fallback['build_manifest'], candidate['build_manifest']), 'same-newbinary manifest differs')
    checks = {}
    for field in CHECKS:
        a, b = fallback['comparable'][field], candidate['comparable'][field]
        require(exact(a, b), 'strict same-runtime identity differs: '+field)
        item = {'equal': True, 'baseline_sha256': digest(a), 'candidate_sha256': digest(b)}
        if field in DICT_CHECKS:
            item['mismatch_count'] = 0
        checks[field] = item
    mae0, top0 = equal_game_metrics(fallback['per_game'])
    mae1, top1 = equal_game_metrics(candidate['per_game'])
    return {'schema': 'sekirei.white-view-same-runtime-comparison.v1', 'status': 'complete',
        'comparison_valid': True, 'checks': checks, 'fallback_bridge': copy.deepcopy(expected_bridge_ref),
        'mae_improved': mae1 < mae0, 'top3_preserved': top1 >= top0, 'adopt': mae1 < mae0 and top1 >= top0,
        'baseline_exact': {'mae': ratio(mae0), 'top3': ratio(top0)},
        'candidate_exact': {'mae': ratio(mae1), 'top3': ratio(top1)}, 'final_used': False}


def ratio(value):
    return {'numerator': value.numerator, 'denominator': value.denominator}


def legacy_reparse_descriptor(desc, config, b, br, top3, cc):
    """Return complete strict legacy evidence; new state is not old state.

    Root's port must first pin all source and input bytes, call new runtime
    verifier before/after, and surround this with a before/after inventory.
    This adapter calls existing parsers without changing their contracts.
    """
    require(not PROTOTYPE_ONLY, 'SOURCE ONLY: actual raw reading disabled before I/O')
    from pathlib import Path
    plain = dict(desc, state=None)
    for key in ('config', 'evaluation', 'runtime', 'mae', 'top3'):
        if plain.get(key) is not None:
            plain[key] = Path(plain[key])
    comparable, summary, mae, rate = cc.verify_run(plain, config)
    stages = {}
    records = None
    for stage, path, run_type in (
        ('mae-pilot', plain['runtime']/'runs'/config['formal']['pilot_evidence']['pilot_run_id'], 'pilot'),
        ('mae', plain['mae'], 'formal')):
        manifest, _, _, rows, _, _ = b.validate_run_artifacts(path, config, require_complete=True, expected_run_type=run_type)
        require(len(rows) == STAGES[stage], 'strict MAE attempt count differs')
        stages[stage] = (manifest, rows)
        if stage == 'mae':
            records = rows
    ctx = top3.context(plain['runtime'], config, plain['mae'])
    formal_manifest = cc.read_json(plain['top3']/'manifest.json')
    for stage, path in (('top3-pilot', plain['runtime']/'runs'/formal_manifest['pilot_run_id']), ('top3', plain['top3'])):
        report = top3.inspect(path, *ctx, config)
        require(report['valid'] is True and report['attempts'] == STAGES[stage], 'strict Top3 attempt count differs')
        stages[stage] = (report, path)
    require(exact(tuple(cc.TEACHER_FIELDS), SEMANTIC_FIELDS), 'frozen legacy semantic field set differs')
    teachers, sekirei = {}, {}
    for row in records:
        require(row['engine_id'] in ('teacher', 'sekirei'), 'unexpected engine')
        result = row['result']
        require(all(k in result for k in cc.TEACHER_FIELDS), 'semantic field missing')
        key = f"{row['game_id']}:{row['ply']}"
        target = teachers if row['engine_id'] == 'teacher' else sekirei
        require(key not in target, 'duplicate semantic occurrence')
        target[key] = {'side_to_move': row['side_to_move'], 'position_sha256': row['position_sha256'],
                       **{field: result[field] for field in cc.TEACHER_FIELDS}}
    pilot_semantics = {}
    for row in stages['mae-pilot'][1]:
        require(all(k in row['result'] for k in SEMANTIC_FIELDS), 'pilot semantic field missing')
        key = row['attempt_id']
        require(key not in pilot_semantics, 'duplicate pilot attempt')
        pilot_semantics[key] = {**{k: row[k] for k in PILOT_METADATA},
                               'semantic': {k: row['result'][k] for k in SEMANTIC_FIELDS}}
    identity, eligible, teacher_reference, legal, games = ctx
    top_domains = {}
    for stage, path, run_type in (
        ('top3-pilot', stages['top3-pilot'][1], 'pilot'), ('top3', plain['top3'], 'formal')):
        positions, repetitions = top3.select_positions(eligible, run_type, config)
        projection = {}
        for position in positions:
            for rep in range(1, repetitions+1):
                exp = top3.expected(position, rep)
                row = cc.read_json(b._attempt_path(path, exp['attempt_id']))
                events = b._read_raw_event_log(b._log_path(path, exp['attempt_id']))
                top3.parse_top3(events, row['result']['bestmove'], legal[(position['game_id'], position['ply'])])
                latest = {}
                for info in b._structured_info_parses(events):
                    if 'score_kind' not in info:
                        continue
                    rank = info['multipv']
                    if rank == 1:
                        latest = {}
                    latest[rank] = {'multipv': rank, 'score_kind': info['score_kind'], 'score_raw': info['score_raw'],
                                    'bound_stm': info.get('bound_stm', 'exact'), 'pv': copy.deepcopy(info['pv'])}
                require(set(latest) == {1, 2, 3}, 'semantic Top3 complete ranks required')
                key = exp['attempt_id'] if stage == 'top3-pilot' else position['occurrence_id']
                require(key not in projection, 'duplicate Top3 semantic attempt')
                projection[key] = {'engine_id': 'sekirei', 'game_id': position['game_id'], 'ply': position['ply'],
                    'repetition': rep, 'side_to_move': position['side_to_move'], 'bestmove': row['result']['bestmove'],
                    'teacher_bestmove': teacher_reference[(position['game_id'], position['ply'])]['bestmove'],
                    'ranks': [latest[i] for i in (1, 2, 3)]}
        require(len(projection) == STAGES[stage], 'Top3 semantic full attempt coverage differs')
        top_domains[stage] = projection
    metrics = {}
    for game, per in summary['per_game'].items():
        n = per['e_count']
        exact_mae = Fraction(per['mae']['numerator'], per['mae']['denominator'])
        error_sum = exact_mae*n
        require(error_sum.denominator == 1, 'integer error sum not recoverable')
        metrics[game] = {'absolute_error_sum_cp': error_sum.numerator, 'e_count': n,
                        'top3_hits': per['top3']['hits'], 'top3_denominator': per['top3']['denominator']}
    require(equal_game_metrics(metrics) == (mae, rate), 'independent rational aggregate differs')
    return {'comparable': comparable, 'per_game': metrics, 'semantic_projections': {
        'mae_pilot_102': pilot_semantics, 'teacher570': teachers, 'sekirei570': sekirei,
        'top3_pilot_36': top_domains['top3-pilot'], 'top3_551': top_domains['top3']},
        'projection_policy': copy.deepcopy(PROJECTION_POLICY), 'legacy_stage_artifacts': stages}


class SyntheticPorts:
    """In-memory-only test port, never imports an engine or reads files."""
    def __init__(self, inputs, outcomes):
        self.inputs, self.outcomes = copy.deepcopy(inputs), copy.deepcopy(outcomes)
        self.calls, self.locked = [], False
    def coordinating_locks(self, spec):
        owner = self
        class Held:
            def __enter__(self): owner.locked = True; return self
            def __exit__(self, *args): owner.locked = False
        return Held()
    def preflight(self, spec, config):
        require(self.locked, 'locks required'); self.calls.append('preflight')
        return {'schema': 'sekirei.white-view-evaluation-preflight-envelope.v1',
            'status': 'verified-before-first-write', 'role': spec['role'], 'runtime': RUNTIME,
            'interpreter': AUD, 'runtime_verified': True, 'original_input_verified': True,
            'source_and_build_unchanged': True, 'externally_frozen_bindings_verified': True,
            'output_and_four_run_paths_absent': True, 'inputs': copy.deepcopy(self.inputs),
            'build_manifest': spec['build_manifest'], 'build_identity': spec['build_identity'],
            'candidate_gate': spec['candidate_gate'], 'fallback_bridge': spec['fallback_bridge'],
            'native_gate_verified': spec['role'] == 'candidate',
            'fallback_bridge_verified': spec['role'] == 'candidate',
            'final_used': False, 'adoption_verified': False}
    def new_output(self, spec, config): self.calls.append('new-output')
    def write_config(self, spec, config): self.calls.append('write-own-pilot-gate')
    def run_stage(self, stage, spec, config):
        require(self.locked, 'locks required')
        if stage == 'mae': require(config['formal']['pilot_evidence']['pilot_run_id'] == spec['run_ids']['mae-pilot'], 'own pilot required')
        self.calls.append(stage)
    def audit_stage(self, stage, spec, config): return copy.deepcopy(self.outcomes[stage])
    def current_inputs(self, spec): self.calls.append('post-inputs'); return copy.deepcopy(self.inputs)


def actual_read(*args, **kwargs):
    raise RuntimeError('SOURCE ONLY: actual controls/model/raw/data reader disabled')


def actual_execute(*args, **kwargs):
    raise RuntimeError('SOURCE ONLY: Root integration and enabling are required')


def main():
    raise RuntimeError('SOURCE ONLY: CLI stops before any actual read or execution')


if __name__ == '__main__':
    main()
