"""Synthetic in-memory receipts only; no actual private artifact or runtime."""
import ast
import copy
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('publication', HERE/'publish_white_view_comparison.py')
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
W = HERE.parent/'white-view-evaluation-runtime-v4'/'white_view_evaluation.py'


def producer_pure():
    """Compile an explicit pure AST allowlist, never actual read/execute ports."""
    names = {'MODE', 'PLAN_SHA256', 'RUNTIME', 'STAGES', 'GAMES', 'CHECKS', 'DICT_CHECKS', 'SEVEN',
             'PROJECTION_POLICY', 'SEMANTIC_FIELDS', 'PILOT_METADATA', 'TOP3_RANK_FIELDS',
             'require', 'exact', 'sha', 'absolute', 'file_identity', 'fullref', 'digest',
             'model_identity', 'validate_stage', 'equal_game_metrics', 'validate_raw_bundle',
             'fallback_bridge', 'compare_same_runtime', 'ratio'}
    tree = ast.parse(W.read_bytes())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names: nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets): nodes.append(node)
    env = {'Fraction': Fraction, 'PurePosixPath': PurePosixPath, 'copy': copy,
           'json': json, 'math': math, 're': re, 'hashlib': hashlib}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(W), 'exec'), env)
    return types.SimpleNamespace(**env)


def fixture(delta=-100, hits_delta=0):
    w = producer_pure(); docs = {}; identities = {}
    def put(path, document):
        raw = (json.dumps(document, sort_keys=True, allow_nan=False)+'\n').encode()
        ref = {'path': path, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        docs[path] = copy.deepcopy(document); identities[path] = p.small(ref)
        return ref
    def marker(path): return put(path, {'synthetic_source': True})
    contract = marker('/private/source/white_view_evaluation.py')
    producer = marker('/private/source/white_view_operations.py')
    worker = marker('/private/source/publish_white_view_comparison.py')
    new_bin = marker('/private/new/bin/sekirei'); teacher_bin = marker('/private/new/bin/yaneuraou')
    new_build_doc = {'schema': 'sekirei.white-view-runtime-build.v1', 'status': 'complete',
                     'architecture': {'native_magic': 'SEKIRW03', 'compile_feature': 'nnue_white_view_aux_tied'},
                     'binaries': {'sekirei': new_bin, 'yaneuraou': teacher_bin}}
    new_build = put('/private/new/build-manifest.json', new_build_doc)
    old_build_doc = {'binaries': {'sekirei': {'path': '/private/old/bin/sekirei', 'sha256': 'e'*64}}}
    old_build = put('/private/old/build-manifest.json', old_build_doc)
    ecounts = (40, 60, 60, 50, 56); denominators = (80, 124, 125, 106, 116)
    games = list(w.GAMES)
    baseline_rows = {g: {'absolute_error_sum_cp': e*1000, 'e_count': e,
                         'top3_hits': n//2, 'top3_denominator': n} for g, e, n in zip(games, ecounts, denominators)}
    candidate_rows = copy.deepcopy(baseline_rows)
    for row in candidate_rows.values():
        row['absolute_error_sum_cp'] += row['e_count']*delta
        row['top3_hits'] += hits_delta
    teacher = {str(i): {'synthetic': i} for i in range(570)}
    projections = {'mae_pilot_102': {str(i): i for i in range(102)}, 'teacher570': teacher,
                   'sekirei570': {str(i): i for i in range(570)},
                   'top3_pilot_36': {str(i): i for i in range(36)}, 'top3_551': {str(i): i for i in range(551)}}
    model = {'kind': 'nnue', 'path': '/private/model/weights.bin', 'bytes': 1305356, 'sha256': '7'*64}
    fallback = {'kind': 'material_fallback', 'path': None, 'bytes': None, 'sha256': None}
    def bundle(kind, root, build, rows, execution):
        role = 'candidate' if kind == 'candidate' else 'fallback'
        m = model if role == 'candidate' else fallback
        ids = {stage: kind+'-'+stage for stage in w.STAGES}
        terminal = put('/private/terminal/'+kind+'.json', {
            'schema': 'sekirei.white-view-evaluation-terminal.v1', 'status': 'observed-stopped',
            'role': role, 'model': m, 'run_ids': ids, 'executor_session_id': 123,
            'exit_code': 0, 'session_closed': True, 'tool_observed_reaped': True,
            'all_related_groups_observed_stopped': True, 'source_head': '1'*40})
        stages = {}
        for stage, count in w.STAGES.items():
            item = {'stage': stage, 'attempts': count, 'raw_sha_verified': count, 'lifecycle_verified': count,
                    'cleanup_ok': count, 'runner_exit_zero': count, 'technical_failures': 0,
                    'timeout_failures': 0, 'complete': True, 'final_used': False,
                    'max_reported_nodes': 1000000, 'fingerprint': 'f'*64}
            if stage in ('mae-pilot', 'top3-pilot'):
                item.update(stable_positions={'teacher': 17, 'sekirei': 17} if stage == 'mae-pilot' else 12,
                            unstable_positions={'teacher': 0, 'sekirei': 0} if stage == 'mae-pilot' else 0)
            stages[stage] = item
        comparable = {'execution_except_model': {'synthetic_runtime': execution}, 'occurrences': list(range(570)),
                      'teacher_results': teacher, 'teacher_E_cp': {str(i): i for i in range(266)},
                      'top3_occurrences': [str(i) for i in range(551)], 'top3_legal_moves_sha256': 'b'*64,
                      'top3_denominators': {g: row['top3_denominator'] for g, row in rows.items()},
                      'top3_teacher_bestmoves': {str(i): 'synthetic' for i in range(551)}}
        inventory = {root+'/runs/'+run_id: [] for run_id in ids.values()}
        immutable = {build['path']: p.small(build), terminal['path']: p.small(terminal),
                     contract['path']: p.small(contract), producer['path']: p.small(producer)}
        payload = {'schema': 'sekirei.white-view-reparsed-role-bundle.v1', 'status': 'complete',
                   'role': role, 'runtime': root, 'model': m, 'raw_reparsed': True,
                   'inputs_unchanged': True, 'cleanup_verified': True, 'executor_reaped': True,
                   'final_used': False, 'adoption_verified': False, 'stages': stages,
                   'comparable': comparable, 'per_game': rows, 'run_ids': ids,
                   'semantic_projections': copy.deepcopy(projections), 'projection_policy': copy.deepcopy(w.PROJECTION_POLICY),
                   'build_manifest': build, 'terminal_ref': terminal,
                   'audit_inputs_before': immutable, 'audit_inputs_after': copy.deepcopy(immutable), 'directory_inventory': inventory}
        w.validate_raw_bundle(payload, role, m, root)
        payload_ref = put('/private/audit/'+kind+'/payload.json', payload)
        receipt = {'schema': 'sekirei.white-view-full-role-audit.v1', 'status': 'complete', 'role': role,
                   'runtime': root, 'model': m, 'payload': payload_ref, 'build_manifest': build,
                   'terminal_ref': terminal, 'attempts': 1829, 'raw_reparsed': True, 'inputs_unchanged': True,
                   'cleanup_verified': True, 'executor_reaped': True, 'final_used': False, 'adoption_verified': False}
        ref = put('/private/audit/'+kind+'/receipt.json', receipt)
        return dict(payload, audit_ref=ref), ref
    old, old_ref = bundle('old', '/private/old', old_build, baseline_rows, 'old')
    base, base_ref = bundle('baseline', w.RUNTIME, new_build, baseline_rows, 'new')
    candidate, candidate_ref = bundle('candidate', w.RUNTIME, new_build, candidate_rows, 'new')
    pins = {'old_audit': old_ref, 'new_audit': base_ref, 'old_build': old_build,
            'new_build': new_build, 'old_runtime': '/private/old'}
    bridge = w.fallback_bridge(old, base, pins)
    bridge_ref = put('/private/bridge.json', bridge)
    comparison = w.compare_same_runtime(base, candidate, bridge, bridge_ref, model)
    comparison.update(baseline_audit=base_ref, candidate_audit=candidate_ref)
    comparison_ref = put('/private/comparison.json', comparison)
    request = {'schema': 'sekirei.white-view-public-projection-request.v1', 'status': 'frozen-before-public-projection',
               'comparison': comparison_ref, 'bridge': bridge_ref, 'old_audit': old_ref,
               'baseline_audit': base_ref, 'candidate_audit': candidate_ref, 'contract_source': contract,
               'comparison_producer': producer, 'worker_source': worker, 'source_head': '1'*40,
               'source_inputs': copy.deepcopy(identities), 'supplemental_evidence': {k: None for k in p.SUPPLEMENTAL},
               'output': '/private/new-public-output'}
    verified = {'comparison': comparison, 'bridge': bridge, 'old': old, 'baseline': base, 'candidate': candidate,
                'new_build': new_build_doc, 'old_build': old_build_doc, 'closure_rehashed': True}
    return w, request, verified, docs


class MemoryReader:
    def __init__(self, expected, docs): self.expected = copy.deepcopy(expected); self.docs = copy.deepcopy(docs)
    def current(self):
        for path, doc in self.docs.items():
            raw = (json.dumps(doc, sort_keys=True, allow_nan=False)+'\n').encode()
            p.require(p.exact(self.expected[path], {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}), 'synthetic hash mismatch')
        return copy.deepcopy(self.expected)
    def document(self, ref):
        p.require(p.exact(self.expected.get(ref['path']), p.small(ref)), 'synthetic external ref mismatch')
        return copy.deepcopy(self.docs[ref['path']])
    def membership(self, inventory):
        p.require(len(inventory) == 4 and all(v == [] for v in inventory.values()), 'synthetic membership differs')


class ProjectionTests(unittest.TestCase):
    def project(self, delta=-100, hits_delta=0):
        w, request, verified, docs = fixture(delta, hits_delta)
        checked = p.validate_private_bundle(MemoryReader(request['source_inputs'], docs), request, w)
        return p.project_public(request, checked)

    def test_normal_complete_producer_shape_and_three_public_files(self):
        public = self.project()
        self.assertTrue(public['decision']['adopt'])
        self.assertEqual(public['schema'], 'sekirei.white-view-public-comparison.v1')
        self.assertEqual([r['game'] for r in public['per_game']], ['G1', 'G2', 'G3', 'G4', 'G5'])
        self.assertEqual(p.fraction(public['headline']['candidate_exact']['mae']), 900)
        md, svg = p.render_markdown(public), p.render_svg(public)
        for text in (json.dumps(public), md, svg):
            for forbidden in ('/private/', '/home/', 'development-01', 'weights.bin', 'inputmaps', 'sfen ', 'teacher_cp', 'raw_log'):
                self.assertNotIn(forbidden, text)
        self.assertIn('MAE (cp)', svg); self.assertIn('Top3 (%)', svg)
        self.assertEqual(svg.count('<rect class='), 22)

    def test_valid_nonadoption_preserves_strict_joint_rule(self):
        for args in ((0, 0), (-100, -1), (100, 1)):
            with self.subTest(args=args): self.assertFalse(self.project(*args)['decision']['adopt'])

    def test_pending_is_never_success_and_provided_refs_are_hash_only(self):
        public = self.project()
        for row in public['supplemental_evidence'].values():
            self.assertFalse(row['provided']); self.assertIsNone(row['sha256'])
            self.assertFalse(row['proof_body_independently_verified_by_publisher'])
        w, request, verified, docs = fixture()
        request['supplemental_evidence']['core'] = request['comparison']
        # Reference-only deliberately makes no interpretation of an unknown proof body.
        row = p.project_public(request, verified)['supplemental_evidence']['core']
        self.assertEqual(row['publisher_validation'], 'sha_reference_only')
        self.assertFalse(row['proof_body_independently_verified_by_publisher'])

    def test_exact_fraction_cannot_be_replaced_by_display_rounding(self):
        w, request, verified, docs = fixture(0)
        verified['candidate']['per_game']['development-01']['absolute_error_sum_cp'] -= 1
        c = p.metrics(verified['candidate']['per_game'])
        verified['comparison'].update(candidate_exact=c, mae_improved=True, adopt=True)
        public = p.project_public(request, verified)
        self.assertEqual(p.fraction(c['mae']), Fraction(199999, 200))
        self.assertTrue(public['decision']['adopt'])

    def test_wrong_joint_bool_and_noncanonical_ratio_rejected(self):
        w, request, verified, docs = fixture()
        verified['comparison']['adopt'] = 1
        with self.assertRaises(ValueError): p.project_public(request, verified)
        with self.assertRaises(ValueError): p.fraction({'numerator': 2, 'denominator': 2})
        with self.assertRaises(ValueError): p.fraction({'numerator': True, 'denominator': 1})

    def test_strict_check_producer_shape_all_eight_and_zero_significance(self):
        w, request, verified, docs = fixture()
        for name in p.CHECKS:
            bad = copy.deepcopy(verified)
            bad['comparison']['checks'][name]['equal'] = 1
            with self.subTest(name=name), self.assertRaises(ValueError): p.project_public(request, bad)
        for mutation in (lambda d: d['comparison']['checks']['occurrences'].update(mismatch_count=0),
                         lambda d: d['comparison']['checks']['teacher_results'].update(mismatch_count=False),
                         lambda d: d['comparison']['checks'].pop('teacher_E_cp')):
            bad = copy.deepcopy(verified); mutation(bad)
            with self.assertRaises(ValueError): p.project_public(request, bad)

    def test_wrong_counts_schema_build_ref_and_input_closure_rejected(self):
        w, request, verified, docs = fixture()
        mutations = (lambda d: d['candidate']['per_game']['development-01'].update(e_count=True),
                     lambda d: d['candidate']['per_game']['development-01'].update(top3_denominator=81),
                     lambda d: d['comparison'].update(schema='sekirei.formal-comparison.v1'),
                     lambda d: d['candidate'].update(build_manifest={**d['candidate']['build_manifest'], 'sha256': '0'*64}),
                     lambda d: d.update(closure_rehashed=False))
        for mutation in mutations:
            bad = copy.deepcopy(verified); mutation(bad)
            with self.assertRaises(ValueError): p.project_public(request, bad)

    def test_receipt_ref_sourcehead_terminal_and_hash_modified_fail(self):
        w, request, verified, docs = fixture()
        reader = MemoryReader(request['source_inputs'], docs)
        reader.docs[request['comparison']['path']]['adopt'] = False
        with self.assertRaises(ValueError): p.validate_private_bundle(reader, request, w)
        bad = copy.deepcopy(request); bad['source_head'] = '2'*40
        with self.assertRaises(ValueError): p.validate_private_bundle(MemoryReader(bad['source_inputs'], docs), bad, w)
        bad = copy.deepcopy(request); bad['source_inputs'].pop(request['baseline_audit']['path'])
        with self.assertRaises(ValueError): p.validate_request(bad)

    def test_private_extra_payload_never_projects_and_bad_stage_string_fails_review(self):
        w, request, verified, docs = fixture()
        verified['candidate']['private_secret'] = {'sfen': 'sfen secret', 'labels': '/private/labels.json'}
        public = p.project_public(request, verified)
        self.assertNotIn('private_secret', public)
        verified['candidate']['stages']['mae']['fingerprint'] = '/home/server/secret'
        with self.assertRaises(ValueError): p.project_public(request, verified)

    def test_json_duplicate_boolsha_nonfinite_request_extra_fail_closed(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e1000}'):
            with self.assertRaises(ValueError): p.strict_json(raw)
        with self.assertRaises(ValueError): p.sha(True)
        w, request, verified, docs = fixture(); request['private_path'] = '/private/raw'
        with self.assertRaises(ValueError): p.validate_request(request)

    def test_actual_entrypoints_are_disabled_before_io_import_or_argparse(self):
        calls = ((p.main, ()), (p.run, ({}, 'x')), (p.FrozenReader, ({},)),
                 (p.write_public, ('/private/output', {}, [])), (p.load_contract, (None, {})))
        with (mock.patch.object(Path, 'read_bytes', side_effect=AssertionError('actual I/O')),
              mock.patch.object(p, 'load_contract', side_effect=AssertionError('actual import'))):
            for fn, args in calls:
                with self.subTest(fn=fn.__name__), self.assertRaisesRegex(ValueError, 'before I/O'):
                    fn(*args)

    def test_toy_reader_detects_symlink_hash_change_and_new_raw_membership(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(p, 'PROTOTYPE_ONLY', False):
            root = Path(directory); item = root/'toy.txt'; item.write_bytes(b'public toy')
            ref = {'path': str(item), 'bytes': 10, 'sha256': hashlib.sha256(b'public toy').hexdigest()}
            reader = p.FrozenReader({str(item): p.small(ref)})
            self.assertEqual(reader.read(ref), b'public toy')
            reader.membership({str(root): ['toy.txt']})
            (root/'extra.txt').write_bytes(b'toy')
            with self.assertRaises(ValueError): reader.membership({str(root): ['toy.txt']})
            item.write_bytes(b'changed toy')
            with self.assertRaises(ValueError): reader.read(ref)
            link = root/'link'; link.symlink_to(item)
            lr = {'path': str(link), 'bytes': 10, 'sha256': ref['sha256']}
            with self.assertRaises(ValueError): p.FrozenReader({str(link): p.small(lr)}).read(lr)

    def test_toy_writer_is_three_files_no_overwrite_and_input_nonoverlap(self):
        public = self.project()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(p, 'PROTOTYPE_ONLY', False):
            root = Path(directory); output = root/'public'
            result = p.write_public(str(output), public, [])
            self.assertEqual(set(result), set(p.FILES)); self.assertEqual({x.name for x in output.iterdir()}, set(p.FILES))
            for name, ident in result.items():
                raw = (output/name).read_bytes()
                self.assertEqual(ident, {'bytes':len(raw), 'sha256':hashlib.sha256(raw).hexdigest()})
            with self.assertRaises(ValueError): p.write_public(str(output), public, [])
            with self.assertRaises(ValueError): p.write_public(str(root/'overlap'), public, [str(root)])

    def test_new_child_of_audited_raw_root_is_rejected_before_write(self):
        _, request, verified, _ = fixture()
        public = p.project_public(request, verified)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(p, 'PROTOTYPE_ONLY', False):
            root = Path(directory); raw_root = root/'audited-raw'; raw_root.mkdir()
            original = raw_root/'synthetic.log'; original.write_bytes(b'synthetic only')
            before = list(raw_root.iterdir())
            for role in ('old', 'baseline', 'candidate'):
                with self.subTest(role=role):
                    toy = copy.deepcopy(verified)
                    toy[role]['directory_inventory'] = {str(raw_root): ['synthetic.log']}
                    protected = p.output_protected_paths(request, {'path':str(root/'request.json')}, toy)
                    output = raw_root/'new-public-child'
                    with self.assertRaises(ValueError): p.write_public(str(output), public, protected)
                    self.assertFalse(output.exists())
                    self.assertEqual(list(raw_root.iterdir()), before)
                    self.assertEqual(original.read_bytes(), b'synthetic only')


if __name__ == '__main__': unittest.main()
