"""Disabled SOURCE prototype: strict W receipts to three aggregate public files.

The publisher hashes the externally frozen closure, rechecks receipt envelopes,
bridge/check digests and exact arithmetic. It does not reparse USI raw evidence,
verify compiler/runtime identity anew, run engines, or validate supplemental
core/numeric/incremental/stop/NAS proof bodies. Those remain independent Root
steps; their optional references are explicitly labelled hash-reference-only.
Actual read/write/import entry points remain closed before I/O.
"""
from fractions import Fraction
import argparse
import copy
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import stat
import types

PROTOTYPE_ONLY = True
MODE = 'white-view-paired-linear-constrained-ridge1-l1-39p5-v1'
PLAN = '1f274b187a96b16674814c51f77cb402ca8a80e6a7e5d738a5caf9354127e836'
STAGES = {'mae-pilot': 102, 'mae': 1140, 'top3-pilot': 36, 'top3': 551}
GAMES = ('development-01', 'development-02', 'development-03', 'development-04', 'development-05')
GAME_LIMITS = dict(zip(GAMES, (86, 126, 127, 108, 123)))
CHECKS = {'execution_except_model', 'occurrences', 'teacher_results', 'teacher_E_cp',
          'top3_occurrences', 'top3_legal_moves_sha256', 'top3_denominators', 'top3_teacher_bestmoves'}
DICT_CHECKS = {'execution_except_model', 'teacher_results', 'teacher_E_cp', 'top3_denominators', 'top3_teacher_bestmoves'}
SUPPLEMENTAL = {'numeric', 'core', 'incremental', 'model_gate', 'stop', 'nas'}
FILES = ('comparison.json', 'comparison.md', 'comparison.svg')


def require(value, message):
    if not value:
        raise ValueError(message)


def runtime_guard():
    require(PROTOTYPE_ONLY is False, 'SOURCE ONLY publisher disabled before I/O/import/write')


def exact(a, b):
    if type(a) is not type(b):
        return False
    if type(a) is dict:
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if type(a) in (list, tuple):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


def sha(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'strict SHA required')
    return value


def fullref(value):
    require(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'}, 'exact fullref required')
    require(type(value['path']) is str and Path(value['path']).is_absolute()
            and '..' not in Path(value['path']).parts, 'absolute canonical-form path required')
    require(type(value['bytes']) is int and value['bytes'] >= 0, 'strict file size required')
    sha(value['sha256'])
    return value


def small(ref):
    fullref(ref)
    return {key: ref[key] for key in ('bytes', 'sha256')}


def fraction(value):
    require(type(value) is dict and set(value) == {'numerator', 'denominator'}, 'exact ratio fields required')
    require(type(value['numerator']) is int and type(value['denominator']) is int
            and value['denominator'] > 0, 'strict rational integers required')
    result = Fraction(value['numerator'], value['denominator'])
    require(value == ratio(result), 'canonical reduced rational required')
    return result


def ratio(value):
    return {'numerator': value.numerator, 'denominator': value.denominator}


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def bad(value):
        raise ValueError('nonfinite JSON number: '+value)
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)
    def walk(item):
        if type(item) is float:
            require(math.isfinite(item), 'nonfinite JSON float')
        elif type(item) is dict:
            for part in item.values(): walk(part)
        elif type(item) is list:
            for part in item: walk(part)
    walk(value)
    return value


def identity_map(value):
    require(type(value) is dict and value, 'nonempty external input map required')
    for path, record in value.items():
        fullref({'path': path, **record})
        require(set(record) == {'bytes', 'sha256'}, 'exact input identity required')
    return value


def validate_request(value):
    keys = {'schema', 'status', 'comparison', 'bridge', 'old_audit', 'baseline_audit',
            'candidate_audit', 'contract_source', 'comparison_producer', 'worker_source',
            'source_head', 'source_inputs', 'supplemental_evidence', 'output'}
    require(type(value) is dict and set(value) == keys and
            value['schema'] == 'sekirei.white-view-public-projection-request.v1' and
            value['status'] == 'frozen-before-public-projection', 'dedicated exact request required')
    require(type(value['source_head']) is str and re.fullmatch('[0-9a-f]{40}', value['source_head']), 'full source HEAD required')
    inputs = identity_map(value['source_inputs'])
    for key in ('comparison', 'bridge', 'old_audit', 'baseline_audit', 'candidate_audit',
                'contract_source', 'comparison_producer', 'worker_source'):
        ref = fullref(value[key])
        require(exact(inputs.get(ref['path']), small(ref)), 'external primary ref missing: '+key)
    require(Path(value['contract_source']['path']).name == 'white_view_evaluation.py' and
            Path(value['comparison_producer']['path']).name == 'white_view_operations.py', 'dedicated W producer source required')
    extra = value['supplemental_evidence']
    require(type(extra) is dict and set(extra) == SUPPLEMENTAL, 'explicit supplemental known/pending set required')
    for key, ref in extra.items():
        if ref is not None:
            fullref(ref)
            require(exact(inputs.get(ref['path']), small(ref)), 'supplemental external ref missing: '+key)
    require(type(value['output']) is str and Path(value['output']).is_absolute()
            and '..' not in Path(value['output']).parts, 'fresh absolute output required')
    return value


class FrozenReader:
    """Runtime-only reader: all parsed bytes and closure membership externally pin."""
    def __init__(self, expected):
        runtime_guard()
        self.expected = copy.deepcopy(identity_map(expected))
        self.seen = {}

    def pin(self, ref):
        runtime_guard(); fullref(ref)
        require(exact(self.expected.get(ref['path']), small(ref)), 'read outside frozen closure')
        path = Path(ref['path'])
        require(path.resolve(strict=True) == path and not path.is_symlink()
                and stat.S_ISREG(path.stat().st_mode), 'canonical regular input required')
        digest = hashlib.sha256(); size = 0
        with path.open('rb') as stream:
            before = os.fstat(stream.fileno())
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                size += len(chunk); digest.update(chunk)
            after = os.fstat(stream.fileno())
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'input changed while reading')
        require(size == ref['bytes'] == after.st_size and digest.hexdigest() == ref['sha256'], 'current bytes differ from external SHA')
        self.seen[ref['path']] = small(ref)
        return copy.deepcopy(self.seen[ref['path']])

    def read(self, ref):
        runtime_guard(); self.pin(ref)
        raw = Path(ref['path']).read_bytes()
        require(len(raw) == ref['bytes'] and hashlib.sha256(raw).hexdigest() == ref['sha256'], 'parsed bytes differ from external SHA')
        self.pin(ref)
        return raw

    def document(self, ref):
        return strict_json(self.read(ref))

    def current(self):
        runtime_guard()
        for path, record in self.expected.items(): self.pin({'path': path, **record})
        require(exact(self.seen, self.expected), 'full input closure not rehashed')
        return copy.deepcopy(self.seen)

    def membership(self, inventory):
        runtime_guard(); require(type(inventory) is dict and inventory, 'four directory inventory required')
        for directory, expected in inventory.items():
            root = Path(directory)
            require(root.is_absolute() and root.resolve(strict=True) == root
                    and root.is_dir() and not root.is_symlink(), 'canonical raw root required')
            children = []
            for child in sorted(root.rglob('*')):
                require(not child.is_symlink(), 'symlink in raw inventory')
                if child.is_file():
                    require(str(child) in self.expected, 'new raw file outside frozen closure')
                    children.append(str(child.relative_to(root)))
                else:
                    require(child.is_dir(), 'special file in raw inventory')
            require(exact(children, expected), 'raw file membership changed')


def load_contract(reader, reference):
    runtime_guard()
    raw = reader.read(reference)
    module = types.ModuleType('_white_public_projection_contract')
    module.__file__ = reference['path']
    exec(compile(raw, reference['path'], 'exec'), module.__dict__)
    require(module.MODE == MODE and module.PLAN_SHA256 == PLAN
            and set(module.CHECKS) == CHECKS and exact(module.STAGES, STAGES), 'source contract is not fixed W mode')
    return module


def read_audit(reader, reference, contract, source_head=None):
    receipt = reader.document(reference)
    fixed = {'schema': 'sekirei.white-view-full-role-audit.v1', 'status': 'complete',
             'attempts': 1829, 'raw_reparsed': True, 'inputs_unchanged': True,
             'cleanup_verified': True, 'executor_reaped': True, 'final_used': False, 'adoption_verified': False}
    require(type(receipt) is dict and all(exact(receipt.get(k), v) for k, v in fixed.items()), 'incomplete W audit envelope')
    payload = reader.document(fullref(receipt['payload']))
    require('audit_ref' not in payload, 'one-way audit payload required')
    for key in ('role', 'runtime', 'model', 'build_manifest', 'terminal_ref'):
        require(exact(payload[key], receipt[key]), 'audit envelope subject differs')
    contract.validate_raw_bundle(payload, receipt['role'], receipt['model'], receipt['runtime'])
    before = identity_map(payload['audit_inputs_before'])
    require(exact(before, payload['audit_inputs_after']), 'audit immutable maps differ')
    require(all(exact(reader.expected.get(path), rec) for path, rec in before.items()), 'transitive audit inputs missing')
    require(type(payload['run_ids']) is dict and set(payload['run_ids']) == set(STAGES), 'four explicit stage IDs required')
    require(set(payload['directory_inventory']) == {
        str(Path(payload['runtime'])/'runs'/run_id) for run_id in payload['run_ids'].values()}, 'all four raw roots required')
    reader.membership(payload['directory_inventory'])
    terminal = reader.document(fullref(payload['terminal_ref']))
    term = {'schema': 'sekirei.white-view-evaluation-terminal.v1', 'status': 'observed-stopped',
            'exit_code': 0, 'session_closed': True, 'tool_observed_reaped': True,
            'all_related_groups_observed_stopped': True, 'role': payload['role'],
            'model': payload['model'], 'run_ids': payload['run_ids']}
    require(all(exact(terminal.get(k), v) for k, v in term.items()) and
            type(terminal.get('executor_session_id')) is int and terminal['executor_session_id'] > 0,
            'normal external terminal required')
    if source_head is not None: require(terminal.get('source_head') == source_head, 'current formal source HEAD differs')
    result = copy.deepcopy(payload); result['audit_ref'] = copy.deepcopy(reference)
    return result


def validate_private_bundle(reader, request, contract):
    """Hash-only raw closure + producer receipt reproduction, never USI reparse."""
    validate_request(request)
    before = reader.current()
    old = read_audit(reader, request['old_audit'], contract)
    base = read_audit(reader, request['baseline_audit'], contract, request['source_head'])
    candidate = read_audit(reader, request['candidate_audit'], contract, request['source_head'])
    bridge = reader.document(request['bridge'])
    pins = {'old_audit': request['old_audit'], 'new_audit': request['baseline_audit'],
            'old_build': old['build_manifest'], 'new_build': base['build_manifest'], 'old_runtime': old['runtime']}
    require(exact(contract.fallback_bridge(old, base, pins), bridge), 'fallback semantic/exact bridge does not reproduce')
    comparison = reader.document(request['comparison'])
    reproduced = contract.compare_same_runtime(base, candidate, bridge, request['bridge'], candidate['model'])
    reproduced.update(baseline_audit=request['baseline_audit'], candidate_audit=request['candidate_audit'])
    require(exact(comparison, reproduced), 'complete W comparison does not reproduce')
    new_build = reader.document(fullref(base['build_manifest']))
    old_build = reader.document(fullref(old['build_manifest']))
    require(new_build.get('schema') == 'sekirei.white-view-runtime-build.v1' and new_build.get('status') == 'complete', 'identified new build envelope required')
    require(new_build['architecture']['native_magic'] == 'SEKIRW03' and
            new_build['architecture']['compile_feature'] == 'nnue_white_view_aux_tied', 'dedicated new architecture required')
    for key in ('sekirei', 'yaneuraou'):
        ref = fullref(new_build['binaries'][key])
        require(exact(reader.expected.get(ref['path']), small(ref)), 'new binary identity absent from closure')
    require(exact(reader.current(), before), 'public projection inputs changed')
    for bundle in (old, base, candidate): reader.membership(bundle['directory_inventory'])
    return {'comparison': comparison, 'bridge': bridge, 'old': old, 'baseline': base, 'candidate': candidate,
            'new_build': new_build, 'old_build': old_build, 'closure_rehashed': True}


def validate_checks(checks):
    require(type(checks) is dict and set(checks) == CHECKS, 'exact eight checks required')
    for name, row in checks.items():
        keys = {'equal', 'baseline_sha256', 'candidate_sha256'} | ({'mismatch_count'} if name in DICT_CHECKS else set())
        require(type(row) is dict and set(row) == keys and row['equal'] is True, 'typed identity check differs')
        require(sha(row['baseline_sha256']) == sha(row['candidate_sha256']), 'identity SHA differs')
        if name in DICT_CHECKS:
            require(type(row['mismatch_count']) is int and row['mismatch_count'] == 0, 'identity mismatch count')


def metrics(rows):
    require(type(rows) is dict and set(rows) == set(GAMES), 'fixed five games required')
    mae = top = Fraction(0); e = n = 0
    for game in GAMES:
        row = rows[game]
        require(type(row) is dict and set(row) == {'absolute_error_sum_cp', 'e_count', 'top3_hits', 'top3_denominator'}, 'strict integer aggregates required')
        require(all(type(x) is int for x in row.values()) and row['absolute_error_sum_cp'] >= 0
                and 0 < row['e_count'] <= GAME_LIMITS[game]
                and 0 <= row['top3_hits'] <= row['top3_denominator'] <= GAME_LIMITS[game]
                and row['top3_denominator'] > 0, 'invalid per-game integer count')
        mae += Fraction(row['absolute_error_sum_cp'], row['e_count'])/5
        top += Fraction(row['top3_hits'], row['top3_denominator'])/5
        e += row['e_count']; n += row['top3_denominator']
    require((e, n) == (266, 551), 'Teacher-E266/Top3551 incomplete')
    return {'mae': ratio(mae), 'top3': ratio(top)}


def project_public(request, verified):
    """Pure allowlist projection; only accepts independently validated documents."""
    validate_request(request)
    comparison, base, candidate = (verified[k] for k in ('comparison', 'baseline', 'candidate'))
    require(verified.get('closure_rehashed') is True, 'runtime closure gate required before projection')
    require(comparison.get('schema') == 'sekirei.white-view-same-runtime-comparison.v1' and
            comparison.get('status') == 'complete' and comparison.get('comparison_valid') is True
            and comparison.get('final_used') is False, 'complete valid development comparison required')
    validate_checks(comparison['checks'])
    b, c = metrics(base['per_game']), metrics(candidate['per_game'])
    require(exact(b, comparison['baseline_exact']) and exact(c, comparison['candidate_exact']), 'exact public aggregate differs')
    bm, cm, bt, ct = (fraction(x) for x in (b['mae'], c['mae'], b['top3'], c['top3']))
    rule = {'mae_improved': cm < bm, 'top3_preserved': ct >= bt, 'adopt': cm < bm and ct >= bt}
    require(all(exact(comparison.get(k), v) for k, v in rule.items()), 'joint exact decision differs')
    require(exact(base['build_manifest'], candidate['build_manifest']) and
            exact(comparison.get('baseline_audit'), request['baseline_audit']) and
            exact(comparison.get('candidate_audit'), request['candidate_audit']) and
            exact(comparison.get('fallback_bridge'), request['bridge']), 'external compared subjects differ')
    model = candidate['model']; fullref({k: model[k] for k in ('path', 'bytes', 'sha256')})
    require(model['kind'] == 'nnue' and model['bytes'] == 1305356, 'native03 model identity required')
    rows = []
    for ordinal, game in enumerate(GAMES, 1):
        br, cr = base['per_game'][game], candidate['per_game'][game]
        require(br['e_count'] == cr['e_count'] and br['top3_denominator'] == cr['top3_denominator'], 'per-game denominators differ')
        rows.append({'game': 'G'+str(ordinal), 'teacher_exact_count': br['e_count'],
                     'top3_denominator': br['top3_denominator'],
                     'baseline_error_sum_cp': br['absolute_error_sum_cp'], 'candidate_error_sum_cp': cr['absolute_error_sum_cp'],
                     'baseline_top3_hits': br['top3_hits'], 'candidate_top3_hits': cr['top3_hits'],
                     'baseline_mae': ratio(Fraction(br['absolute_error_sum_cp'], br['e_count'])),
                     'candidate_mae': ratio(Fraction(cr['absolute_error_sum_cp'], cr['e_count'])),
                     'baseline_top3': ratio(Fraction(br['top3_hits'], br['top3_denominator'])),
                     'candidate_top3': ratio(Fraction(cr['top3_hits'], cr['top3_denominator']))})
    build = verified['new_build']; old_build = verified['old_build']
    old_sha = sha(old_build['binaries']['sekirei']['sha256'])
    refs = {name: request[key]['sha256'] for name, key in
            (('comparison', 'comparison'), ('fallback_bridge', 'bridge'), ('old_fallback_audit', 'old_audit'),
             ('new_fallback_audit', 'baseline_audit'), ('candidate_audit', 'candidate_audit'),
             ('comparison_contract_source', 'contract_source'), ('comparison_producer_source', 'comparison_producer'),
             ('publisher_source', 'worker_source'))}
    refs.update(old_fallback_terminal=verified['old']['terminal_ref']['sha256'],
                new_fallback_terminal=base['terminal_ref']['sha256'], candidate_terminal=candidate['terminal_ref']['sha256'])
    supplemental = {name: {'provided': ref is not None, 'sha256': None if ref is None else ref['sha256'],
                           'publisher_validation': 'not_provided' if ref is None else 'sha_reference_only',
                           'proof_body_independently_verified_by_publisher': False}
                    for name, ref in sorted(request['supplemental_evidence'].items())}
    result = {'schema': 'sekirei.white-view-public-comparison.v1', 'status': 'complete',
              'candidate': MODE, 'source_head': request['source_head'], 'plan_sha256': PLAN,
              'model': {'kind': 'nnue', 'native_magic': 'SEKIRW03', 'bytes': model['bytes'], 'sha256': model['sha256']},
              'conditions': {'sekirei': 'v0.3.39', 'compile_feature': 'nnue_white_view_aux_tied',
                             'teacher': 'Suisho Concerto 202512 (Suisho11beta)', 'requested_nodes': 1000000,
                             'max_reported_nodes': 1010000, 'threads': 1, 'hash_mib': 128,
                             'mae_multipv': 1, 'candidate_top3_multipv': 3, 'games': 5,
                             'occurrences': 570, 'teacher_exact_count': 266, 'top3_eligible_count': 551,
                             'aggregation': 'equal mean of five per-game metrics', 'nnue_output': 'absolute', 'final_used': False},
              'identities': {'new_build_manifest_sha256': base['build_manifest']['sha256'],
                             'old_build_manifest_sha256': verified['bridge']['old_build']['sha256'],
                             'new_sekirei_binary_sha256': sha(build['binaries']['sekirei']['sha256']),
                             'old_sekirei_binary_sha256': old_sha,
                             'teacher_binary_sha256': sha(build['binaries']['yaneuraou']['sha256'])},
              'decision': {'comparison_valid': True, **rule,
                           'rule': 'MAE strictly lower AND Top3 non-decreasing; exact five-game rational comparison'},
              'headline': {'baseline_exact': b, 'candidate_exact': c,
                           'mae_delta_cp': float(cm-bm), 'top3_delta_percentage_points': float(100*(ct-bt))},
              'per_game': rows, 'reference_checks': copy.deepcopy(comparison['checks']),
              'fallback_bridge': {'semantic_and_exact_integer_metrics_equal': True,
                                  'execution_identities_equal': False, 'old_results_reused_as_new': False},
              'stage_evidence': {role: {stage: {key: bundle['stages'][stage][key] for key in
                                   ('attempts', 'raw_sha_verified', 'lifecycle_verified', 'cleanup_ok', 'runner_exit_zero', 'max_reported_nodes', 'fingerprint')}
                                   for stage in STAGES} for role, bundle in (('baseline', base), ('candidate', candidate))},
              'verification_sha256': refs, 'supplemental_evidence': supplemental,
              'scope': {'private_inputs_rehashed_unchanged': True, 'receipt_arithmetic_reproduced': True,
                        'usi_raw_reparsed_by_publisher': False, 'runtime_rebuilt_or_reverified_by_publisher': False,
                        'supplemental_proof_bodies_verified_by_publisher': False,
                        'final_data_used': False, 'general_playing_strength_proven': False}}
    review_public(result)
    return result


def review_public(value):
    """Supplemental text barrier; primary boundary is explicit key projection."""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    forbidden = ('/home/', '/mnt/', '/tmp/', 'sfen ', 'startpos moves', 'raw_log', 'input_receipt',
                 'inputmaps', 'inputs_before', 'inputs_after', 'source_game_id', 'development-0',
                 'position_sha256', 'candidate.json', 'evaluation.json', 'weights.bin')
    # The CHECK name teacher_results is permitted only as a strict digest row.
    require(not any(x in encoded for x in forbidden), 'private content in public projection')
    require(not re.search(r'(?:^|[\s="\'])/[A-Za-z0-9_.-]+(?:/|$)', encoded), 'absolute path in public projection')
    return encoded


def render_markdown(public):
    review_public(public)
    h = public['headline']; b, c = h['baseline_exact'], h['candidate_exact']
    verdict = '採用条件を満たした。' if public['decision']['adopt'] else '有効な正式比較で、採用条件を満たさなかった。'
    lines = ['# White-view 固定development比較', '', verdict, '',
             '| 指標 | 同じ新binaryの駒得fallback | 候補 |', '| --- | ---: | ---: |',
             f"| MAE | {float(fraction(b['mae'])):.3f} cp | {float(fraction(c['mae'])):.3f} cp |",
             f"| Top3 | {100*float(fraction(b['top3'])):.4f}% | {100*float(fraction(c['top3'])):.4f}% |", '',
             '採否は5局等重みの有理数で、MAEの厳密低下とTop3の非低下を同時に要求した。表示の丸め値では判定しない。', '',
             '100万ノード指定、Threads=1、Hash=128 MiB。MAEはMultiPV=1、Top3は別runでMultiPV=3。',
             '旧stockと新fallbackのsemanticと整数集計は一致した。binary/buildの相違を保持し、旧結果を新測定へ再利用していない。', '',
             '| 局 | Teacher-E | MAE: 基準 / 候補 | Top3 hit: 基準 / 候補 / 分母 |',
             '| --- | ---: | ---: | ---: |']
    for row in public['per_game']:
        lines.append(f"| {row['game']} | {row['teacher_exact_count']} | {float(fraction(row['baseline_mae'])):.3f} / {float(fraction(row['candidate_mae'])):.3f} | {row['baseline_top3_hits']} / {row['candidate_top3_hits']} / {row['top3_denominator']} |")
    lines += ['', '[正確な集計・条件・検証SHA](comparison.json) · [2指標の集計グラフ](comparison.svg)', '',
              '固定development 5局の結果であり、最終データや一般的な棋力の改善を示さない。補助的な技術・停止・NAS参照はJSONに提供状態とSHAを示す。publisher自身によるproof body検証成功には読み替えない。', '']
    text = '\n'.join(lines)
    review_public({'markdown': text})
    return text


def render_svg(public):
    """Two aggregate bar panels only; no ply/teacher score or position series."""
    review_public(public)
    rows = public['per_game']; out = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 390" width="960" height="390">',
           '<title>White-view aggregate MAE and Top3 comparison</title>',
           '<desc>Five ordinal game aggregates; no per-position values.</desc>',
           '<rect width="960" height="390" fill="white"/>',
           '<style>text{font-family:sans-serif;font-size:12px;fill:#172033}.base{fill:#64748b}.candidate{fill:#0369a1}.axis{stroke:#64748b}</style>']
    for panel, metric, title in ((0, 'mae', 'MAE (cp)'), (1, 'top3', 'Top3 (%)')):
        left = panel*480+55; y0 = 320; height = 260
        values = [(float(fraction(row['baseline_'+metric])), float(fraction(row['candidate_'+metric]))) for row in rows]
        scale = max(1.0, max(v for pair in values for v in pair)*1.08) if metric == 'mae' else 1.0
        out.append(f'<text x="{left}" y="26">{html.escape(title)}</text>')
        out.append(f'<line class="axis" x1="{left}" y1="60" x2="{left}" y2="{y0}"/>')
        out.append(f'<line class="axis" x1="{left}" y1="{y0}" x2="{left+395}" y2="{y0}"/>')
        for i, (old, new) in enumerate(values):
            x = left+18+i*77
            for j, val in enumerate((old, new)):
                h = height*val/scale
                display = val*100 if metric == 'top3' else val
                out.append(f'<rect class="{"base" if j == 0 else "candidate"}" x="{x+j*23}" y="{y0-h:.3f}" width="20" height="{h:.3f}"/>')
                out.append(f'<text x="{x+j*23}" y="{y0-h-5:.3f}" font-size="9">{display:.1f}</text>')
            out.append(f'<text x="{x+10}" y="342">G{i+1}</text>')
    out += ['<rect class="base" x="310" y="362" width="12" height="12"/><text x="328" y="372">new-binary material fallback</text>',
            '<rect class="candidate" x="550" y="362" width="12" height="12"/><text x="568" y="372">candidate</text>', '</svg>\n']
    text = '\n'.join(out); review_public({'svg': text})
    return text


def write_public(output, public, protected):
    runtime_guard()
    path = Path(output)
    require(path.is_absolute() and not os.path.lexists(path)
            and path.parent.resolve(strict=True) == path.parent and path.parent.is_dir(), 'fresh canonical output with existing parent required')
    for source in protected:
        source = Path(source)
        require(not path.is_relative_to(source) and not source.is_relative_to(path), 'output overlaps frozen private input')
    data = {'comparison.json': (json.dumps(public, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode(),
            'comparison.md': render_markdown(public).encode(), 'comparison.svg': render_svg(public).encode()}
    # All three public serializations are reviewed before first write.
    for raw in data.values(): review_public({'file': raw.decode()})
    path.mkdir(mode=0o700)
    result = {}
    for name in FILES:
        fd = os.open(path/name, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data[name]); stream.flush(); os.fsync(stream.fileno())
        result[name] = {'bytes': len(data[name]), 'sha256': hashlib.sha256(data[name]).hexdigest()}
    return result


def output_protected_paths(request, request_ref, verified):
    """Protect whole audited roots, including presently empty descendants."""
    return [*request["source_inputs"], request_ref["path"],
            *[root for role in ("old", "baseline", "candidate")
              for root in verified[role]["directory_inventory"]]]


def run(request_ref, expected_worker_sha256):
    runtime_guard()
    # Request itself is bound externally; all subsequent files are explicit.
    fullref(request_ref)
    bootstrap = FrozenReader({request_ref['path']: small(request_ref)})
    request = validate_request(bootstrap.document(request_ref))
    require(request['worker_source']['path'] == str(Path(__file__).resolve()) and
            request['worker_source']['sha256'] == sha(expected_worker_sha256), 'external publisher source identity differs')
    reader = FrozenReader(request['source_inputs'])
    reader.read(request['worker_source'])
    contract = load_contract(reader, request['contract_source'])
    verified = validate_private_bundle(reader, request, contract)
    public = project_public(request, verified)
    require(exact(reader.current(), request['source_inputs']), 'private closure changed before first output')
    # No completed publication receipt is emitted on partial output failure.
    result = write_public(request['output'], public, output_protected_paths(request, request_ref, verified))
    require(exact(reader.current(), request['source_inputs']), 'private closure changed during public output')
    for name in ('old', 'baseline', 'candidate'): reader.membership(verified[name]['directory_inventory'])
    bootstrap.current()
    return result


def main():
    runtime_guard()  # Even --help cannot read/import/write actual artifacts.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--expected-request-sha256', required=True)
    parser.add_argument('--expected-request-bytes', type=int, required=True)
    parser.add_argument('--expected-worker-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(run({'path': args.request, 'bytes': args.expected_request_bytes,
                          'sha256': args.expected_request_sha256}, args.expected_worker_sha256), indent=2))


if __name__ == '__main__': main()
