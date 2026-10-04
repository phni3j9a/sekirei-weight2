"""Pure, dedicated white-view postformal/stop/archive contracts; no I/O."""
from fractions import Fraction
from pathlib import PurePosixPath as P
import copy
import re

PROTOTYPE_ONLY = True
C = P('/home/server/.local/share/sekirei-weight2/campaign-17-autonomous-v1')
R = P('/home/server/worktrees/sekirei-weight2/issue-17-autonomous-weight-improvement')
N = C.parent/'suisho11beta-sekirei-v0.3.39-white-view-v1'
B = C.parent/'suisho11beta-sekirei-v0.3.39-v1'
T = C.parent/'training-15-v1'
Q = C.parent/'training-17-v1/white-view-paired-linear-constrained-ridge1-v1/run-v1'
FAN = C.parent/'training-17-v1/bounded-material-fanin509-trainer-v1'
MODE = 'white-view-paired-linear-constrained-ridge1-l1-39p5-v1'
HEAD = '6d099c79b4cac03390b9d861f11d05e083f0e66b'
FACTORY_SHA = 'a6f835348716585925bce2396cf5b8021835d44031648b7971cefb2914dea556'
ALLOW = C/'white-view-preexisting-opaque-service-allowlist-v1.json'
ALLOW_SHA = '0cb4e02dd29beb136fc3f20d492e4d1846f93ad5dcd22f3cfe312ab6d9e64098'
OUT = C/'white-view-formal-stopped-v1.json'
MODULE_NAMES = ('white_view_evaluation', 'white_view_runtime_io', 'white_view_production_ports', 'white_view_operations')
LOCKS = ((T/'.training.lock', True), (B/'.prepare.lock', False),
         (Q.parent/'.fit.lock', True), (FAN/'.build.lock', True),
         (FAN/'.training.lock', True), (N/'.build.lock', False),
         (N/'.prepare.lock', False), (B/'.benchmark.lock', True), (N/'.benchmark.lock', True))
NAS = P('/mnt/storage/NAS/sekirei-weight2')
HELPER = C.parent/'campaign-15-20261002/archive-helper.py'
HELPER_SHA = '5520a647ed77f25e47cd7208fd13ee9b6ae10d619a836c8b36709d33308a2816'
# The unchanged helper takes four locks; the parent must not hold duplicates.
ARCHIVE_HELPER_LOCKS = (T/'.training.lock', B/'.prepare.lock', B/'.benchmark.lock', FAN/'.training.lock')
ARCHIVE_OUTER_LOCKS = (Q.parent/'.fit.lock', FAN/'.build.lock', N/'.build.lock', N/'.prepare.lock', N/'.benchmark.lock')
REF_NAMES = {'comparison_terminal', 'comparison', 'comparison_request', 'candidate_preflight',
             'candidate_gate', 'bridge', 'old_audit', 'fallback_audit', 'candidate_audit',
             'candidate_terminal', 'factory_source', 'opaque_allowlist'}
FIXED_REF_PATHS = {
    'comparison_terminal': C/'white-view-comparison-executor-terminal-v1.json',
    'comparison': C/'white-view-candidate-same-runtime-comparison-v1.json',
    'comparison_request': C/'white-view-compare-request-v1.json',
    'candidate_preflight': C/'white-view-candidate-formal-launch-preflight-v1.json',
    'candidate_gate': C/'white-view-model-technical-gate-v1.json',
    'bridge': C/'white-view-fallback-semantic-bridge-v1.json',
    'old_audit': C/'white-view-old-fallback-full-raw-audit-v1/receipt.json',
    'fallback_audit': C/'white-view-fallback-full-raw-audit-v1/receipt.json',
    'candidate_audit': C/'white-view-candidate-full-raw-audit-v1/receipt.json',
    'candidate_terminal': C/'white-view-candidate-executor-terminal-v1.json',
    'opaque_allowlist': ALLOW}


def require(ok, message):
    if not ok: raise ValueError(message)


def exact(a, b):
    if type(a) is not type(b): return False
    if type(a) is dict: return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if type(a) in (list, tuple): return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


def sha(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'strict SHA256 required')
    return value


def path(value):
    require(type(value) is str and P(value).is_absolute() and str(P(value)) == value
            and '..' not in P(value).parts, 'canonical absolute path text required')
    return P(value)


def identity(value):
    require(type(value) is dict and set(value) == {'bytes', 'sha256'}
            and type(value['bytes']) is int and value['bytes'] >= 0, 'strict file identity required')
    sha(value['sha256']); return value


def fullref(value):
    require(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'}, 'strict fullref required')
    path(value['path']); identity({k: value[k] for k in ('bytes', 'sha256')}); return value


def filemap(value):
    require(type(value) is dict and value, 'nonempty immutable map required')
    for name, rec in value.items(): path(name); identity(rec)
    return value


def contains(values, rec):
    fullref(rec)
    require(exact(values.get(rec['path']), {k: rec[k] for k in ('bytes', 'sha256')}), 'reference absent or different in immutable map')


def fields(value, fixed):
    require(type(value) is dict and all(exact(value.get(k), v) for k, v in fixed.items()), 'typed document fields differ')


def validate_request(value, w):
    keys = {'schema', 'status', 'candidate', 'source_head', 'executor_session_id', 'comparison_session_id',
            'model', 'refs', 'gate_binding', 'inputs', 'source_helpers', 'modules', 'output', 'final_used', 'goal_complete'}
    require(type(value) is dict and set(value) == keys, 'exact white-view postformal request required')
    fields(value, {'schema': 'sekirei.white-view-postformal-request.v1',
        'status': 'frozen-after-reaped-comparison', 'candidate': MODE, 'source_head': HEAD,
        'output': str(OUT), 'final_used': False, 'goal_complete': False})
    for key in ('executor_session_id', 'comparison_session_id'):
        require(type(value[key]) is int and value[key] > 0, 'external positive session required')
    w.model_identity('candidate', value['model']); filemap(value['inputs'])
    require(type(value['refs']) is dict and set(value['refs']) == REF_NAMES, 'exact current control references required')
    for rec in value['refs'].values(): contains(value['inputs'], rec)
    refs = value['refs']
    for name, expected in FIXED_REF_PATHS.items():
        require(refs[name]['path'] == str(expected), 'fixed current record path differs: '+name)
    require(refs['factory_source']['sha256'] == FACTORY_SHA, 'fixed factory source differs')
    require(refs['opaque_allowlist']['path'] == str(ALLOW) and refs['opaque_allowlist']['sha256'] == ALLOW_SHA,
            'fixed preexisting-service authority differs')
    w.validate_candidate_binding(value['gate_binding'], value['inputs'])
    require(exact(refs['candidate_gate'], value['gate_binding']['gate']), 'current gate reference differs')
    require(exact({k: value['model'][k] for k in ('path', 'bytes', 'sha256')}, value['gate_binding']['native']),
            'current model differs from technical proof')
    require(type(value['modules']) is dict and set(value['modules']) == set(MODULE_NAMES), 'four actual evaluation source refs required')
    for name, rec in value['modules'].items():
        contains(value['inputs'], rec); require(P(rec['path']).name == name+'.py', 'actual module basename differs')
    require(type(value['source_helpers']) is dict and value['source_helpers'], 'complete frozen helper inventory required')
    for name, digest in value['source_helpers'].items():
        path(name); sha(digest); require(value['inputs'].get(name, {}).get('sha256') == digest, 'helper source absent')
    for rec in value['modules'].values():
        require(value['source_helpers'].get(rec['path']) == rec['sha256'], 'evaluation source not frozen as helper')
    return value


def validate_comparison_terminal(value, request):
    require(type(value) is dict and set(value) == {'schema', 'status', 'comparison_session_id', 'exit_code',
        'session_closed', 'tool_observed_reaped', 'all_related_groups_observed_stopped', 'comparison',
        'comparison_request', 'source_head', 'final_used', 'goal_complete'}, 'exact Root comparison terminal required')
    fields(value, {'schema': 'sekirei.white-view-comparison-terminal.v1', 'status': 'observed-stopped',
        'comparison_session_id': request['comparison_session_id'], 'exit_code': 0, 'session_closed': True,
        'tool_observed_reaped': True, 'all_related_groups_observed_stopped': True,
        'comparison': request['refs']['comparison'], 'comparison_request': request['refs']['comparison_request'],
        'source_head': HEAD, 'final_used': False, 'goal_complete': False})
    return value


def validate_documents(request, docs, audits, bridge, w, p):
    """Only dedicated documents; no previous-candidate schema or synthesized flags."""
    validate_request(request, w)
    require(type(docs) is dict and set(docs) == {'comparison_terminal', 'comparison', 'comparison_request',
        'candidate_preflight', 'candidate_gate', 'candidate_terminal', 'opaque_allowlist'}, 'exact postformal documents required')
    validate_comparison_terminal(docs['comparison_terminal'], request)
    pf = p.validate_launch_preflight(docs['candidate_preflight']); spec = pf['spec']
    require(spec['role'] == 'candidate' and exact(spec['model'], request['model'])
        and exact(pf['candidate_binding'], request['gate_binding']), 'candidate launch subject differs')
    require(exact(pf['repo_identity'], {'commit': HEAD, 'dirty': False, 'status_porcelain': ''}),
            'measurement source HEAD differs')
    for name, rec in spec['inputs'].items(): require(exact(request['inputs'].get(name), rec), 'launch closure missing')
    require(exact(spec['candidate_gate'], request['refs']['candidate_gate']), 'technical gate ref differs from frozen launch')
    w.validate_native_gate(docs['candidate_gate'], request['model'], pf['new_build_binding'],
                           request['gate_binding'], request['inputs'])
    p.validate_audit_terminal(docs['candidate_terminal'], 'candidate', request['model'], spec['run_ids'])
    fields(docs['candidate_terminal'], {'executor_session_id': request['executor_session_id'], 'source_head': HEAD,
                                      'final_used': False, 'goal_complete': False})
    require(type(audits) is dict and set(audits) == {'old', 'fallback', 'candidate'}, 'three full role audits required')
    for name, key in (('old', 'old_audit'), ('fallback', 'fallback_audit'), ('candidate', 'candidate_audit')):
        require(exact(audits[name]['audit_ref'], request['refs'][key]), 'external audit subject differs')
    require(exact(audits['candidate']['terminal_ref'], request['refs']['candidate_terminal']), 'candidate audit terminal differs')
    require(exact(audits['candidate']['run_ids'], spec['run_ids']), 'current four candidate runs differ')
    cr = docs['comparison_request']
    keys = {'schema', 'status', 'action', 'source_inputs', 'source_helpers', 'output',
            'baseline_audit', 'candidate_audit', 'bridge', 'model'}
    require(type(cr) is dict and set(cr) == keys, 'exact dedicated comparison request required')
    fields(cr, {'schema': 'sekirei.white-view-comparison-request.v1', 'status': 'frozen-before-comparison',
        'action': 'compare', 'baseline_audit': request['refs']['fallback_audit'],
        'candidate_audit': request['refs']['candidate_audit'], 'bridge': request['refs']['bridge'],
        'model': request['model'], 'output': request['refs']['comparison']['path']})
    filemap(cr['source_inputs'])
    for name, rec in cr['source_inputs'].items(): require(exact(request['inputs'].get(name), rec), 'comparison closure missing')
    pins = {'old_audit': request['refs']['old_audit'], 'new_audit': request['refs']['fallback_audit'],
        'old_runtime': str(B), 'old_build': audits['old']['build_manifest'], 'new_build': audits['fallback']['build_manifest']}
    require(exact(w.fallback_bridge(audits['old'], audits['fallback'], pins), bridge), 'semantic bridge reconstruction differs')
    actual = w.compare_same_runtime(audits['fallback'], audits['candidate'], bridge,
                                   request['refs']['bridge'], request['model'])
    actual.update(baseline_audit=request['refs']['fallback_audit'], candidate_audit=request['refs']['candidate_audit'])
    require(exact(actual, docs['comparison']), 'complete exact rational comparison does not reproduce')
    allow = docs['opaque_allowlist']
    require(type(allow) is dict and type(allow.get('descriptors')) is dict and len(allow['descriptors']) == 5,
            'exact five preexisting opaque services required')
    return actual['adopt']


def stop_document(request, request_ref, scans, before, after, adopt, allow, audits):
    require(type(adopt) is bool and exact(before, after), 'stop requires typed decision and unchanged inputs')
    filemap(before); fullref(request_ref)
    require(type(scans) is list and len(scans) == 2, 'two process scans required')
    for scan in scans:
        require(type(scan) is dict and set(scan) == {'related_processes', 'excluded_preexisting_services'}
            and exact(scan['related_processes'], []) and type(scan['excluded_preexisting_services']) is list
            and len(scan['excluded_preexisting_services']) == 5, 'empty related scan plus five fixed exclusions required')
        actual = {}
        for item in scan['excluded_preexisting_services']:
            require(type(item) is dict and type(item.get('descriptor')) is dict
                and item.get('identity_reread_unchanged') is True and exact(item.get('errno'), 13)
                and exact(item.get('allowlist'), request['refs']['opaque_allowlist']), 'fixed opaque exclusion evidence required')
            desc = item['descriptor']; pid = desc.get('pid')
            require(type(pid) is int and pid > 0 and str(pid) not in actual
                and item.get('denied_path') in ('/proc/'+str(pid)+'/exe', '/proc/'+str(pid)+'/cwd')
                and exact(item.get('reason'), allow['excluded_reason']), 'opaque identity/permission scope differs')
            actual[str(pid)] = desc
        require(exact(actual, allow['descriptors']), 'full five-service allowlist identities differ')
    require(type(audits) is dict and set(audits) == {'old','fallback','candidate'}, 'three raw audit inventories required')
    raw_files = {}; roots = []
    for audit in audits.values():
        inventory = audit.get('directory_inventory')
        require(type(inventory) is dict and len(inventory) == 4, 'four archived run inventories per role required')
        for root in inventory:
            root = path(root)
            require(root.parent in (B/'runs',N/'runs') and str(root) not in roots, 'distinct explicit runtime run root required')
            roots.append(str(root))
        values = filemap(audit.get('audit_inputs_after'))
        require(exact(values,audit.get('audit_inputs_before')), 'audit source inventory changed')
        for name,rec in values.items():
            if any(path(name).is_relative_to(path(root)) for root in inventory):
                require(exact(before.get(name),rec), 'raw audit file absent from stopped closure')
                require(name not in raw_files or exact(raw_files[name],rec), 'raw audit source rebinding')
                raw_files[name] = copy.deepcopy(rec)
    filemap(raw_files)
    return {'schema': 'sekirei.white-view-formal-stopped.v1', 'status': 'verified-stopped-under-locks',
        'candidate': MODE, 'model': copy.deepcopy(request['model']), 'source_head': HEAD,
        'request': copy.deepcopy(request_ref), 'refs': copy.deepcopy(request['refs']),
        'gate_binding': copy.deepcopy(request['gate_binding']), 'comparison_valid': True,
        'adopt': adopt, 'all_groups_stopped': True, 'executor_session_id': request['executor_session_id'],
        'comparison_session_id': request['comparison_session_id'],
        'held_locks': [{'path': str(p), 'exclusive': e} for p, e in LOCKS], 'process_scans': copy.deepcopy(scans),
        'audited_run_roots': roots, 'audited_raw_files': raw_files,
        'inputs_before': copy.deepcopy(before), 'inputs_after': copy.deepcopy(after), 'inputs_unchanged': True,
        'independent_raw_role_audits_verified': 3, 'attempts_per_role': 1829,
        'source_originals_preserved': True, 'final_used': False, 'adoption_applied': False,
        'best_model_updated': False, 'goal_complete': False}


def validate_archive_equality(source_before, source_after, destination):
    """Exact whole selected closure, including directories; no backup/restore claim."""
    for value in (source_before, source_after, destination):
        require(type(value) is dict and set(value) == {'files', 'directories'}, 'whole archive inventory required')
        require(type(value['files']) is dict and value['files'] and type(value['directories']) is list,
                'nonempty archive files and directory list required')
        require(len(set(value['directories'])) == len(value['directories']), 'duplicate archive directories')
        for name, rec in value['files'].items():
            require(type(name) is str and name and not P(name).is_absolute()
                and all(t not in ('', '.', '..') for t in name.split('/')), 'strict relative archive file required')
            identity(rec)
        for name in value['directories']:
            require(type(name) is str and (name == '.' or (name and not P(name).is_absolute()
                and all(t not in ('', '.', '..') for t in name.split('/')))), 'strict relative archive directory required')
    require(exact(source_before, source_after) and exact(source_before, destination), 'source before/after/destination archive differs')
    return {'files': len(destination['files']), 'directories': len(destination['directories']),
            'bytes': sum(rec['bytes'] for rec in destination['files'].values())}


def validate_mapping(mapping):
    require(type(mapping) is dict and mapping, 'nonempty selected source root mapping required')
    rows = []
    for original, relative in mapping.items():
        source = path(original)
        require(source not in (C,C.parent,R,N,B,T,FAN) and not source.is_relative_to(NAS),
                'mutable campaign/shared runtime/build root must not be copied wholesale')
        require(type(relative) is str and relative and not P(relative).is_absolute()
            and all(t not in ('','.','..') for t in relative.split('/')), 'strict relative NAS root required')
        target = P(relative)
        for prior, previous_target in rows:
            require(not (source.is_relative_to(prior) or prior.is_relative_to(source)
                or target.is_relative_to(previous_target) or previous_target.is_relative_to(target)),
                'duplicate/overlapping source or target roots')
        rows.append((source,target))
    return mapping


def validate_archive_request(value, stopped, stop_terminal, mapping, selected_inventory):
    """Root's already frozen SSD closure; this never manufactures that closure.

    Copying all selected roots and keeping external dependency references are
    different scopes. Neither helper success nor a valid negative comparison
    is an adoption action or a standalone environment restore certificate.
    """
    keys = {'schema','status','candidate','source_head','stop','stop_terminal','mapping','closure_inventory',
        'source_roots','inputs','required_archived_refs','external_dependencies','destination','receipt_name',
        'helper','parent_exclusive_locks','helper_exclusive_locks','source_originals_preserved',
        'standalone_environment_restore','final_used','adoption_applied','best_model_updated','goal_complete'}
    require(type(value) is dict and set(value) == keys, 'exact white-view archive request required')
    fields(value, {'schema':'sekirei.white-view-archive-request.v1','status':'frozen-after-stopped-before-nas-copy',
        'candidate':MODE,'source_head':HEAD,'source_originals_preserved':True,'standalone_environment_restore':False,
        'final_used':False,'adoption_applied':False,'best_model_updated':False,'goal_complete':False,
        'parent_exclusive_locks':list(map(str,ARCHIVE_OUTER_LOCKS)),
        'helper_exclusive_locks':list(map(str,ARCHIVE_HELPER_LOCKS))})
    filemap(value['inputs'])
    for key in ('stop','stop_terminal','mapping','closure_inventory','helper'): contains(value['inputs'],value[key])
    require(value['stop']['path'] == str(OUT), 'dedicated white-view stopped receipt required')
    require(value['helper']['path'] == str(HELPER) and value['helper']['sha256'] == HELPER_SHA,
            'unchanged helper source identity required')
    fields(stopped, {'schema':'sekirei.white-view-formal-stopped.v1','status':'verified-stopped-under-locks',
        'candidate':MODE,'source_head':HEAD,'comparison_valid':True,'all_groups_stopped':True,
        'inputs_unchanged':True,'source_originals_preserved':True,'final_used':False,
        'adoption_applied':False,'best_model_updated':False,'goal_complete':False})
    require(type(stopped.get('adopt')) is bool and exact(stopped.get('inputs_before'),stopped.get('inputs_after')),
            'valid positive or negative stop, unchanged inputs required')
    fields(stop_terminal, {'schema':'sekirei.white-view-postformal-stop-terminal.v1', 'status':'observed-stopped',
        'stop_receipt':value['stop'],'request':stopped['request'],'exit_code':0,'session_closed':True,
        'tool_observed_reaped':True,'all_related_groups_observed_stopped':True,'source_head':HEAD,
        'final_used':False,'goal_complete':False})
    require(type(stop_terminal.get('stop_session_id')) is int and stop_terminal['stop_session_id'] > 0,
            'Root-observed actual stopped-worker session required')
    for name,rec in filemap(stopped['inputs_before']).items():
        require(exact(value['inputs'].get(name),rec), 'complete stopped input map must remain externally pinned')
    validate_mapping(mapping); require(exact(value['source_roots'],mapping), 'selected source mapping differs')
    require(type(selected_inventory) is dict and set(selected_inventory) == {'schema','source_roots','files',
        'directories','original_paths','original_copy_ledger'}, 'dedicated frozen closure inventory required')
    require(selected_inventory['schema'] == 'sekirei.white-view-selected-archive-inventory.v1'
        and exact(selected_inventory['source_roots'],mapping), 'selected closure source mapping differs')
    tree = {k:selected_inventory[k] for k in ('files','directories')}
    validate_archive_equality(tree,tree,tree)
    originals = selected_inventory['original_paths']
    require(type(originals) is dict and set(originals) == set(tree['files']), 'every selected file provenance required')
    for relative,original in originals.items():
        original = path(original)
        matches = [(path(root),P(target)) for root,target in mapping.items() if original.is_relative_to(path(root))]
        require(len(matches) == 1 and str(matches[0][1]/original.relative_to(matches[0][0])) == relative,
                'file provenance does not match unique source root')
    require(type(value['required_archived_refs']) is list and value['required_archived_refs'], 'selected required controls required')
    reverse = {original:relative for relative,original in originals.items()}
    require(len(reverse) == len(originals), 'same source file copied twice')
    ledger = selected_inventory['original_copy_ledger']
    require(type(ledger) is dict, 'exclusive SSD control-copy provenance ledger required')
    for original, rec in ledger.items():
        path(original); fullref(rec)
        contains(value['inputs'], {'path':original,**{k:rec[k] for k in ('bytes','sha256')}})
        require(original not in reverse and rec['path'] in reverse
            and exact(tree['files'][reverse[rec['path']]],{k:rec[k] for k in ('bytes','sha256')}),
            'original/control-copy provenance differs or duplicates an original')
        reverse[original] = reverse[rec['path']]
    for rec in value['required_archived_refs']:
        contains(value['inputs'],rec)
        require(rec['path'] in reverse and exact(tree['files'][reverse[rec['path']]],
                {k:rec[k] for k in ('bytes','sha256')}), 'mandatory selected control missing from copied closure')
    mandatory = [value['stop'],stopped['request'],*stopped['refs'].values(),
                 *[stopped['gate_binding'][k] for k in stopped['gate_binding'] if k != 'schema']]
    for rec in mandatory:
        require(any(exact(rec,provided) for provided in value['required_archived_refs']), 'current stop/control/gate closure omission')
    roots = stopped.get('audited_run_roots')
    require(type(roots) is list and len(roots) == 12 and len(set(roots)) == 12
        and all(root in mapping for root in roots), 'all old/fresh/candidate four-stage roots must be selected')
    for root in (str(Q), str(P(stopped['gate_binding']['core']['path']).parent),
                 str(P(stopped['gate_binding']['incremental']['path']).parent)):
        require(root in mapping, 'whole fit or raw model-proof tree omitted from selected closure')
    for name,rec in filemap(stopped.get('audited_raw_files')).items():
        require(name in reverse and exact(tree['files'][reverse[name]],rec), 'raw audit source missing from archived closure')
    filemap(value['external_dependencies'])
    for name,rec in value['external_dependencies'].items():
        require(exact(value['inputs'].get(name),rec) and name not in reverse,
                'external dependency identity missing or ambiguously copied')
    require(set(value['inputs']) <= set(reverse)|set(value['external_dependencies']),
            'every immutable input must be copied or explicitly retained as external dependency')
    destination = path(value['destination'])
    require(destination.is_relative_to(NAS/'archives') and destination != NAS/'archives', 'fresh dedicated NAS destination required')
    require(type(value['receipt_name']) is str and re.fullmatch('[a-z0-9][a-z0-9_.-]{0,127}',value['receipt_name']),
            'dedicated receipt basename required')
    return value
