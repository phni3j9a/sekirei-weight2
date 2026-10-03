"""Dedicated candidate6 activation after SHA-pinned fifth valid rejection.

Pure validate_bundle accepts synthetic objects and identities. Runtime entry
requires canonical public source and externally pinned activation before verification.
This verifier launches no process, fits no model and decides no adoption.
"""
import hashlib
import json
from pathlib import Path
import stat
from datetime import datetime

PROTOTYPE_ONLY = False
C = Path('/home/server/.local/share/sekirei-weight2/campaign-17-autonomous-v1')
R = Path('/home/server/worktrees/sekirei-weight2/issue-17-autonomous-weight-improvement')
B = C.parent / 'suisho11beta-sekirei-v0.3.39-v1'
T = C.parent / 'training-15-v1'
F = C.parent / 'training-17-v1/bounded-material-fanin509-trainer-v1'
E = C / 'eval-bounded-material-fanin509-100cp-e3-v1'
W = F / 'run-e3-v1/weights.bin'
PLAN = C / 'paired-linear-constrained-ridge1-plan-v1.json'
ACTIVATION = C / 'paired-linear-activation-v1.json'
PRIOR = 'bounded-material-fanin509-100cp-e3-v1'
NEXT = 'paired-linear-constrained-ridge1-l1-39p5-v1'
PLAN_SHA256 = 'dabad54e419237335fd1f370063a0c6e82ac7b58ac27f0d20fd91ca21d9dd900'
MODEL_SHA256 = '2ec2eacb5f3cb9f72921efd75478f024614a3f944c3ee9a528df9898d6d93164'
META_SHA256 = '8e91a7e49cbda6f9a6295030771e68b7deb779f047cf027e88514f9be17df966'
PREREG_SHA256 = '23cc5507699e5a605f980137761a1ba10674655e1cec8faa43a3ff8d3887f424'
BUILD_SHA256 = 'a2b626051ec839e8f41cddb0d5b42892926944c58cf9929cc0672fa093544e57'
PREFLIGHT_SHA256 = '947e657557d6b54234189e7497afac16b544da54c8bfb1986193e5b868b25b94'
PREFIX = 'development-17-bounded-fanin509-e3-v1'
PYTHON = str(C.parent / 'suisho11beta-v1/venv/bin/python')
PATHS = {
 'comparison': C / 'comparison-bounded-material-fanin509-100cp-e3-v1.json',
 'independent_review': C / 'fanin509-independent-formal-review-v1.json',
 'stop_receipt': C / 'fanin509-formal-stopped-v1.json',
 'operational': C / 'fanin509-formal-operational-verification-v1.json',
 'terminal': C / 'fanin509-formal-top3-recovery-terminal-observation-v1.json',
 'evaluation': E / 'evaluation.json',
 'failure_closure': C / 'fanin509-formal-failure-closure-v1.json',
 'failed_terminal': C / 'fanin509-formal-executor-failed-terminal-observation-v1.json',
 'recovery_preflight': C / 'fanin509-formal-top3-recovery-preflight-v2.json',
 'original_preflight': C / 'launch-preflight-bounded-material-fanin509-e3-v1.json',
 'preregistration': C / 'bounded-material-fanin509-preregistration-v1.json',
 'plan': PLAN}
FIXED_SHA = {
 'comparison': '9a2107a9bd7032adcaf990011fc772debe8b19dc54c083810e48afb3b08055b8',
 'independent_review': '44239bf763de9cf9cf7e2c65b5fd85e42ed875477054d700cf3290b41c1c3c26',
 'stop_receipt': 'd15748058da591357c4f718e2520662826a698a8ae80f144edd8ac99ca36757d',
 'failure_closure': '065ebf160dc502730df19f8c4984460f16d48bc0a1ebb2d817ad55b9be61a905',
 'failed_terminal': 'dd47776c9fe4859f69170d18fc9ef203c925cedd74faa20e6fea6bb2d094b7cd',
 'recovery_preflight': 'ae718b417207ae89089fa5b677b03941e30e2677e74ad266f50f0f9e7007ad91',
 'original_preflight': PREFLIGHT_SHA256, 'preregistration': PREREG_SHA256, 'plan': PLAN_SHA256}
STOP_WORKER = C / 'fanin509-formal-stop-worker-v3.py'
ORIGINAL_WORKER = C / 'fanin509-formal-evaluation-worker-v1.py'
RECOVERY_WORKER = C / 'fanin509-formal-top3-recovery-worker-v2.py'
RECOVERY_CONTRACT = C / 'fanin509_recovery_contract_v3.py'
WORKERS = sorted(map(str, (E / 'evaluation-script.py', ORIGINAL_WORKER, RECOVERY_WORKER)))
LOCKS = list(map(str, (T / '.training.lock', B / '.prepare.lock', B / '.benchmark.lock',
                      F / '.build.lock', F / '.training.lock')))
EXECUTABLES = sorted(map(str, (B / 'build/sekirei/release/sekirei', B / 'sources/yaneuraou/source/YaneuraOu-by-gcc')))
STAGES = {'mae-pilot': 102, 'mae': 1140, 'top3-pilot': 36, 'top3': 551}
BASE_RUNS = {'mae-pilot': 'development-pilot-20261002-v039-v1',
 'mae': 'development-baseline-20261002-v039-v1',
 'top3-pilot': 'development-top3-baseline-pilot-20261002-v1', 'top3': 'development-top3-baseline-20261002-v1'}
IDENTITIES = {'execution_except_model', 'occurrences', 'teacher_results', 'teacher_E_cp',
 'top3_occurrences', 'top3_legal_moves_sha256', 'top3_denominators', 'top3_teacher_bestmoves'}
SCOPE = {'split':'development', 'games':5, 'teacher_E':266, 'top3_denominator':551,
 'candidate_attempts':1829, 'baseline_attempts':1829, 'own_pilot_attempts':138,
 'requested_nodes':1000000, 'maximum_reported_nodes':1010000}
HELPERS = {'evaluate_candidate.py','compare_candidates.py','publish_comparison.py','benchmark.py',
 'benchmark_report.py','top3.py','prepare.py','audit_pack.py','smoke.py','bounded_material.py',
 'diagnose_anchor.py','diagnose_bounded.py','diagnose_fanin509.py','diagnose_weights.py','export_nearest.py',
 'fanin509_activation.py','functional_anchor.py','material_init.py','pack_dataset.py','prepare_bounded.py',
 'prepare_fanin509.py','train_bounded.py','train_cpu.py','train_fanin509.py'}
STOP_HELPERS = HELPERS - {'audit_pack.py','benchmark_report.py','compare_candidates.py',
                         'evaluate_candidate.py','publish_comparison.py','smoke.py','top3.py'}
CORE_SOURCE_HASHES = {
 'crates/sekirei-core/src/nnue.rs':'a467e8b1b75f6b82629f369a1c804476b1c610354a561822365f980e6e428561',
 'crates/sekirei-core/src/eval.rs':'69fce9b3522a96c68a7f6dce15a342ba34c90f0addd0fa13b7478bef4d25e839',
 'crates/sekirei-core/src/piece.rs':'2ccf9248daf5a5b5eba2651ce03a5f85918608b4c5651ab5c728ddf6ba0a50a3',
 'crates/sekirei-core/src/square.rs':'60ee89de778fc18e30bbe0897f7380e4dfbf2d80f27afbe6d2b4e405c92e68b2'}


def require(condition, reason):
    if not condition: raise ValueError(reason)


def exact(actual, expected):
    if type(actual) is not type(expected): return False
    if type(expected) is dict:
        return actual.keys() == expected.keys() and all(exact(actual[k],v) for k,v in expected.items())
    if type(expected) is list:
        return len(actual)==len(expected) and all(exact(a,b) for a,b in zip(actual,expected))
    return actual == expected


def sha(value):
    require(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value), 'invalid typed SHA256')
    return value


def inventory(value):
    require(type(value) is dict, 'typed file inventory required')
    for p,v in value.items():
        require(type(p) is str and Path(p).is_absolute() and Path(p).as_posix()==p and '..' not in Path(p).parts
                and type(v) is dict and set(v)=={'bytes','sha256'} and type(v['bytes']) is int and v['bytes']>=0,
                'invalid canonical path or file identity')
        sha(v['sha256'])
    return value


def validate_trigger_reference(value):
    require(type(value) is dict and set(value)=={'path','sha256'} and value['path']==str(ACTIVATION), 'wrong activation reference')
    sha(value['sha256']); return value


def validate_build_binding(build, reference, verifier_sha256, proof):
    validate_trigger_reference(reference);sha(verifier_sha256)
    require(type(build) is dict and exact(build.get('trigger'),reference)
            and exact(build.get('activation_helper_sha256'),verifier_sha256)
            and exact(build.get('activation_validation'),proof), 'activation/build/source binding differs')


def timestamp(value):
    require(type(value) is str, 'typed timestamp required')
    result=datetime.fromisoformat(value)
    require(result.tzinfo is not None, 'aware timestamp required')
    return result


def validate_bundle(activation, docs, actual):
    """Pure validation. actual contains CURRENT file identities supplied by driver."""
    inventory(actual)
    require(type(docs) is dict and set(docs)==set(PATHS), 'complete fixed control bundle required')
    def identity(path):
        require(str(path) in actual, 'current identity missing: '+str(path));return actual[str(path)]
    def bound(ref,path,full=False):
        require(type(ref) is dict and ref.get('path')==str(path) and ref.get('sha256')==identity(path)['sha256'], 'control path/SHA mismatch')
        if full: require(exact(ref, {'path':str(path),**identity(path)}), 'control byte/field mismatch')
    for name,expected in FIXED_SHA.items(): require(identity(PATHS[name])['sha256']==expected, 'fixed receipt or plan changed: '+name)
    cp,au,stop=(docs[n] for n in ('comparison','independent_review','stop_receipt'))
    op,term,ev=(docs[n] for n in ('operational','terminal','evaluation'))
    plan=docs['plan'];pf=docs['original_preflight'];rp=docs['recovery_preflight'];failure=docs['failure_closure'];failedterm=docs['failed_terminal']
    require(type(activation) is dict and activation.get('schema')=='sekirei.paired-linear-activation.v1'
            and activation.get('status')=='verified-after-valid-nonadopt-stopped'
            and activation.get('prior_candidate')==PRIOR and activation.get('next_candidate')==NEXT
            and activation.get('plan_sha256')==PLAN_SHA256, 'wrong candidate activation')
    for k,v in {'formal_valid':True,'adopted':False,'all_groups_stopped':True,'final_used':False,
                'best_baseline_updated':False,'measurement_settings_changed':False}.items():
        require(exact(activation.get(k),v), 'activation flag/type differs: '+k)
    for name in ('comparison','independent_review','stop_receipt'):
        ref=activation.get(name);require(type(ref) is dict and set(ref)=={'path','sha256'}, 'activation receipt reference shape differs');bound(ref,PATHS[name])
    require(exact(activation.get('bound_inputs'),actual), 'activation current input map differs')
    require(plan.get('schema')=='sekirei.paired-linear-constrained-ridge-plan.v1'
            and plan.get('status')=='fixed-before-fifth-formal-outcome-not-activated' and plan.get('candidate')==NEXT,
            'fixed candidate plan identity differs')
    for k,v in {'formal_selection_rule_unchanged':True,'baseline_best_updated':False,'final_used':False,'implementation_or_fit_activated':False}.items():
        require(exact(plan.get(k),v), 'plan flag differs: '+k)
    require(timestamp(plan.get('created_at')) < timestamp(cp.get('created_at'))
            <= timestamp(stop.get('created_at')) <= timestamp(activation.get('created_at')), 'plan must precede outcome; activation follows stop')
    for value,schema in ((cp,'sekirei.formal-comparison.v1'),(au,'sekirei.fanin509-independent-formal-review.v1')):
        require(type(value) is dict and value.get('schema')==schema and value.get('status')=='complete', 'incomplete formal/audit')
        for k,v in {'comparison_valid':True,'adopt':False,'inputs_unchanged':True,'final_used':False}.items(): require(exact(value.get(k),v), 'formal flag/type differs: '+k)
        require(value.get('decision') in ('hold','reject') and type(value.get('mae_improved')) is bool
                and type(value.get('top3_preserved')) is bool and value['adopt']==(value['mae_improved'] and value['top3_preserved']), 'formal decision inconsistent')
        checks=value.get('checks');require(type(checks) is dict and set(checks)==IDENTITIES, 'eight identities required')
        for item in checks.values():
            require(type(item) is dict and item.get('equal') is True and sha(item.get('baseline_sha256'))==sha(item.get('candidate_sha256')), 'identity differs')
            if 'mismatch_count' in item:require(exact(item['mismatch_count'],0),'identity mismatch count/type')
    for key in ('decision','adopt','mae_improved','top3_preserved','checks'):require(exact(cp.get(key),au.get(key)), 'formal/audit decision conflict')
    require(exact(cp.get('requested_nodes'),1000000) and au.get('goal_achieved') is False
            and au.get('preregistration_sha256')==PREREG_SHA256 and au.get('launch_preflight_sha256')==PREFLIGHT_SHA256, 'formal provenance/scope differs')
    require(type(au.get('scope')) is dict and all(exact(au['scope'].get(k),v) for k,v in SCOPE.items()), 'full audit scope differs')
    bound(au.get('comparison_receipt'),PATHS['comparison']);bound(au.get('operational_receipt'),PATHS['operational'])
    model={'path':str(W),'bytes':1305356,'sha256':MODEL_SHA256}
    require(exact(identity(W),{k:model[k] for k in ('bytes','sha256')}) and identity(W.with_suffix('.meta.json'))['sha256']==META_SHA256, 'stopped native/meta changed')
    require(exact(cp.get('candidate',{}).get('model_identity'),{'kind':'nnue',**model})
            and exact(cp.get('baseline',{}).get('model_identity'),{'kind':'material_fallback','path':None,'bytes':None,'sha256':None}), 'model/best fallback differs')
    require(type(au.get('candidate_model')) is dict and all(exact(au['candidate_model'].get(k),v) for k,v in model.items())
            and au['candidate_model'].get('metadata_sha256')==META_SHA256, 'audited native/meta differs')
    audits=au.get('run_audits');require(type(audits) is dict and set(audits)=={'candidate','baseline'}, 'both full stage audits required')
    for subject,stages in audits.items():
        require(type(stages) is dict and set(stages)==set(STAGES), 'four stage audits required')
        for stage,count in STAGES.items():
            rec=stages[stage];path=B/'runs'/((PREFIX+'-'+stage) if subject=='candidate' else BASE_RUNS[stage])
            require(type(rec) is dict and rec.get('path')==str(path), 'audited stage path differs')
            for key,expected in {'attempts':count,'cleanup_ok':count,'runner_exit_zero':count,
                'raw_sha256_and_lifecycle_verified':count,'technical_failures':0,'timeout_failures':0}.items():require(exact(rec.get(key),expected),'stage cleanup/count/type differs: '+key)
    require(op.get('schema')=='sekirei.fanin509-formal-operational-verification.v1' and op.get('status')=='complete', 'incomplete operational receipt')
    for k,v in {'inputs_unchanged':True,'recovery_inputs_unchanged':True,'build_unchanged':True,'adoption_claimed':False,'final_used':False,'selected_epoch':3}.items():require(exact(op.get(k),v), 'operational flag/type differs')
    require(op.get('preregistration_sha256')==PREREG_SHA256 and op.get('evaluation_sha256')==identity(PATHS['evaluation'])['sha256'], 'operational provenance differs')
    original=inventory(op.get('inputs_before'));require(len(original)==1222 and exact(original,op.get('inputs_after')), 'original input snapshot differs')
    expected=dict(inventory(pf.get('bound_files')))
    require(len(expected)==1221, 'original PF scope differs')
    for path in (PATHS['original_preflight'],PATHS['preregistration'],ORIGINAL_WORKER):
        current=identity(path)
        if str(path) in expected: require(exact(expected[str(path)],current), 'original PF union collision')
        expected[str(path)]=current
    require(exact(original,expected), 'original operational union map differs')
    recovery=inventory(op.get('recovery_inputs_before'));require(len(recovery)==4207 and exact(recovery,op.get('recovery_inputs_after')), 'recovery inputs differ')
    rpexpected=dict(inventory(rp.get('immutable_inputs')));require(len(rpexpected)==4206,'recovery PF scope differs')
    for path in (PATHS['failure_closure'],PATHS['recovery_preflight'],RECOVERY_WORKER):
        current=identity(path)
        if str(path) in rpexpected: require(exact(rpexpected[str(path)],current), 'recovery PF union collision')
        rpexpected[str(path)]=current
    require(exact(recovery,rpexpected), 'recovery operational union differs')
    require(ev.get('status')=='complete' and ev.get('weight_sha256')==MODEL_SHA256
            and exact(ev.get('run_ids'),{s:PREFIX+'-'+s for s in STAGES}), 'completed stages/model differ')
    require(term.get('schema')=='sekirei.executor-terminal-observation.v1' and exact(term.get('session_id'),26958)
            and exact(term.get('exit_code'),0) and term.get('session_closed') is True
            and exact(term.get('completed_recovery_attempts'),587) and term.get('rerun_mae') is False,
            'recovery executor not reaped successfully')
    for k,v in {'final_used':False,'adoption_claimed':False}.items():require(exact(term.get(k),v),'terminal flag differs')
    bound(term.get('operational_receipt'),PATHS['operational'],True);bound(term.get('evaluation_receipt'),PATHS['evaluation'],True)
    history=[{'session_id':89125,'exit_code':1,'session_closed':True,
              'failure_closure':{'path':str(PATHS['failure_closure']),**identity(PATHS['failure_closure'])},
              'failed_terminal':{'path':str(PATHS['failed_terminal']),**identity(PATHS['failed_terminal'])}}]
    require(exact(term.get('failed_executor_history'),history) and exact(stop.get('failed_executor_history'),history), 'original failed executor history lost')
    require(failure.get('schema')=='sekirei.fanin509-formal-failure-closure.v1'
            and failure.get('status')=='verified-partial-stopped-after-interpreter-error'
            and exact(failure.get('failed_executor_session_id'),89125) and exact(failure.get('failed_executor_exit_code'),1)
            and failure.get('session_closed') is True and failure.get('all_related_groups_observed_stopped') is True,
            'original failed closure differs')
    require(failedterm.get('schema')=='sekirei.executor-terminal-observation.v1'
            and exact(failedterm.get('session_id'),89125) and exact(failedterm.get('exit_code'),1)
            and failedterm.get('session_closed') is True, 'failed executor terminal differs')
    segments=op.get('execution_segments');require(type(segments) is list and len(segments)==2 and exact(ev.get('execution_segments'),segments), 'two honest execution segments required')
    first,second=segments
    for k,v in {'kind':'original-system-python-launch','session':89125,'exit_code':1,'session_closed':True,
       'status':'failed-before-top3-creation','completed_stages':['mae-pilot','mae'],'completed_attempts':[102,1140],
       'failed_stage':'top3-pilot','top3_attempts':0}.items():require(exact(first.get(k),v),'original execution segment differs')
    bound(first.get('failure_closure'),PATHS['failure_closure'],True)
    for k,v in {'kind':'pinned-audit-venv-top3-recovery','status':'complete','stages':['top3-pilot','top3'],
                'attempts':[36,551],'python':PYTHON,'dependencies':{'cshogi':'1.0.4','numpy':'1.26.4'}}.items():require(exact(second.get(k),v),'recovery segment differs')
    require(rp.get('correct_interpreter')==PYTHON and exact(rp.get('decoder_versions'),second['dependencies'])
            and rp.get('measurement_settings_changed') is False and rp.get('final_used') is False, 'recovery interpreter/measurement differs')
    recovery_evidence=au.get('recovery_evidence');require(exact(recovery_evidence,stop.get('recovery_evidence')), 'audit/stop recovery evidence conflict')
    for k,v in {'recovery_session_id':26958,'recovery_input_count':4207,'recovery_preflight_input_count':4206,
       'immutable_failure_snapshot_count':18,'controls_unchanged':True,'recovery_inputs_unchanged':True,
       'decoder_versions':{'cshogi':'1.0.4','numpy':'1.26.4'},'execution_segments':segments}.items():require(exact(recovery_evidence.get(k),v),'recovery evidence differs')
    for field,name in {'failure_closure':'failure_closure','failed_terminal':'failed_terminal',
                       'recovery_preflight':'recovery_preflight','recovery_terminal':'terminal'}.items():bound(recovery_evidence.get(field),PATHS[name],True)
    bound(recovery_evidence.get('recovery_worker'),RECOVERY_WORKER,True);bound(recovery_evidence.get('contract_source'),RECOVERY_CONTRACT,True)
    require(stop.get('schema')=='sekirei.fanin509-formal-stopped.v1' and stop.get('status')=='verified-stopped-under-locks'
            and stop.get('candidate')==PRIOR and stop.get('bounded_mode')=='bounded-material-fanin509-v1'
            and stop.get('preregistration_sha256')==PREREG_SHA256 and stop.get('build_manifest_sha256')==BUILD_SHA256, 'wrong stopped candidate provenance')
    for k,v in {'all_groups_stopped':True,'comparison_valid':True,'adopt':False,'final_used':False,'inputs_unchanged':True,
       'executor_session_id':26958,'executor_exit_code':0,'recorded_cleanup_verified_candidate_attempts':1829,
       'recorded_cleanup_verified_baseline_attempts':1829,'source_context_unchanged':True}.items():require(exact(stop.get(k),v),'stopped flag/type/scope differs')
    require(exact(stop.get('exclusive_locks'),LOCKS) and exact(stop.get('related_worker_paths'),WORKERS)
            and exact(stop.get('related_executables'),EXECUTABLES), 'stopped exclusion contract differs')
    scans=stop.get('process_scans');require(type(scans) is list and len(scans)==2, 'two process scans required')
    for scan in scans:require(type(scan) is dict and set(scan)=={'related_processes','owned_processes_scanned'}
                and exact(scan['related_processes'],[]) and type(scan['owned_processes_scanned']) is int
                and scan['owned_processes_scanned']>=0, 'related process or invalid process scan')
    for field,name in {'comparison_receipt':'comparison','independent_review_receipt':'independent_review',
       'operational_receipt':'operational','terminal_observation':'terminal','evaluation_receipt':'evaluation'}.items():bound(stop.get(field),PATHS[name],True)
    bound(stop.get('candidate_model'),W,True)
    snapshot={str(p):identity(p) for p in (STOP_WORKER,PATHS['terminal'],PATHS['operational'],PATHS['comparison'],PATHS['independent_review'],PATHS['evaluation'],W,W.with_suffix('.meta.json'))}
    require(exact(stop.get('inputs_before'),snapshot) and exact(stop.get('inputs_after'),snapshot), 'eight current stop snapshots differ')
    require(exact(stop.get('source_context_before'),stop.get('source_context_after')), 'stop source context changed')
    helpers=au.get('source_helpers_sha256');require(type(helpers) is dict and set(helpers)==HELPERS, 'formal helper inventory changed')
    stopped_helpers=stop.get('source_helpers_sha256')
    require(type(stopped_helpers) is dict and set(stopped_helpers)==STOP_HELPERS
            and exact(stopped_helpers,{name:helpers[name] for name in STOP_HELPERS}), 'stopped helper inventory changed')
    for name,value in helpers.items():require(identity(R/'scripts'/name)['sha256']==sha(value), 'old helper source changed: '+name)
    for name,value in CORE_SOURCE_HASHES.items():require(identity(T/'source'/name)['sha256']==value, 'stock core source changed: '+name)
    expected_paths=set(map(str,PATHS.values()))|{str(p) for p in (STOP_WORKER,ORIGINAL_WORKER,RECOVERY_WORKER,RECOVERY_CONTRACT,W,W.with_suffix('.meta.json'))}
    expected_paths|={str(R/'scripts'/name) for name in HELPERS}|{str(T/'source'/name) for name in CORE_SOURCE_HASHES}
    require(set(actual)==expected_paths, 'new activation input map incomplete or unexpected')
    return {'schema':'sekirei.paired-linear-activation-validation.v1','status':'verified',
      'prior_candidate':PRIOR,'next_candidate':NEXT,'plan_sha256':PLAN_SHA256,'formal_valid':True,'adopted':False,
      'all_groups_stopped':True,'best_baseline_updated':False,'final_used':False,
      'receipt_and_stopped_input_identities':actual,
      'limitation':'Stopped state is a locked observation; no actual fit, core bound, replay or adoption is proven.'}


def make_activation(docs,actual,created_at):
    """Pure document construction; caller freezes the resulting bytes externally."""
    value={'schema':'sekirei.paired-linear-activation.v1','status':'verified-after-valid-nonadopt-stopped',
       'created_at':created_at,'prior_candidate':PRIOR,'next_candidate':NEXT,'plan_sha256':PLAN_SHA256,
       'formal_valid':True,'adopted':False,'all_groups_stopped':True,'final_used':False,
       'best_baseline_updated':False,'measurement_settings_changed':False,'bound_inputs':actual}
    for name in ('comparison','independent_review','stop_receipt'):
        value[name]={'path':str(PATHS[name]),'sha256':actual[str(PATHS[name])]['sha256']}
    validate_bundle(value,docs,actual);return value


def strict_json(data):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,'duplicate JSON key');result[key]=value
        return result
    return json.loads(data,object_pairs_hook=pairs,parse_constant=lambda value:require(False,'nonfinite JSON constant'))


def file_bytes(path):
    path=Path(path);require(path.is_absolute() and path.resolve(strict=True)==path and not path.is_symlink()
           and stat.S_ISREG(path.stat().st_mode),'noncanonical/nonregular activation input')
    return path.read_bytes()


def file_info(path):
    data=file_bytes(path);return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


def verify_public_module(expected_sha256):
    expected=R/'scripts/paired_linear_activation.py'
    require(Path(__file__)==expected and Path(__file__).resolve()==expected,'canonical public activation module required')
    value=file_info(expected);require(value['sha256']==sha(expected_sha256),'activation helper source changed');return value


def verify_activation(reference,expected_helper_sha256):
    require(not PROTOTYPE_ONLY,'PROTOTYPE_ONLY: activation runtime integration not enabled')
    validate_trigger_reference(reference);verify_public_module(expected_helper_sha256)
    actual={}
    def read(path):
        data=file_bytes(path);actual[str(path)]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()};return strict_json(data)
    activation=read(ACTIVATION);require(actual[str(ACTIVATION)]['sha256']==reference['sha256'],'externally fixed activation changed')
    actual.pop(str(ACTIVATION))
    docs={name:read(path) for name,path in PATHS.items()}
    for path in (STOP_WORKER,ORIGINAL_WORKER,RECOVERY_WORKER,RECOVERY_CONTRACT,W,W.with_suffix('.meta.json')):
        actual[str(path)]=file_info(path)
    for name in HELPERS:actual[str(R/'scripts'/name)]=file_info(R/'scripts'/name)
    for name in CORE_SOURCE_HASHES:actual[str(T/'source'/name)]=file_info(T/'source'/name)
    proof=validate_bundle(activation,docs,actual)
    require(all(exact(file_info(path),value) for path,value in actual.items()),'activation inputs changed during verification')
    require(file_info(ACTIVATION)['sha256']==reference['sha256'],'activation changed during verification')
    verify_public_module(expected_helper_sha256);return proof
