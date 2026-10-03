"""Synthetic typed fifth/recovery controls; no real controls or model reads."""
import copy
import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from unittest import mock

import paired_linear_activation as a


def synthetic_bundle():
    paths=set(map(str,a.PATHS.values()))|{str(p) for p in (a.STOP_WORKER,a.ORIGINAL_WORKER,a.RECOVERY_WORKER,a.RECOVERY_CONTRACT,a.W,a.W.with_suffix('.meta.json'))}
    paths|={str(a.R/'scripts'/name) for name in a.HELPERS}|{str(a.T/'source'/name) for name in a.CORE_SOURCE_HASHES}
    actual={p:{'bytes':1,'sha256':'1'*64} for p in paths}
    for name,value in a.FIXED_SHA.items():actual[str(a.PATHS[name])]['sha256']=value
    actual[str(a.W)]={'bytes':1305356,'sha256':a.MODEL_SHA256}
    actual[str(a.W.with_suffix('.meta.json'))]['sha256']=a.META_SHA256
    for name,value in a.CORE_SOURCE_HASHES.items():actual[str(a.T/'source'/name)]['sha256']=value
    ref=lambda p:{'path':str(p),**actual[str(p)]}
    checks={name:{'equal':True,'baseline_sha256':'2'*64,'candidate_sha256':'2'*64,'mismatch_count':0} for name in a.IDENTITIES}
    cp={'schema':'sekirei.formal-comparison.v1','status':'complete','comparison_valid':True,'adopt':False,
        'inputs_unchanged':True,'final_used':False,'decision':'hold','mae_improved':False,'top3_preserved':True,
        'checks':checks,'requested_nodes':1000000,'created_at':'2026-10-03T15:58:16+00:00',
        'baseline':{'model_identity':{'kind':'material_fallback','path':None,'bytes':None,'sha256':None}},
        'candidate':{'model_identity':{'kind':'nnue',**ref(a.W)}}}
    au={k:copy.deepcopy(v) for k,v in cp.items() if k not in ('requested_nodes','baseline','candidate','created_at')}
    au.update(schema='sekirei.fanin509-independent-formal-review.v1',goal_achieved=False,
       preregistration_sha256=a.PREREG_SHA256,launch_preflight_sha256=a.PREFLIGHT_SHA256,
       scope=copy.deepcopy(a.SCOPE),comparison_receipt=ref(a.PATHS['comparison']),operational_receipt=ref(a.PATHS['operational']),
       candidate_model={**ref(a.W),'metadata_sha256':a.META_SHA256},source_helpers_sha256={name:'1'*64 for name in a.HELPERS})
    au['run_audits']={}
    for subject in ('candidate','baseline'):
        au['run_audits'][subject]={}
        for stage,count in a.STAGES.items():
            path=a.B/'runs'/((a.PREFIX+'-'+stage) if subject=='candidate' else a.BASE_RUNS[stage])
            au['run_audits'][subject][stage]={'path':str(path),'attempts':count,'cleanup_ok':count,
              'runner_exit_zero':count,'raw_sha256_and_lifecycle_verified':count,'technical_failures':0,'timeout_failures':0}
    original={p:v for p,v in actual.items() if p!=str(a.PATHS['original_preflight'])}
    for i in range(1221-len(original)):original['/tmp/synthetic-original/'+str(i)]={'bytes':1,'sha256':'3'*64}
    pf={'bound_files':original}
    union={**original,str(a.PATHS['original_preflight']):actual[str(a.PATHS['original_preflight'])]}
    recovery={p:v for p,v in actual.items() if p!=str(a.PATHS['recovery_preflight'])}
    for i in range(4206-len(recovery)):recovery['/tmp/synthetic-recovery/'+str(i)]={'bytes':1,'sha256':'3'*64}
    rp={'immutable_inputs':recovery,'correct_interpreter':a.PYTHON,'decoder_versions':{'cshogi':'1.0.4','numpy':'1.26.4'},
        'measurement_settings_changed':False,'final_used':False}
    recovery_union={**recovery,str(a.PATHS['recovery_preflight']):actual[str(a.PATHS['recovery_preflight'])]}
    segments=[{'kind':'original-system-python-launch','session':89125,'exit_code':1,'session_closed':True,
       'status':'failed-before-top3-creation','completed_stages':['mae-pilot','mae'],'completed_attempts':[102,1140],
       'failed_stage':'top3-pilot','top3_attempts':0,'failure_closure':ref(a.PATHS['failure_closure'])},
      {'kind':'pinned-audit-venv-top3-recovery','status':'complete','stages':['top3-pilot','top3'],'attempts':[36,551],
       'python':a.PYTHON,'dependencies':{'cshogi':'1.0.4','numpy':'1.26.4'}}]
    ev={'status':'complete','weight_sha256':a.MODEL_SHA256,'run_ids':{s:a.PREFIX+'-'+s for s in a.STAGES},'execution_segments':segments}
    op={'schema':'sekirei.fanin509-formal-operational-verification.v1','status':'complete',
       'inputs_unchanged':True,'recovery_inputs_unchanged':True,'build_unchanged':True,'adoption_claimed':False,
       'final_used':False,'selected_epoch':3,'preregistration_sha256':a.PREREG_SHA256,
       'evaluation_sha256':actual[str(a.PATHS['evaluation'])]['sha256'],'inputs_before':union,'inputs_after':copy.deepcopy(union),
       'recovery_inputs_before':recovery_union,'recovery_inputs_after':copy.deepcopy(recovery_union),'execution_segments':segments}
    history=[{'session_id':89125,'exit_code':1,'session_closed':True,'failure_closure':ref(a.PATHS['failure_closure']),
              'failed_terminal':ref(a.PATHS['failed_terminal'])}]
    term={'schema':'sekirei.executor-terminal-observation.v1','session_id':26958,'exit_code':0,'session_closed':True,
       'completed_recovery_attempts':587,'rerun_mae':False,'final_used':False,'adoption_claimed':False,
       'operational_receipt':ref(a.PATHS['operational']),'evaluation_receipt':ref(a.PATHS['evaluation']),
       'failed_executor_history':history}
    failure={'schema':'sekirei.fanin509-formal-failure-closure.v1','status':'verified-partial-stopped-after-interpreter-error',
       'failed_executor_session_id':89125,'failed_executor_exit_code':1,'session_closed':True,'all_related_groups_observed_stopped':True}
    failedterm={'schema':'sekirei.executor-terminal-observation.v1','session_id':89125,'exit_code':1,'session_closed':True}
    evidence={'recovery_session_id':26958,'recovery_input_count':4207,'recovery_preflight_input_count':4206,
       'immutable_failure_snapshot_count':18,'controls_unchanged':True,'recovery_inputs_unchanged':True,
       'decoder_versions':{'cshogi':'1.0.4','numpy':'1.26.4'},'execution_segments':segments,
       'failure_closure':ref(a.PATHS['failure_closure']),'failed_terminal':ref(a.PATHS['failed_terminal']),
       'recovery_preflight':ref(a.PATHS['recovery_preflight']),'recovery_terminal':ref(a.PATHS['terminal']),
       'recovery_worker':ref(a.RECOVERY_WORKER),'contract_source':ref(a.RECOVERY_CONTRACT)}
    au['recovery_evidence']=evidence
    snapshots={str(p):actual[str(p)] for p in (a.STOP_WORKER,a.PATHS['terminal'],a.PATHS['operational'],a.PATHS['comparison'],
      a.PATHS['independent_review'],a.PATHS['evaluation'],a.W,a.W.with_suffix('.meta.json'))}
    stop={'schema':'sekirei.fanin509-formal-stopped.v1','status':'verified-stopped-under-locks','candidate':a.PRIOR,
       'created_at':'2026-10-03T16:01:50+00:00','bounded_mode':'bounded-material-fanin509-v1',
       'preregistration_sha256':a.PREREG_SHA256,'build_manifest_sha256':a.BUILD_SHA256,
       'all_groups_stopped':True,'comparison_valid':True,'adopt':False,'final_used':False,'inputs_unchanged':True,
       'executor_session_id':26958,'executor_exit_code':0,'recorded_cleanup_verified_candidate_attempts':1829,
       'recorded_cleanup_verified_baseline_attempts':1829,'source_context_unchanged':True,'failed_executor_history':history,
       'recovery_evidence':evidence,'exclusive_locks':a.LOCKS,'related_worker_paths':a.WORKERS,'related_executables':a.EXECUTABLES,
       'process_scans':[{'related_processes':[],'owned_processes_scanned':2}]*2,
       'candidate_model':ref(a.W),'inputs_before':snapshots,'inputs_after':copy.deepcopy(snapshots),
       'source_context_before':actual,'source_context_after':copy.deepcopy(actual),'source_helpers_sha256':{name:au['source_helpers_sha256'][name] for name in a.STOP_HELPERS}}
    for field,name in {'comparison_receipt':'comparison','independent_review_receipt':'independent_review',
       'operational_receipt':'operational','terminal_observation':'terminal','evaluation_receipt':'evaluation'}.items():stop[field]=ref(a.PATHS[name])
    plan={'schema':'sekirei.paired-linear-constrained-ridge-plan.v1','status':'fixed-before-fifth-formal-outcome-not-activated',
       'candidate':a.NEXT,'created_at':'2026-10-03T15:00:39+00:00','formal_selection_rule_unchanged':True,
       'baseline_best_updated':False,'final_used':False,'implementation_or_fit_activated':False}
    docs={'comparison':cp,'independent_review':au,'stop_receipt':stop,'operational':op,'terminal':term,'evaluation':ev,
          'failure_closure':failure,'failed_terminal':failedterm,'recovery_preflight':rp,'original_preflight':pf,
          'preregistration':{},'plan':plan}
    activation=a.make_activation(docs,actual,'2026-10-03T16:02:00+00:00')
    return activation,docs,actual


class ActivationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.activation,cls.docs,cls.actual=synthetic_bundle()
    def reject(self,mutate):
        activation,docs,actual=copy.deepcopy((self.activation,self.docs,self.actual));mutate(activation,docs,actual)
        with self.assertRaises(ValueError):a.validate_bundle(activation,docs,actual)

    def test_complete_control_bundle_and_claim_limits(self):
        result=a.validate_bundle(self.activation,self.docs,self.actual)
        self.assertEqual(result['status'],'verified');self.assertIs(result['best_baseline_updated'],False)
        self.assertIs(result['final_used'],False);self.assertEqual(len(self.docs['operational']['inputs_before']),1222)
        self.assertEqual(len(self.docs['operational']['recovery_inputs_before']),4207)

    def test_receipt_hash_typing_and_eight_identities(self):
        for name in a.FIXED_SHA:
            self.reject(lambda v,d,x,n=name:x[str(a.PATHS[n])].update(sha256='f'*64))
        self.reject(lambda v,d,x:x[str(a.W)].update(bytes=True))
        self.reject(lambda v,d,x:d['comparison']['checks']['occurrences'].update(equal=1))
        self.reject(lambda v,d,x:d['comparison']['checks'].pop('occurrences'))
        self.reject(lambda v,d,x:d['comparison']['checks']['occurrences'].update(mismatch_count=False))
        self.reject(lambda v,d,x:d['comparison'].update(adopt=True))

    def test_scope_attempts_cleanup_and_best_baseline(self):
        self.reject(lambda v,d,x:d['comparison']['baseline'].update(model_identity={'kind':'nnue'}))
        self.reject(lambda v,d,x:d['independent_review']['scope'].update(candidate_attempts=1829.0))
        self.reject(lambda v,d,x:d['independent_review']['run_audits']['candidate']['top3'].update(cleanup_ok=550))
        self.reject(lambda v,d,x:d['independent_review']['run_audits']['baseline']['mae'].update(technical_failures=1))
        self.reject(lambda v,d,x:d['stop_receipt'].update(recorded_cleanup_verified_candidate_attempts=1828))
        self.reject(lambda v,d,x:v.update(best_baseline_updated=True))

    def test_timing_stop_locks_process_scans_and_snapshot(self):
        self.reject(lambda v,d,x:d['plan'].update(created_at='2026-10-03T15:59:00+00:00'))
        self.reject(lambda v,d,x:v.update(created_at='2026-10-03T15:59:00+00:00'))
        self.reject(lambda v,d,x:d['stop_receipt']['exclusive_locks'].pop())
        self.reject(lambda v,d,x:d['stop_receipt']['process_scans'][0].update(related_processes=[{'pid':1}]))
        self.reject(lambda v,d,x:d['stop_receipt']['process_scans'][0].update(owned_processes_scanned=False))
        self.reject(lambda v,d,x:d['stop_receipt']['inputs_after'].pop(str(a.W)))
        self.reject(lambda v,d,x:d['stop_receipt']['candidate_model'].update(bytes=0))

    def test_honest_recovery_and_original_failed_history(self):
        self.reject(lambda v,d,x:d['terminal'].update(session_id=89125))
        self.reject(lambda v,d,x:d['terminal'].update(exit_code=False))
        self.reject(lambda v,d,x:d['terminal'].update(failed_executor_history=[]))
        self.reject(lambda v,d,x:d['operational']['execution_segments'][0].update(exit_code=0))
        self.reject(lambda v,d,x:d['operational']['execution_segments'][1].update(python='/usr/bin/python3'))
        self.reject(lambda v,d,x:d['operational']['execution_segments'][1].update(attempts=[36,550]))
        self.reject(lambda v,d,x:d['recovery_preflight'].update(measurement_settings_changed=True))
        self.reject(lambda v,d,x:d['independent_review']['recovery_evidence'].update(immutable_failure_snapshot_count=17))

    def test_union_collision_input_type_and_current_source(self):
        self.reject(lambda v,d,x:d['operational']['inputs_after'].pop(next(iter(d['operational']['inputs_after']))))
        self.reject(lambda v,d,x:d['original_preflight']['bound_files'][str(a.ORIGINAL_WORKER)].update(sha256='e'*64))
        self.reject(lambda v,d,x:d['recovery_preflight']['immutable_inputs'][str(a.RECOVERY_WORKER)].update(bytes=2))
        self.reject(lambda v,d,x:d['stop_receipt']['source_helpers_sha256'].pop('bounded_material.py'))
        self.reject(lambda v,d,x:x[str(a.R/'scripts/bounded_material.py')].update(sha256='e'*64))
        self.reject(lambda v,d,x:x[str(a.T/'source/crates/sekirei-core/src/nnue.rs')].update(sha256='e'*64))
        self.reject(lambda v,d,x:v.update(bound_inputs={}))

    def test_runtime_barrier_before_any_io_and_canonical_module(self):
        with mock.patch.object(a,'PROTOTYPE_ONLY',True),mock.patch.object(a,'file_bytes',side_effect=AssertionError('must not read')):
            with self.assertRaisesRegex(ValueError,'PROTOTYPE_ONLY'):a.verify_activation({'path':str(a.ACTIVATION),'sha256':'1'*64},'2'*64)
        with mock.patch.object(a,'PROTOTYPE_ONLY',False),mock.patch.object(a,'__file__','/tmp/not-public.py'),mock.patch.object(a,'file_bytes',side_effect=AssertionError('must not read')):
            with self.assertRaisesRegex(ValueError,'canonical public'):a.verify_activation({'path':str(a.ACTIVATION),'sha256':'1'*64},'2'*64)
        for value in ({'path':str(a.ACTIVATION),'sha256':True},{'path':'/tmp/activation.json','sha256':'1'*64}):
            with self.assertRaises(ValueError):a.validate_trigger_reference(value)
        for value in (b'{"x":1,"x":2}',b'{"x":Infinity}'):
            with self.assertRaises(ValueError):a.strict_json(value)

if __name__=='__main__':unittest.main()
