"""Synthetic source/control/math/lifecycle fixture; no actual process/artifact I/O."""
import ast
from fractions import Fraction
import importlib.util
import math
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'preparations/white-view-fit-operational-v2'
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(1,str(HERE))
import white_view_fit_operational_common as common
import white_view_fit_contract as contract
import fit_white_view_paired_linear as driver
import diagnose_anchor as guard


def worker(name):
    spec=importlib.util.spec_from_file_location(name.replace('-','_'),HERE/(name+'.py'))
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
ACT=worker('white-view-paired-linear-activation-worker-v1')
PF=worker('white-view-paired-linear-source-preflight-worker-v1')
LAUNCH=worker('white-view-paired-linear-fit-launcher-v1')


def previous():
    checks={name:{'equal':True,'baseline_sha256':'a'*64,'candidate_sha256':'a'*64,
        **({'mismatch_count':0} if name in common.DICT_CHECKS else {})} for name in common.CHECKS}
    shared={'status':'complete','comparison_valid':True,'adopt':False,'inputs_unchanged':True,'final_used':False,
        'mae_improved':True,'top3_preserved':False,'checks':checks}
    cp={'schema':'sekirei.formal-comparison.v1',**shared,'candidate':{'model_identity':{'sha256':common.PRIOR_MODEL_SHA256}}}
    au={'schema':'sekirei.paired-linear-independent-formal-review.v1',**shared,
        'candidate_model':{'sha256':common.PRIOR_MODEL_SHA256},
        'scope':{'candidate_attempts':1829,'baseline_attempts':1829,'requested_nodes':1000000,'games':5,'teacher_E':266,'top3_denominator':551},
        'run_audits':{subject:{stage:{'attempts':count,'cleanup_ok':count,'runner_exit_zero':count,
            'raw_sha256_and_lifecycle_verified':count,'technical_failures':0,'timeout_failures':0}
            for stage,count in common.STAGES.items()} for subject in ('candidate','baseline')}}
    st={'schema':'sekirei.paired-linear-formal-stopped.v1','status':'verified-stopped-under-locks','candidate':common.PRIOR,
        'all_groups_stopped':True,'comparison_valid':True,'adopt':False,'final_used':False,'inputs_unchanged':True,
        'executor_session_id':91053,'executor_exit_code':0,'recorded_cleanup_verified_candidate_attempts':1829,
        'recorded_cleanup_verified_baseline_attempts':1829,
        'exclusive_locks':[str(x) for x in (common.T/'.training.lock',common.B/'.prepare.lock',common.B/'.benchmark.lock',
            common.F/'.build.lock',common.F/'.training.lock',common.C.parent/'training-17-v1/paired-linear-constrained-ridge1-v1/.fit.lock')],
        'inputs_before':{'/synthetic/input':{'bytes':1,'sha256':'a'*64}},
        'inputs_after':{'/synthetic/input':{'bytes':1,'sha256':'a'*64}},
        'comparison_receipt':common.PREVIOUS['comparison'],'independent_review_receipt':common.PREVIOUS['independent_review'],
        'process_scans':[{'related_processes':[],'owned_processes_scanned':1}]*2}
    return {'comparison':cp,'independent_review':au,'stopped':st}


def synthetic_numeric():
    G=(256,0,0,256);h=(32,0);u=(1.,0.)
    result={'lambda':1,'loss_is_half_unnormalized_sum':True,'fallback_used':False,
        'saved_f32_gap_is_solver_stopping_criterion':False,'coefficient_f64':driver.rational_json(u),
        'coefficient_f32':driver.rational_json(u),'lipschitz_exact':driver.rational_json(Fraction(2)),
        'lipschitz_numeric_upper':driver.rational_json(math.nextafter(2.,math.inf)),
        'one_if_needed_f64_correction_factor':driver.rational_json(1.-2.**-40),'one_if_needed_f64_correction_used':False}
    for name,radius in (('certificate_f64',Fraction(161791,4096)),('certificate_f32_solver_radius',Fraction(161791,4096)),
                        ('certificate_f32_saved_radius',Fraction(79,2))):
        result[name]=driver.rational_json({'norm':Fraction(1),'radius':radius,'feasible':True,'fw_gap':Fraction(0),
            'fw_gap_per_sample':Fraction(0),'solver_threshold_met':True,'objective_delta_vs_zero':Fraction(-1),
            'gradient_numerators':(0,0),'gradient_denominator':256,'coefficient_values':u})
    return result,struct.pack('<4q',*G),struct.pack('<2q',*h),struct.pack('<2d',*u),struct.pack('<2f',*u)

class Process:
    pid=12345
    def __init__(self,returncode=0,error=None):self.returncode=returncode;self.error=error
    def wait(self,timeout=None):
        if self.error:raise self.error
        return self.returncode

class OperationalTests(unittest.TestCase):
    def test_previous_source_contract(self):self.assertTrue(common.validate_previous(previous(),common.PREVIOUS))
    def test_previous_reject_invalid_reused_or_missing_cleanup(self):
        for key,value in (('comparison_valid',False),('adopt',True),('final_used',True)):
            docs=previous();docs['comparison'][key]=value
            with self.assertRaises(ValueError):common.validate_previous(docs,common.PREVIOUS)
        docs=previous();docs['independent_review']['run_audits']['candidate']['mae']['cleanup_ok']=1139
        with self.assertRaises(ValueError):common.validate_previous(docs,common.PREVIOUS)
    def test_previous_check_exact_producer_shapes(self):
        for name in common.CHECKS:
            docs=previous()
            if name in common.DICT_CHECKS:docs['comparison']['checks'][name].pop('mismatch_count')
            else:docs['comparison']['checks'][name]['mismatch_count']=0
            with self.assertRaises(ValueError):common.validate_previous(docs,common.PREVIOUS)
    def test_previous_stop_snapshot_and_two_scans(self):
        for change in ('map','scans','locks'):
            docs=previous()
            if change=='map':docs['stopped']['inputs_after']['/synthetic/input']['bytes']=False
            elif change=='scans':docs['stopped']['process_scans']=[{'related_processes':[],'owned_processes_scanned':1}]
            else:docs['stopped']['exclusive_locks'].pop()
            with self.assertRaises(ValueError):common.validate_previous(docs,common.PREVIOUS)
    def test_exact_numeric_two_dimensions_no_fit(self):
        self.assertTrue(common.validate_numeric_artifacts(*synthetic_numeric(),dimension=2,samples=1)['all_three_certificates_exactly_recomputed'])
    def test_numeric_gradient_objective_and_rounded_coefficient_tamper(self):
        for name in ('gradient_numerators','gradient_denominator','objective_delta_vs_zero','norm','radius'):
            args=synthetic_numeric();args[0]['certificate_f64'][name]=1
            with self.assertRaises(ValueError):common.validate_numeric_artifacts(*args,dimension=2,samples=1)
        result,G,h,f64,f32=synthetic_numeric()
        with self.assertRaises(ValueError):common.validate_numeric_artifacts(result,G,h,f64,struct.pack('<2f',1.25,0.),dimension=2,samples=1)
    def test_numeric_gershgorin_recipe_tamper(self):
        for key,value in (('lambda',2),('fallback_used',True),('lipschitz_exact',{'numerator':1,'denominator':1}),
            ('one_if_needed_f64_correction_factor',{'numerator':1,'denominator':1})):
            args=synthetic_numeric();args[0][key]=value
            with self.assertRaises(ValueError):common.validate_numeric_artifacts(*args,dimension=2,samples=1)
    def test_runtime_barriers_before_inputs_and_process(self):
        for func in (ACT.activate,PF.main,LAUNCH.launch):
            with self.assertRaisesRegex(ValueError,'SOURCE ONLY'):func(object())
        with self.assertRaisesRegex(ValueError,'SOURCE ONLY'):common.require_enabled()
    def test_six_lock_partition_exact(self):
        self.assertEqual(len(common.lock_paths()),6)
        self.assertEqual(common.lock_paths()[-1],contract.Q.parent/'.fit.lock')
        self.assertEqual(common.lock_paths()[3],common.F/'.build.lock')
        self.assertEqual(common.architecture_shared_locks(),[contract.RUNTIME/'.build.lock',contract.RUNTIME/'.prepare.lock'])
    def lifecycle(self):return {'spawned':False,'waited':False,'reaped':False,'returncode':None,
        'timed_out':False,'process_group_stopped':False,'process_group_scans':[],'process_handle_retained':False}
    def test_supervisor_zero_exit_real_wait_modelled(self):
        child=Process();holder={'process':None};life=self.lifecycle()
        def spawn(command,output,stdout,stderr,h):h['process']=child
        with patch.object(guard,'spawn_process',side_effect=spawn),patch.object(guard,'cleanup_group'),patch.object(guard,'group_exists',return_value=False):
            LAUNCH.run_child(['synthetic'],None,None,None,holder,life)
        self.assertTrue(life['waited'] and life['reaped'] and life['process_group_stopped']);self.assertIsNone(holder['process'])
    def test_spawn_cancel_handle_retained_until_cleanup(self):
        child=Process();holder={'process':None};life=self.lifecycle()
        def spawn(command,output,stdout,stderr,h):h['process']=child;raise RuntimeError('synthetic cancellation after spawn')
        with patch.object(guard,'spawn_process',side_effect=spawn),patch.object(guard,'cleanup_group'),patch.object(guard,'group_exists',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'cancellation'):LAUNCH.run_child([],None,None,None,holder,life)
        self.assertTrue(life['waited'] and life['reaped']);self.assertIsNone(holder['process'])
    def test_cleanup_failure_never_forges_wait_or_reap(self):
        child=Process(error=RuntimeError('wait failed'));holder={'process':None};life=self.lifecycle()
        def spawn(command,output,stdout,stderr,h):h['process']=child
        with patch.object(guard,'spawn_process',side_effect=spawn),patch.object(guard,'cleanup_group',side_effect=RuntimeError('cleanup failed')):
            with self.assertRaisesRegex(RuntimeError,'wait failed'):LAUNCH.run_child([],None,None,None,holder,life)
        self.assertFalse(life['waited']);self.assertFalse(life['reaped']);self.assertFalse(life['process_group_stopped']);self.assertIs(holder['process'],child)
        self.assertIn('cleanup failed',life['cleanup_error']);self.assertIn('wait failed',life['original_error'])
    def test_children_do_not_inherit_blocked_signal_mask(self):
        tree=ast.parse(Path(guard.__file__).read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='spawn_process')
        names=[n.func.id for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        self.assertIn('deferred_termination',names);self.assertNotIn('blocked_termination',names)
    def test_inherited_replay_raw1000_original_validation_still_literal(self):
        source=Path(PF.__file__).read_text()
        for term in ('len(replay_inputs) == 1042','len(raw) == 1000','original.validate_original',
            "len(core['files']) == 617","len(source['fixed_source_tree']) == 526", "len(source['fixed_build_dependencies']) == 63"):
            self.assertIn(term,source)
    def test_parser_external_sha_contracts(self):
        for module in (ACT,PF,LAUNCH):
            options={x.dest for x in module.parser()._actions}
            self.assertTrue({'expected_worker_sha256','expected_common_sha256','expected_public_source_head',
                'expected_build_manifest_sha256','expected_build_identity_sha256','expected_build_validator_sha256'}<=options)
        self.assertIn('expected_source_inventory_sha256',{x.dest for x in ACT.parser()._actions})
    def test_ast_all_new_sources(self):
        for path in HERE.glob('*.py'):ast.parse(path.read_text())

if __name__=='__main__':unittest.main()
