"""Only synthetic/memory/source fixtures; no actual model or data access."""
import ast
from contextlib import contextmanager
from copy import deepcopy
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

D=Path(__file__).parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(D))
import white_view_proof_common_v2 as p


def module(name,file):
    spec=importlib.util.spec_from_file_location(name,D/file);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

G=module('_white_gate_fixture','white-view-candidate-model-gate-worker-v2.py')
CORE=module('_white_core_fixture','white-view-candidate-core-proof-worker-v2.py')
INC=module('_white_incremental_fixture','white-view-candidate-incremental-proof-worker-v2.py')
SHA='a'*64

def ref(path,code=SHA):return {'path':str(path),'bytes':1,'sha256':code}
def rat(v):v=Fraction(v);return {'numerator':v.numerator,'denominator':v.denominator}

def arguments():
    names=('worker','common','preregistration','activation','source_preflight','fit_run','native','metadata','coefficients_f32',
        'coefficients_f64','solver_certificate','reference03','numeric_audit','numeric_worker','core','core_worker','incremental','incremental_worker')
    return SimpleNamespace(**{'expected_'+n+'_sha256':SHA for n in names})

def certificate():
    u=[rat(0)]*254
    result={'fallback_used':False,'lambda':1,'loss_is_half_unnormalized_sum':True,
        'saved_f32_gap_is_solver_stopping_criterion':False,'iterations':0,'restart_count':0,
        'one_if_needed_f64_correction_used':False,'one_if_needed_f64_correction_factor':rat(1-Fraction(1,2**40)),
        'lipschitz_exact':rat(1),'lipschitz_numeric_upper':rat(1),'coefficient_f64':u,'coefficient_f32':u}
    for key,radius in (('certificate_f64',p.RHO),('certificate_f32_solver_radius',p.RHO),('certificate_f32_saved_radius',Fraction(79,2))):
        result[key]={'coefficient_values':u,'norm':rat(0),'radius':rat(radius),'feasible':True,
            'gradient_numerators':[0]*254,'gradient_denominator':1,'fw_gap':rat(0),'fw_gap_per_sample':rat(0),
            'solver_threshold_met':True,'objective_delta_vs_zero':rat(0)}
    return result


def core_results():
    return {name:{'count':count,'positions_sha256':SHA,'stdin_sha256':SHA,'stdout_sha256':SHA,'stderr_sha256':SHA,
        'ordered_sfens_sha256':SHA,'returncode':0,'timeout':False,'cleanup_status':'ok',**p.CORE_FLAGS,
        'maximum_observed_candidate_material_difference_cp':0,'maximum_float_core_bridge_cp':0.0}
        for name,count in (('train',112681),('holdout',5895),('fixtures',15))}


def incremental():
    return {'positions_checked':8185,'fixtures':15,'walks':16,'search_walks':8,'captures':1,'promotions':1,'drops':1,'undos':1,
        'null_undos':1,'max_material_difference_cp':100,'incremental_refresh_error_cp':0,'accumulator_refresh_equal':True,
        'parent_restoration_equal':True,'observed_float_intermediates_finite':True,'mxcsr_control':'9fc0','material_bound_enabled':True}


def gate_fixture():
    args=arguments();inputs={str(p.Q/'run.json'):{'bytes':1,'sha256':SHA}}
    build={'runtime':str(p.NEW),'manifest':ref(p.NEW/'build-manifest.json'),'identity':ref(p.NEW/'white-view-build-identity.json'),
        'validator_source':ref(p.C/'build/prepare-white-view-runtime-worker-v1.py')}
    ctx={'binding':{'fit_receipt':ref(p.Q/'run.json')},'native_ref':ref(p.Q/'weights.bin'),'numeric_ref':ref(p.NUMERIC),
        'reference_ref':ref(p.REFERENCE),'feature_binding':{'native_magic':'SEKIRW03'},'preregistration':{'new_build_binding':build}}
    with patch.object(p,'bound_ref',side_effect=lambda reader,path,expected=None:ref(path)):
        gate=p.gate_document(args,None,ctx,{},inputs)
    return gate


class Tests(unittest.TestCase):
    def test_runtime_entries_block_before_io(self):
        for obj in (CORE,INC,G):
            with patch.object(p,'load_public',side_effect=AssertionError('I/O boundary crossed')):
                with self.assertRaises(ValueError):obj.run(SimpleNamespace())
        with patch.object(p.PR.__class__,'read_bytes',side_effect=AssertionError('I/O boundary crossed')):
            with self.assertRaises(ValueError):p.load_public(SimpleNamespace())
            with self.assertRaises(ValueError):G.verify_candidate_inputs(None,None,None)

    def test_reference_reconstruction_not_header_retag(self):
        white=SimpleNamespace(transform_initializer=lambda old,binding:b'full-transform',validate_hand_ties=lambda data:None,
            protected_bytes=lambda data:b'protected')
        self.assertEqual(p.reference_bytes(white,b'original',b'full-transform',{})['header_retag_only_accepted'],False)
        with self.assertRaises(ValueError):p.reference_bytes(white,b'original',b'retag-only',{})

    def test_core_candidate_reference_distinction(self):
        row={'index':0,'native_core_cp':110,'nearest_core_cp':10,'native_quantized_float_cp':110.5,
            'nearest_quantized_float_cp':10.0,'material_cp':10}
        self.assertEqual(p.core_row(json.dumps(row),0,10)['native_core_cp'],110)
        for key,value in (('native_core_cp',111),('nearest_core_cp',11),('material_cp',False),('index',True),
            ('native_quantized_float_cp',float('nan')),('nearest_quantized_float_cp',10.5)):
            changed={**row,key:value}
            with self.assertRaises(ValueError):p.core_row(json.dumps(changed),0,10)
        with self.assertRaises(ValueError):p.core_row(json.dumps(row),1,10)
        with self.assertRaises(ValueError):p.core_row('{"index":0,"index":0}',0,10)

    def test_core_strict_counts_and_cleanup(self):
        p.validate_core_results(core_results())
        for key,value in (('count',True),('returncode',False),('timeout',0),('cleanup_status','pending'),
            ('every_reference03_float_equals_material',False),('maximum_float_core_bridge_cp',float('inf'))):
            doc=core_results();doc['train'][key]=value
            with self.assertRaises(ValueError):p.validate_core_results(doc)
        doc=core_results();del doc['fixtures']
        with self.assertRaises(ValueError):p.validate_core_results(doc)

    def test_incremental_exact8185_and_flags(self):
        p.incremental_result(incremental())
        for key,value in (('positions_checked',8184),('null_undos',0),('accumulator_refresh_equal',1),
            ('parent_restoration_equal',False),('mxcsr_control','1fc0'),('max_material_difference_cp',101)):
            with self.assertRaises(ValueError):p.incremental_result({**incremental(),key:value})

    def test_three_numeric_artifact_certificates(self):
        doc=certificate();p.validate_solver_certificate(doc,b'\0'*2032,b'\0'*1016)
        doc['certificate_f64']['solver_threshold_met']=1
        with self.assertRaises(ValueError):p.validate_solver_certificate(doc,b'\0'*2032,b'\0'*1016)
        doc=certificate();doc['certificate_f32_saved_radius']['objective_delta_vs_zero']=rat(1)
        with self.assertRaises(ValueError):p.validate_solver_certificate(doc,b'\0'*2032,b'\0'*1016)

    def test_canonical_negative_zero_cast(self):
        f64=struct.pack('<d',-1e-300)+b'\0'*2024;doc=certificate()
        doc['coefficient_f64']=deepcopy(doc['coefficient_f64']);doc['coefficient_f64'][0]=rat(-1e-300)
        cert=doc['certificate_f64'];cert['coefficient_values']=deepcopy(doc['coefficient_f64']);cert['norm']=rat(Fraction(1e-300))
        p.validate_solver_certificate(doc,f64,b'\0'*1016)
        with self.assertRaises(ValueError):p.validate_solver_certificate(doc,f64,b'\0\0\0\x80'+b'\0'*1012)

    def test_full_gate_document_and_typed_corruption(self):
        gate=gate_fixture();p.validate_gate_document(gate,deepcopy(gate))
        for key,value in (('core_total_rows',118590),('incremental_observations',True),('cleanup_verified',1),
            ('new_build_unchanged',False),('adoption_verified',True),('final_used',True)):
            corrupted={**gate,key:value}
            with self.assertRaises(ValueError):p.validate_gate_document(corrupted,gate)
        corrupted=deepcopy(gate);corrupted['inputs_after'][str(p.Q/'run.json')]['sha256']='b'*64
        with self.assertRaises(ValueError):p.validate_gate_document(corrupted,gate)

    def test_binding_exact_refs_and_new_abi_paths(self):
        binding={'schema':'sekirei.white-view-candidate-gate-binding.v1',**{name:ref(path) for name,path in G.BINDING_PATHS.items()}}
        G.validate_binding(binding);self.assertEqual(G.arguments_from_binding(binding).expected_numeric_worker_sha256,SHA)
        for key,value in (('schema','old-Adam-E3'),('reference03',ref(p.Q/'reference03.bin')),('native',ref(p.Q/'weights.bin','A'*64)),
            ('numeric_audit',ref(p.C/'white-view-paired-linear-independent-numeric-audit-v1.json')),
            ('numeric_worker',ref(p.C/'white-view-paired-linear-independent-numeric-audit-worker-v1.py')),
            ('common',ref(p.C/'white_view_proof_common.py')),
            ('gate_worker',ref(p.C/'white-view-candidate-model-gate-worker-v1.py'))):
            with self.assertRaises(ValueError):G.validate_binding({**binding,key:value})
        with self.assertRaises(ValueError):G.validate_binding({**binding,'extra':None})

    def test_numeric_receipt_artifacts_and_independent_scope(self):
        cert=certificate();outputs={key:ref(p.Q/(key+'.synthetic')) for key in
            ('gram_z','rhs_z','coefficients_f64','coefficients_f32','solver_certificate','design_receipt','design_z','targets_d','weights')}
        ctx={'run':{'outputs':outputs},'binding':{'fit_receipt':ref(p.Q/'run.json')},
            'preregistration':{'source_helpers':{'synthetic.py':SHA},'new_build_binding':{}},'certificate':cert,
            'sidecar_ref':ref(p.Q/'weights.meta.json'),'reference_ref':ref(p.REFERENCE),'feature_binding':{}}
        values={r['path']:{k:r[k] for k in ('bytes','sha256')} for r in outputs.values()}
        values[str(p.Q/'run.json')]={'bytes':1,'sha256':SHA}
        mathproof={'schema':'sekirei.white-view-paired-linear-independent-numeric-math.v1','status':'complete',
            'sample_count':112681,'dimension':254,'all_three_certificates_exactly_recomputed':True,
            'saved_f32_nearest_canonical_cast_verified':True,'stored_gradient_matches_Hu_minus_b':True,
            'raw_design_extraction_replayed':False,'gram_product_recomputed':False,'gram_psd_independently_reproved':False,
            'producer_origin_acknowledged_separately':True,'solver_trajectory_replayed':False,'second_fit_run':False,
            'native_core_proof_claimed':False,'final_used':False,'adoption_claimed':False,
            'recomputed_certificates':{k:cert[k] for k in ('certificate_f64','certificate_f32_solver_radius','certificate_f32_saved_radius')}}
        for key in ('gram_bytes_sha256','rhs_bytes_sha256','coefficient_f64_bytes_sha256','coefficient_f32_bytes_sha256',
            'solver_certificate_sha256','design_receipt_sha256','design_z_sha256','targets_d_sha256'):mathproof[key]=SHA
        doc={'schema':'sekirei.white-view-paired-linear-independent-numeric-audit.v1','status':'complete',
            'candidate':p.CANDIDATE,'plan_sha256':p.PLAN_SHA,'fit_receipt':ctx['binding']['fit_receipt'],
            'worker_sha256':SHA,'source_helpers_sha256':ctx['preregistration']['source_helpers'],
            'inputs_unchanged':True,'source_unchanged':True,'no_child_process_started':False,'no_engine_or_fit_process_started':True,
            'runtime_validator_may_invoke_readonly_source_compiler_checks':True,'actual_fit_repeated':False,
            'native_core_proof_claimed':False,'final_used':False,'adoption_claimed':False,
            'inputs_before':deepcopy(values),'inputs_after':deepcopy(values),'numerical_math':mathproof,
            'native':outputs['weights'],'metadata':ctx['sidecar_ref'],'reference03':ctx['reference_ref'],
            'coefficients_f64':outputs['coefficients_f64'],'coefficients_f32':outputs['coefficients_f32'],
            'solver_certificate':outputs['solver_certificate'],'design_receipt':outputs['design_receipt'],
            'feature_binding':{},'new_build_binding':{},'new_build_verified':True,
            'forward_artifact_binding':{'native03_full_reconstruction_verified':True,'sidecar_complete_run_binding_verified':True,
            'reference03_full_transform_verified':True,'reference03_sha256':SHA,'native_magic':'SEKIRW03',
            'feature_schema':'flat_white_view_aux_tied_v1','coefficient_nearest_cast_core_forward_claimed':False,
            'actual_native_core_forward_verified':False,'actual_native_core_covariance_verified':False,
            'actual_incremental_verified':False,'forward_proof_requires_separate_new_runtime_core_execution':True}}

        p.numeric_receipt_fields(doc,ctx,SHA)
        for key,value in (('all_three_certificates_exactly_recomputed',1),('stored_gradient_matches_Hu_minus_b',False),
            ('gram_bytes_sha256','b'*64),('second_fit_run',True)):
            changed=deepcopy(doc);changed['numerical_math'][key]=value
            with self.assertRaises(ValueError):p.numeric_receipt_fields(changed,ctx,SHA)
        changed=deepcopy(doc);del changed['inputs_before'][outputs['weights']['path']];changed['inputs_after']=deepcopy(changed['inputs_before'])
        with self.assertRaises(ValueError):p.numeric_receipt_fields(changed,ctx,SHA)

    def test_shared_lifecycle_and_math_source_unchanged(self):
        old=D.parent/'white-view-source-fixtures-v1'/'paired-proof-common-v2.py.txt';raw=old.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),p.SHARED_SOURCE_SHA)
        def nodes(raw):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        oldnodes=nodes(raw);newnodes=nodes((D/'white_view_proof_common_v2.py').read_bytes())
        for name in p.SHARED_FUNCTIONS:self.assertEqual(oldnodes[name],newnodes[name],name)

    def test_separate_abi_no_old_fit_validation_delegation(self):
        source=(D/'white_view_proof_common_v2.py').read_text()
        for forbidden in ('verify_epochs(', 'prepare_bounded.verify_build(', 'verify_checkpoint_metadata(', 'diagnose_bounded.'):
            self.assertNotIn(forbidden,source)
        self.assertIn("'selfload_pair':False",source);self.assertIn("'native_nearest_quantizer_reexport_claimed':False",source)
        self.assertIn('worker.c is contract',source)

    def test_private_adjacent_import_restores_public_cache(self):
        class Reader:
            def pin(self,*args):pass
            def read(self,*args):pass
        public=object();contract=SimpleNamespace(__file__='/tmp/build/white_view_build_contract.py')
        worker=SimpleNamespace(c=contract)
        prereg={'new_build_binding':{'validator_source':ref(Path('/tmp/build/worker.py'))},'source_helpers':{'white_view_build_contract.py':SHA}}
        with patch.dict(sys.modules,{'white_view_build_contract':public}):
            with patch.object(p,'load_bound_module',side_effect=[contract,worker]):
                self.assertIs(p.load_runtime_validator(Reader(),prereg),worker)
                self.assertIs(sys.modules['white_view_build_contract'],public)
            with patch.object(p,'load_bound_module',side_effect=[contract,ValueError('synthetic import failure')]):
                with self.assertRaises(ValueError):p.load_runtime_validator(Reader(),prereg)
                self.assertIs(sys.modules['white_view_build_contract'],public)

if __name__=='__main__':unittest.main()
