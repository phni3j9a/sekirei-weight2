"""Tiny synthetic memory tests only; never reads actual O/control/model artifacts."""
import ast
import hashlib
import json
from pathlib import Path
import struct
import sys
import unittest
import importlib.util
from unittest import mock
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
HAS_NUMPY=importlib.util.find_spec('numpy') is not None
import fit_white_view_paired_linear as fit
import white_view_fit_contract as c
import white_view_paired_linear as white
import functional_anchor as anchor
import material_init as material

SFENS=('4k4/9/9/9/9/9/9/9/4K4 b - 1',
       '4k4/9/9/4p4/9/3P5/9/9/4K4 b PLns 1',
       '4k4/9/9/9/4+r4/9/2+B6/9/4K4 w 2Pn 1')

def enc(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def reference(path,data=b'public synthetic'):
    return {'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def binding():
    return {'schema':'sekirei.white-view-paired-linear-feature-binding.v1','mode':white.MODE,
        'plan_sha256':white.PLAN_SHA256,'feature_schema':white.FEATURE_SCHEMA,
        'compile_feature':white.COMPILE_FEATURE,'native_magic':'SEKIRW03',
        'dimensions':{'input':2420,'l1':256,'l2':32},'stock_initializer_sha256':white.STOCK_INITIALIZER_SHA256,
        'feature_source_sha256':dict(c.FEATURE_HASHES),'core_binary_sha256':'b'*64,
        'helper_source_sha256':c.HELPER_SHA256}
def build_binding():
    return {'runtime':str(c.RUNTIME),'manifest':reference(c.RUNTIME/'build-manifest.json'),
        'identity':reference(c.RUNTIME/'white-view-build-identity.json'),
        'validator_source':reference('/synthetic/prepare-white-view-runtime-worker-v1.py')}
def helpers():return {**fit.NUMERIC_HASHES,'white_view_fit_contract.py':'c'*64,
    'fit_white_view_paired_linear.py':'d'*64,'white_view_build_contract.py':'e'*64}
def numeric_execution():
    return {'schema':'sekirei.white-view-paired-linear-numeric-execution.v1',
        'whole_child_budget_seconds':1200,'parent_process_group_deadline_required':True,
        'boundary_checks_cannot_interrupt_blas':True,'single_gram_produced':True,'single_fixed_fit':True,
        'lambda':1,'max_iterations':20000,'loss_half_unnormalized_sum':True,
        'saved_f32_positive_zero':True,'actual_cleanup_claimed':False,'remaining_inner_budget_seconds':1100.,
        'fit_wall_seconds':1.,'whole_child_elapsed_seconds':2.,'iterations':22,'restart_count':0,
        'openblas':{'schema':'sekirei.numpy-openblas-thread-proof.v1','numpy_version':'1.26.4',
            'library':reference('/synthetic/numpy.libs/libopenblas64_p-example.so'),
            'threads_before':1,'threads_during':1,'threads_after':1,'rtld_noload':True,'threadpool_verified':True,
            'get_symbol':'openblas_get_num_threads64_','set_symbol':'openblas_set_num_threads64_',
            'config_symbol':'openblas_get_config64_','config':'synthetic USE64BITINT',
            'numpy_preimported_when_driver_loaded':False,'environment':{key:'1' for key in fit.THREAD_ENV}}}
def context():
    return {'preregistration_sha256':'1'*64,'activation_sha256':'2'*64,'source_helpers_sha256':helpers(),
        'source_preflight_sha256':'3'*64,'inputs_before':{'/synthetic/input':{'bytes':3,'sha256':'4'*64}},
        'inputs_after':{'/synthetic/input':{'bytes':3,'sha256':'4'*64}},
        'runtime_sources':{'/synthetic/runtime':{'bytes':3,'sha256':'4'*64}},
        'numeric_execution':numeric_execution(),'feature_binding':binding(),
        'new_build_binding':build_binding(),'new_build_verified':True}

def manifest_data(sfens=SFENS,label_edit=None,position_edit=None):
    gid=next(f'{i:064x}' for i in range(100) if anchor._split_for_game(f'{i:064x}','synthetic-seed')=='train')
    positions=[{'schema_version':1,'sfen':sfen,'source':{'kind':'gensfen-pack','path':gid}} for sfen in sfens]
    labels=[{'sfen':sfen,'score_cp':100+i,'teacher_identity':white.ORIGINAL_TEACHER,'label_depth':0,
             'extra_metadata':'retained via original byte hash'} for i,sfen in enumerate(sfens)]
    if label_edit:label_edit(labels)
    if position_edit:position_edit(positions)
    pb=b''.join(enc(row) for row in positions);lb=b''.join(enc(row) for row in reversed(labels))
    placeholder=enc({'synthetic':'declared holdout not parsed'})
    files={name:{k:reference('/synthetic/'+name,data)[k] for k in ('bytes','sha256')}
        for name,data in [('train.positions.jsonl',pb),('train.labels.jsonl',lb),
                          ('holdout.positions.jsonl',placeholder),('holdout.labels.jsonl',placeholder)]}
    manifest={'schema_version':1,'files':files,'positions':{'train':len(sfens),'holdout':1},
        'source_corpus_manifest_sha256':white.ORIGINAL_TEACHER.rsplit(':',1)[1],
        'teacher_identity':white.ORIGINAL_TEACHER,'seed':'synthetic-seed','split':anchor.SPLIT_RECIPE,
        'sampling':anchor.SAMPLING,'games_per_pack':1000,'dependencies':{'synthetic':'1'},
        'independent_exclusions':{'games':1000,'policy':anchor.EXCLUSION_POLICY,'unique_positions':1,
            'corpus_canonical_sha256':'a'*64,'source_manifest_sha256':'b'*64},
        'derivation':{'kind':'expanded-train-frozen-holdout-v1','input_sha256':{'synthetic':'c'*64}},
        'games':[{'game_id':gid,'pack_sha256':'d'*64,'game_index':0,'split':'train'}]}
    mb=enc(manifest);return mb,pb,lb

def control_records():
    pr={'schema':'sekirei.white-view-paired-linear-preregistration.v1','status':'frozen-before-fit',
        'candidate':c.MODE,'mode':c.MODE,'plan_sha256':c.PLAN_SHA256,'dataset':str(c.DATASET),
        'dataset_manifest_sha256':c.MANIFEST_SHA256,'teacher_identity':white.ORIGINAL_TEACHER,
        'counts':dict(c.COUNTS),'fit':dict(c.FIT),'output':str(c.Q),'fit_started':False,'final_used':False,
        'adoption_claimed':False,'feature_binding':binding(),'new_build_binding':build_binding(),
        'source_helpers':helpers(),'runtime_sources':{'/synthetic/runtime':{'bytes':3,'sha256':'4'*64}},
        'fit_driver':reference('/synthetic/fit_white_view_paired_linear.py'),
        'initial_weights':{'path':str(c.ORIGINAL_INITIALIZER),'bytes':white.WEIGHT_BYTES,'sha256':white.STOCK_INITIALIZER_SHA256},
        'trigger':reference(c.ACTIVATION),'source_preflight':{'path':str(c.SOURCE_PREFLIGHT),'state':'required-before-fit'}}
    act={'schema':'sekirei.white-view-paired-linear-activation.v1','status':'verified-before-fit',
        'candidate':c.MODE,'mode':c.MODE,'plan_sha256':c.PLAN_SHA256,
        'previous_candidate_valid_nonadopt':True,'previous_candidate_groups_stopped':True,
        'previous_archive_verified':True,'previous_public_docs_pushed':True,'new_build_verified':True,
        'source_preparation_reviewed':True,'inputs_unchanged':True,'source_unchanged':True,
        'fit_started':False,'final_used':False,'adoption_claimed':False,
        'feature_binding':pr['feature_binding'],'new_build_binding':pr['new_build_binding'],
        'previous_completion':{key:reference('/synthetic/'+key) for key in
            ('comparison','independent_review','stopped','archive','public_docs')},
        'inputs_before':{'/synthetic/control':{'bytes':1,'sha256':'9'*64}},
        'inputs_after':{'/synthetic/control':{'bytes':1,'sha256':'9'*64}}}
    pf={'schema':'sekirei.white-view-paired-linear-original-source-preflight.v1','status':'complete',
        'candidate':c.MODE,'plan_sha256':c.PLAN_SHA256,'preregistration_sha256':'1'*64,
        'activation_sha256':pr['trigger']['sha256'],'source_helpers_sha256':pr['source_helpers'],
        'counts':c.COUNTS,'teacher_identity':white.ORIGINAL_TEACHER,'original_manifest_sha256':c.MANIFEST_SHA256,
        'feature_binding':pr['feature_binding'],'new_build_binding':pr['new_build_binding'],
        'original_labels_reparsed':True,'raw1000_membership_reverified':True,'raw_replay_rerun':False,
        'original_legal_replay_inherited':True,'whole_pool_exclusions_reverified':True,
        'source_unchanged':True,'inputs_unchanged':True,'new_build_verified':True,'final_used':False,'fit_started':False,
        'inherited_replay_receipt':reference('/synthetic/replay'),
        'inherited_source_preflight':reference('/synthetic/original-source')}
    records=[reference(c.PLAN),reference(c.DATASET/'manifest.json'),*
        (reference(c.DATASET/name) for name in ('train.positions.jsonl','train.labels.jsonl','holdout.positions.jsonl','holdout.labels.jsonl')),
        pr['initial_weights'],pr['trigger'],*
        (pr['new_build_binding'][key] for key in ('manifest','identity','validator_source')),
        pf['inherited_replay_receipt'],pf['inherited_source_preflight']]
    before={ref['path']:{key:ref[key] for key in ('bytes','sha256')} for ref in records}
    before[str(c.PLAN)]['sha256']=c.PLAN_SHA256
    before[str(c.DATASET/'manifest.json')]['sha256']=c.MANIFEST_SHA256
    pf['inputs_before']=before;pf['inputs_after']=json.loads(json.dumps(before))
    return pr,act,pf

class DriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original=material.encode(material.build_weights(42));cls.bound=binding()
        cls.init03=white.transform_initializer(cls.original,binding=cls.bound)
    def design(self,**kwargs):
        mb,pb,lb=manifest_data(**kwargs)
        return fit.build_train_design(mb,pb,lb,self.original,expected_manifest_sha256=fit.sha(mb),
            feature_binding=self.bound)
    @unittest.skipUnless(HAS_NUMPY,'requires fixed NumPy audit environment')
    def test_numpy_adapter_matches_pure_reference(self):
        design=self.design();expected=[];targets=[]
        for i,sfen in enumerate(SFENS):
            Z,M,_=white.design_row(material.parse_sfen(sfen),self.init03,self.original,binding=self.bound)
            expected.extend(Z);targets.append(100+i-M)
        self.assertEqual(design.z_bytes,struct.pack('<762b',*expected))
        self.assertEqual(design.d_bytes,struct.pack('<3i',*targets))
        self.assertEqual(design.provenance['material_initializer_sha256'],white.STOCK_INITIALIZER_SHA256)
        self.assertEqual(design.provenance['transformed_initializer_sha256'],white.sha(self.init03))
        self.assertFalse(design.provenance['whole_dataset_replay_proof_claimed'])
        self.assertFalse(design.provenance['actual_build_verified_by_builder'])
    @unittest.skipUnless(HAS_NUMPY,'requires fixed NumPy audit environment')
    def test_covariance_tiny_numpy_and_reference(self):
        np=fit.numpy();ft=np.frombuffer(self.init03,dtype='<i2',offset=8,count=2420*256).reshape(2420,256)[:,2:]
        for sfen in SFENS:
            p=material.parse_sfen(sfen)
            for q in (p,white.rotated_color_swap(p),{**p,'stm':1-p['stm']}):
                us=white.active_features(q,q['stm']);them=white.active_features(q,1-q['stm'])
                z=((ft[list(us),:].sum(axis=0,dtype=np.int16)-ft[list(them),:].sum(axis=0,dtype=np.int16))//2).astype(np.int8)
                self.assertEqual(tuple(int(v) for v in z),white.design_row(q,self.init03,self.original,binding=self.bound)[0])
    def test_no_retag_or_arbitrary_initializer(self):
        mb,pb,lb=manifest_data()
        with self.assertRaises(ValueError):fit.build_train_design(mb,pb,lb,self.init03,
            expected_manifest_sha256=fit.sha(mb),feature_binding=self.bound)
    def test_original_manifest_filehash_guards(self):
        mb,pb,lb=manifest_data()
        for args in ((mb,pb+b' ',lb),(mb,pb,lb+b' ')):
            with self.assertRaises(ValueError):fit.build_train_design(*args,self.original,
                expected_manifest_sha256=fit.sha(mb),feature_binding=self.bound)
    @unittest.skipUnless(HAS_NUMPY,'requires fixed NumPy audit environment')
    def test_teacher_depth_and_join_reject(self):
        for changed in ({'score_cp':True},{'score_cp':1.0},{'score_cp':30000},
                        {'label_depth':True},{'teacher_identity':'external:other'},{'sfen':'missing'}):
            with self.assertRaises(ValueError):self.design(label_edit=lambda rows:rows[0].update(changed))
    @unittest.skipUnless(HAS_NUMPY,'requires fixed NumPy audit environment')
    def test_duplicate_semantic_and_wrong_source(self):
        with self.assertRaises(ValueError):self.design(sfens=(SFENS[0],SFENS[0].replace(' 1',' 2')))
        with self.assertRaises(ValueError):self.design(position_edit=lambda rows:rows[0]['source'].update(path='e'*64))
    def test_deadline_during_preparation(self):
        mb,pb,lb=manifest_data()
        def expired():raise ValueError('synthetic expired')
        with self.assertRaisesRegex(ValueError,'expired'):fit.build_train_design(mb,pb,lb,self.original,
            expected_manifest_sha256=fit.sha(mb),feature_binding=self.bound,check_deadline=expired)
    def test_nearest_cast_negative_tiny_zero(self):
        r={'coefficient_f64':(-1e-50,)+(0.,)*253,'coefficient_f32':(0.,)*254}
        f64,f32=fit.coefficient_bytes(r);self.assertEqual(f32,b'\0'*1016)
        r['coefficient_f32']=(.1,)+(0.,)*253
        with self.assertRaises(ValueError):fit.coefficient_bytes(r)
    def test_parent_reaping_and_sidecar_one_way(self):
        zero=b'\0'*1016;native=white.serialize_coefficients(zero,self.original,binding=self.bound)
        outputs={key:reference(c.Q/name) for key,name in c.OUTPUT_NAMES.items()}
        outputs['weights']=reference(c.W,native);outputs['coefficients_f32']=reference(c.Q/'coefficients.f32.bin',zero)
        ctx=context();draft=fit.make_run_template(ctx,outputs,outputs['solver_certificate'])
        self.assertIs(draft['cleanup_verified'],False)
        for change in ({'reaped':False},{'returncode':1},{'process_group_stopped':False},{'waited':False},{'timed_out':True}):
            with self.assertRaises(ValueError):fit.construct_fit_run(ctx,outputs,outputs['solver_certificate'],{**c.LIFECYCLE,**change})
        run=fit.construct_fit_run(ctx,outputs,outputs['solver_certificate'],c.LIFECYCLE)
        raw=enc(run)
        bound={'plan_sha256':c.PLAN_SHA256,'preregistration_sha256':ctx['preregistration_sha256'],
            'activation_sha256':ctx['activation_sha256'],'source_preflight_sha256':ctx['source_preflight_sha256'],
            'source_helpers_sha256':ctx['source_helpers_sha256'],'feature_binding':ctx['feature_binding'],
            'new_build_binding':ctx['new_build_binding'],'fit_receipt':reference(c.FIT_RECEIPT,raw),
            'solver_certificate':outputs['solver_certificate']}
        meta=c.sidecar_for_fit(native,zero,self.original,raw,bound)
        self.assertTrue(c.validate_sidecar(enc(meta),native,zero,self.original,raw,bound))
        self.assertNotIn('paired_linear_constrained_ridge_v1',meta)
        with self.assertRaises(ValueError):c.sidecar_for_fit(native,zero,self.original,enc(draft),bound)
    def test_outputs_selfhash_and_typed_maps_reject(self):
        out={k:reference(c.Q/n) for k,n in c.OUTPUT_NAMES.items()}
        for path in (c.FIT_RECEIPT,c.Q/'weights.meta.json',c.Q/'fit-result.json'):
            changed={**out,'weights':reference(path)}
            with self.assertRaises(ValueError):fit.make_run_template(context(),changed,changed['solver_certificate'])
        ctx=context();ctx['inputs_after']['/synthetic/input']['bytes']=True
        with self.assertRaises(ValueError):fit.make_run_template(ctx,out,out['solver_certificate'])
    def test_identity_map_extra_path_rejected(self):
        with self.assertRaises(ValueError):c.identity_map({'/synthetic/map-key':{'path':'/synthetic/replacement','bytes':1,'sha256':'a'*64}})
    def test_strict_json_nonfinite_and_duplicate(self):
        for raw in (b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":1e1000}'):
            with self.assertRaises(ValueError):c.strict_json(raw)
    def test_dedicated_control_schemas(self):
        pr,act,pf=control_records()
        self.assertIs(c.validate_preregistration(pr),pr)
        self.assertIs(c.validate_activation(act,pr),act)
        self.assertIs(c.validate_source_preflight(pf,pr,'1'*64),pf)
        for target,key,value in (('pr','mode','paired-linear-constrained-ridge1-l1-39p5-v1'),
            ('act','previous_archive_verified',False),('pf','original_labels_reparsed',False),
            ('pf','raw_replay_rerun',True),('pf','whole_pool_exclusions_reverified',0),('pf','new_build_verified',False)):
            pr,act,pf=control_records();record={'pr':pr,'act':act,'pf':pf}[target];record[key]=value
            with self.assertRaises(ValueError):
                if target=='pr':c.validate_preregistration(pr)
                elif target=='act':c.validate_activation(act,pr)
                else:c.validate_source_preflight(pf,pr,'1'*64)
    def test_control_identity_cannot_be_rebound(self):
        for path in (c.PLAN,c.DATASET/'manifest.json',c.ORIGINAL_INITIALIZER,c.ACTIVATION):
            pr,act,pf=control_records();pf['inputs_before'][str(path)]['sha256']='0'*64
            pf['inputs_after'][str(path)]['sha256']='0'*64
            with self.assertRaises(ValueError):c.validate_source_preflight(pf,pr,'1'*64)
    def test_numeric_budget_pool_and_fixed_recipe_fail_closed(self):
        for key,value in (('lambda',2),('max_iterations',20001),('whole_child_elapsed_seconds',1200.),
            ('iterations',True),('single_gram_produced',False),('actual_cleanup_claimed',True)):
            ex=numeric_execution();ex[key]=value
            with self.assertRaises(ValueError):c.numeric_execution(ex)
        for key,value in (('threads_during',2),('threads_after',True),('rtld_noload',False)):
            ex=numeric_execution();ex['openblas'][key]=value
            with self.assertRaises(ValueError):c.numeric_execution(ex)
        ex=numeric_execution();ex['openblas']['environment']['OMP_NUM_THREADS']='2'
        with self.assertRaises(ValueError):c.numeric_execution(ex)
        pr,_,_=control_records();pr['fit']['ridge']=2
        with self.assertRaises(ValueError):c.validate_preregistration(pr)
    def test_runtime_blocks_before_any_actual_io(self):
        with mock.patch.object(fit,'PROTOTYPE_ONLY',True),self.assertRaisesRegex(ValueError,'blocked'):fit.actual_main(object())
        with self.assertRaisesRegex(RuntimeError,'SOURCE ONLY'):c.runtime_entry()
    def test_old_guards_and_numeric_body_ast_preserved(self):
        root=Path(__file__).resolve().parents[1]/'scripts'
        old=ast.parse((root/'fit_paired_linear.py').read_text());new=ast.parse(Path(fit.__file__).read_text())
        names=('_manifest_train','OpenBLASOneThread','rational_json','coefficient_bytes','sole_gram','_read_pinned')
        for name in names:
            a=next(x for x in old.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name==name)
            b=next(x for x in new.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name==name)
            self.assertEqual(ast.dump(a,include_attributes=False),ast.dump(b,include_attributes=False),name)
        a=next(x for x in old.body if isinstance(x,ast.FunctionDef) and x.name=='actual_main')
        b=next(x for x in new.body if isinstance(x,ast.FunctionDef) and x.name=='actual_main')
        oldsolve=next(x for x in ast.walk(a) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='solve_verified_design_bytes')
        newsolve=next(x for x in ast.walk(b) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='solve_verified_design_bytes')
        self.assertEqual(ast.dump(oldsolve,include_attributes=False),ast.dump(newsolve,include_attributes=False))

if __name__=='__main__':unittest.main()
