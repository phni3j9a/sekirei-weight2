"""Tiny pure analytic synthetic fixtures only; dimension <=2, no real artifacts."""
import copy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import unittest
import ast
from unittest.mock import patch
sys.dont_write_bytecode=True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(1,str(REPO/'scripts'))
sys.path.insert(2,str(REPO/'tests'))
import white_view_independent_numeric_audit as audit


def rational_oracle(G, h, N, coefficients, radius):
    """Tiny direct Fraction oracle, independent of integer-denominator worker."""
    u = tuple(Fraction(x) for x in coefficients)
    H = tuple(tuple(Fraction(G[i][j]+(256 if i==j else 0),256) for j in range(len(u)))
              for i in range(len(u)))
    b = tuple(Fraction(x,16) for x in h)
    g = tuple(sum(H[i][j]*u[j] for j in range(len(u)))-b[i] for i in range(len(u)))
    norm = sum(abs(x) for x in u)
    gap = sum(x*y for x,y in zip(g,u)) + radius*max(abs(x) for x in g)
    delta = sum(u[i]*H[i][j]*u[j]/2 for i in range(len(u)) for j in range(len(u))) - sum(x*y for x,y in zip(b,u))
    den = max(x.as_integer_ratio()[1] for x in coefficients)*256
    return {'norm':audit.record(norm),'radius':audit.record(radius),'feasible':norm<=radius,
        'fw_gap':audit.record(gap),'fw_gap_per_sample':audit.record(gap/N),
        'solver_threshold_met':norm<=radius and 0<=gap/N<=audit.GAP_PER_SAMPLE,
        'objective_delta_vs_zero':audit.record(delta),
        'gradient_numerators':[int(x*den) for x in g], 'gradient_denominator':den,
        'coefficient_values':[audit.record(x) for x in coefficients]}


def fixture(tiny=False):
    N, p = (1, 1) if tiny else (2, 2)
    G = ((0,),) if tiny else ((256,0),(0,256))
    h = (0,) if tiny else (32,-64)
    f64 = (-1e-50,) if tiny else (1.0,-2.0)
    f32 = (0.0,) if tiny else f64
    gram = b''.join(struct.pack('<q',v) for row in G for v in row)
    rhs = b''.join(struct.pack('<q',v) for v in h)
    c64 = b''.join(struct.pack('<d',v) for v in f64)
    c32 = b''.join(struct.pack('<f',v) for v in f32)
    z = b'\0' if tiny else struct.pack('<4b',16,0,0,16)
    d = b'\0'*4 if tiny else struct.pack('<2i',2,-4)
    dh = hashlib.sha256(audit.GRAM_SCHEMA.encode()+b'\x00design\x00'+struct.pack('<QQ',N,p)+z+d).hexdigest()
    metadata = [['numpy_version','1.26.4'],['dtype','float64'],['design_order','C'],
        ['transpose','Z.T@Z;Z.T@d'],['first_numpy_preimported',True],
        ['environment_set_before_first_numpy_import',False],['existing_import_threadpool_evidence_not_claimed',True]]
    gh = audit.producer_gram_digest(gram,rhs,N,p,dh,tuple(tuple(row) for row in metadata),audit.PRODUCER_SHA)
    L = Fraction(1 if tiny else 2)
    result = {'status':'synthetic-certified','iterations':1,'restart_count':0,
        'coefficient_f64':[audit.record(x) for x in f64], 'coefficient_f32':[audit.record(x) for x in f32],
        'certificate_f64':rational_oracle(G,h,N,f64,audit.RHO),
        'certificate_f32_solver_radius':rational_oracle(G,h,N,f32,audit.RHO),
        'certificate_f32_saved_radius':rational_oracle(G,h,N,f32,audit.SAVED_RADIUS),
        'saved_f32_gap_is_solver_stopping_criterion':False,'lipschitz_exact':audit.record(L),
        'lipschitz_numeric_upper':audit.record(math.nextafter(float(L),math.inf)),
        'fallback_used':False,'one_if_needed_f64_correction_used':False,
        'one_if_needed_f64_correction_factor':audit.record(1-2.0**-40), 'lambda':1,
        'loss_is_half_unnormalized_sum':True,
        'verified_design_gram':{'schema':audit.GRAM_SCHEMA,'producer':audit.GRAM_PRODUCER,
            'design_sha256':dh,'gram_sha256':gh,'producer_source_sha256':audit.PRODUCER_SHA,
            'solver_source_sha256':audit.SOLVER_SHA,'backend_metadata':metadata,'sample_count':N,'dimension':p,
            'algebraic_psd_proven_from_issued_design':True,'runtime_threadpool_proof_claimed':False,
            'actual_dataset_provenance_claimed':False,'source_prototype_only':True}}
    design={'schema':'sekirei.white-view-paired-linear-train-design.v1','count':N,'dimension':p,
            'z_sha256':audit.sha(z),'d_sha256':audit.sha(d)}
    return {'gram':gram,'rhs':rhs,'c64':c64,'c32':c32,'result':result,'design':design,
            'N':N,'p':p,'zh':audit.sha(z),'dh':audit.sha(d),'design_sha':dh}


def run(data):
    return audit.audit_numeric_bytes(data['gram'],data['rhs'],data['c64'],data['c32'],
        audit.canonical_json(data['result']),audit.canonical_json(data['design']),N=data['N'],dimension=data['p'],
        design_z_sha256=data['zh'],targets_d_sha256=data['dh'],design_sha256=data['design_sha'])


class NumericAuditTests(unittest.TestCase):
    def test_tiny_analytic_zero_gradient_half_objective(self):
        result=run(fixture())
        self.assertTrue(result['all_three_certificates_exactly_recomputed'])
        cert=result['recomputed_certificates']['certificate_f64']
        self.assertEqual(cert['gradient_numerators'],[0,0])
        self.assertEqual(cert['fw_gap'],{'numerator':0,'denominator':1})
        self.assertEqual(cert['objective_delta_vs_zero'],{'numerator':-5,'denominator':1})
        self.assertFalse(result['gram_product_recomputed'])
        self.assertFalse(result['gram_psd_independently_reproved'])

    def test_common_dyadic_nonzero_gradients_match_fraction_oracle(self):
        G=((256,32),(32,512));h=(3,-7);N=3;v=(0.125,-0.0625)
        _,K,_=audit.parse_matrix(b''.join(struct.pack('<q',x) for row in G for x in row),
            b''.join(struct.pack('<q',x) for x in h),N,2)
        actual=audit.independent_certificate(K,h,N,v,audit.RHO)
        self.assertEqual(actual,rational_oracle(G,h,N,v,audit.RHO))

    def test_negative_underflow_f64_saves_canonical_positive_f32_zero(self):
        result=run(fixture(True))
        self.assertTrue(result['saved_f32_nearest_canonical_cast_verified'])
        data=fixture(True);data['c32']=struct.pack('<f',-0.0)
        with self.assertRaises(audit.InvalidProof):run(data)

    def test_corrupt_each_exact_certificate_field_rejected(self):
        original=fixture()
        for certificate in ('certificate_f64','certificate_f32_solver_radius','certificate_f32_saved_radius'):
            for field in ('norm','fw_gap','fw_gap_per_sample','objective_delta_vs_zero','radius',
                          'gradient_denominator','gradient_numerators','coefficient_values','feasible','solver_threshold_met'):
                data=copy.deepcopy(original)
                value=data['result'][certificate][field]
                if type(value) is dict:value['numerator']+=1
                elif type(value) is list:
                    if type(value[0]) is dict:value[0]['numerator']+=1
                    else:value[0]+=1
                elif type(value) is bool:data['result'][certificate][field]=int(value)
                else:data['result'][certificate][field]+=1
                with self.subTest(certificate=certificate,field=field),self.assertRaises(audit.InvalidProof):run(data)

    def test_coefficients_exact_artifacts_and_nearest_rounding_rejected(self):
        for key in ('c64','c32'):
            data=fixture();data[key]=bytearray(data[key]);data[key][0]^=1;data[key]=bytes(data[key])
            with self.subTest(key=key),self.assertRaises(audit.InvalidProof):run(data)
        data=fixture();data['result']['coefficient_f64'][0]['numerator']=2
        with self.assertRaises(audit.InvalidProof):run(data)

    def test_gram_symmetry_bounds_and_rhs_mutation_rejected(self):
        data=fixture(); data['gram']=struct.pack('<4q',256,1,0,256)
        with self.assertRaises(audit.InvalidProof):run(data)
        data=fixture(); data['gram']=struct.pack('<4q',3201,0,0,256)
        with self.assertRaises(audit.InvalidProof):run(data)
        data=fixture(); data['rhs']=struct.pack('<2q',33,-64)
        with self.assertRaises(audit.InvalidProof):run(data)

    def test_producer_full_content_source_order_design_bindings_rejected(self):
        for field in ('producer_source_sha256','solver_source_sha256','design_sha256','gram_sha256'):
            data=fixture();data['result']['verified_design_gram'][field]='f'*64
            with self.subTest(field=field),self.assertRaises(audit.InvalidProof):run(data)
        data=fixture();data['result']['verified_design_gram']['backend_metadata'][3][1]='Z@Z.T'
        with self.assertRaises(audit.InvalidProof):run(data)
        data=fixture();data['design']['z_sha256']='f'*64
        with self.assertRaises(audit.InvalidProof):run(data)

    def test_fraction_noncanonical_bool_recipe_and_gershgorin_rejected(self):
        for change in ('fraction','lambda_bool','fallback','savedstop','lipschitz'):
            data=fixture()
            if change=='fraction':data['result']['lipschitz_exact']={'numerator':4,'denominator':2}
            if change=='lambda_bool':data['result']['lambda']=True
            if change=='fallback':data['result']['fallback_used']=True
            if change=='savedstop':data['result']['saved_f32_gap_is_solver_stopping_criterion']=True
            if change=='lipschitz':data['result']['lipschitz_numeric_upper']=audit.record(2.0)
            with self.subTest(change=change),self.assertRaises(audit.InvalidProof):run(data)

    def test_strict_count_dimension_and_nonfinite_ieee_rejected(self):
        for field in ('N','p'):
            data=fixture();data[field]=True
            with self.subTest(field=field),self.assertRaises(audit.InvalidProof):run(data)
        for key,dtype in (('c64','d'),('c32','f')):
            data=fixture();data[key]=struct.pack('<2'+dtype,float('nan'),-2.0)
            with self.subTest(key=key),self.assertRaises(audit.InvalidProof):run(data)
        with self.assertRaises(audit.InvalidProof):audit.strict_json(b'{"x":1,"x":2}')
        with self.assertRaises(audit.InvalidProof):audit.strict_json(b'{"x":NaN}')
        with self.assertRaises(audit.InvalidProof):audit.strict_json(b'{"x":1e1000}')

    def test_main_entry_remains_blocked_before_actual_io(self):
        result=subprocess.run([sys.executable,'-B',str(Path(audit.__file__))],capture_output=True,text=True,timeout=5)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('actual numerical audit entry blocked',result.stderr)

    def test_actual_function_blocks_before_reader_or_file_hash(self):
        with patch.object(audit,'file_identity',side_effect=AssertionError('actual I/O forbidden')):
            with self.assertRaisesRegex(audit.InvalidProof,'actual numerical audit entry blocked'):
                audit.actual_main(object())

    def test_original_exact_arithmetic_ast_preserved(self):
        old_path=Path(__file__).resolve().parent.parent/'white-view-source-fixtures-v1'/'paired-numeric-original.py.txt'
        old=ast.parse(old_path.read_text());new=ast.parse(Path(audit.__file__).read_text())
        funcs=lambda t:{n.name:n for n in t.body if isinstance(n,ast.FunctionDef)}
        old,new=funcs(old),funcs(new)
        for name in ('record','fraction_record','parse_matrix','parse_coefficients','independent_certificate',
                     '_backend_metadata','producer_gram_digest','design_digest_files'):
            self.assertEqual(ast.dump(old[name],include_attributes=False),ast.dump(new[name],include_attributes=False),name)
        class SchemaNormalize(ast.NodeTransformer):
            def visit_Constant(self,n):
                if type(n.value) is str:n.value=n.value.replace('sekirei.white-view-','sekirei.')
                return n
        self.assertEqual(ast.dump(old['audit_numeric_bytes'],include_attributes=False),
            ast.dump(SchemaNormalize().visit(new['audit_numeric_bytes']),include_attributes=False))

    def test_full_external_cli_fields_and_new_reference_path(self):
        names={a.dest for a in audit.parser()._actions}
        self.assertTrue({'expected_worker_sha256','expected_run_sha256','expected_contract_sha256',
            'expected_driver_sha256','expected_preregistration_sha256','expected_activation_sha256',
            'expected_source_preflight_sha256','expected_build_manifest_sha256','expected_build_identity_sha256',
            'expected_build_validator_sha256','expected_reference03_sha256','expected_metadata_sha256'}<=names)
        self.assertTrue({'expected_'+key+'_sha256' for key in ('weights','gram_z','rhs_z','coefficients_f64',
            'coefficients_f32','solver_certificate','design_receipt','design_z','targets_d')}<=names)
        self.assertEqual(audit.REFERENCE03,audit.C.parent/'training-17-v1/white-view-material-init-seed42-v1/reference03.bin')


class NewABITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Caller-authorized public initializer fixture, in memory only.
        sys.path.insert(0,str(REPO/'scripts'))
        sys.path.insert(1,str(REPO/'tests'))
        import white_view_fit_contract as c
        import material_init as material
        import white_view_paired_linear as white
        import test_white_view_fit_driver as fixture_driver
        cls.c=c;cls.white=white;cls.fd=fixture_driver
        cls.stock=material.encode(material.build_weights(42));cls.feature=fixture_driver.binding()
        cls.ref03=white.transform_initializer(cls.stock,binding=cls.feature)

    def artifacts(self):
        c=self.c;zero=b'\0'*1016;native=self.white.serialize_coefficients(zero,self.stock,binding=self.feature)
        outputs={k:self.fd.reference(c.Q/name) for k,name in c.OUTPUT_NAMES.items()}
        outputs['weights']=self.fd.reference(c.W,native)
        outputs['coefficients_f32']=self.fd.reference(c.Q/'coefficients.f32.bin',zero)
        context=self.fd.context();run=c.construct_fit_run(context,outputs,outputs['solver_certificate'],c.LIFECYCLE)
        raw=self.fd.enc(run)
        bind={'plan_sha256':c.PLAN_SHA256,'preregistration_sha256':context['preregistration_sha256'],
            'activation_sha256':context['activation_sha256'],'source_preflight_sha256':context['source_preflight_sha256'],
            'fit_receipt':self.fd.reference(c.FIT_RECEIPT,raw),'solver_certificate':outputs['solver_certificate'],
            'source_helpers_sha256':context['source_helpers_sha256'],'feature_binding':context['feature_binding'],
            'new_build_binding':context['new_build_binding']}
        meta=self.fd.enc(c.sidecar_for_fit(native,zero,self.stock,raw,bind))
        return native,zero,raw,meta,bind

    def test_new03_forward_binding_full_rebuild_scope(self):
        native,zero,raw,meta,bind=self.artifacts()
        result,_=audit.forward_artifact_binding(native,zero,self.stock,self.ref03,raw,meta,bind,self.c)
        self.assertTrue(result['native03_full_reconstruction_verified'])
        self.assertTrue(result['reference03_full_transform_verified'])
        self.assertFalse(result['actual_native_core_forward_verified'])
        self.assertFalse(result['actual_incremental_verified'])
        self.assertFalse(result['actual_native_core_covariance_verified'])

    def test_old01_reference_or_native_and_sidecar_tamper_rejected(self):
        native,zero,raw,meta,bind=self.artifacts()
        for changed_native,changed_ref,changed_meta in ((self.stock,self.ref03,meta),
                (native,self.stock,meta),(native,self.ref03,meta.replace(b'SEKIRW03',b'SEKIRW01'))):
            with self.assertRaises(ValueError):
                audit.forward_artifact_binding(changed_native,zero,self.stock,changed_ref,raw,changed_meta,bind,self.c)

    def provenance(self):
        root=audit.C/'source-input-recovery-v1/dataset'
        files={s+'.'+k+'.jsonl':{'bytes':3,'sha256':str(i+1)*64}
            for i,(s,k) in enumerate((('train','positions'),('train','labels'),('holdout','positions'),('holdout','labels')))}
        manifest={'files':files,**{k:{'synthetic':True} for k in ('source_corpus_manifest_sha256',
            'seed','split','sampling','games_per_pack','dependencies','independent_exclusions','derivation','games')}}
        before={str(root/name):dict(info) for name,info in files.items()}
        before[str(root/'manifest.json')]={'bytes':10,'sha256':audit.ORIGINAL_MANIFEST_SHA}
        design={'schema':'sekirei.white-view-paired-linear-train-design.v1','count':2,'dimension':2,
            'teacher_identity':audit.TEACHER,'original_manifest_sha256':audit.ORIGINAL_MANIFEST_SHA,
            'material_initializer_sha256':audit.INITIALIZER_SHA,'transformed_initializer_sha256':audit.sha(self.ref03),
            'native_magic':'SEKIRW03','feature_schema':'flat_white_view_aux_tied_v1','feature_binding':self.feature,
            'feature_index_implementation':'fixed white_view_paired_linear.active_features',
            'single_numpy_integer_adapter':True,'actual_build_verified_by_builder':False,
            'position_order_preserved':True,'labels_joined_by_exact_sfen':True,
            'labels_extra_metadata_bound_in_original_sha':True,'z_dtype':'signed-int8-C-row-major',
            'd_dtype':'signed-int32-little-endian','z_definition':'(raw_us-raw_them)/2; channels 2:256',
            'x_definition':'Z/16','d_definition':'original absolute STM teacher T minus fixed material M',
            'holdout_rows_read':False,'development_used':False,'final_used':False,'whole_dataset_replay_proof_claimed':False,
            'original_files':files,'source_provenance':{k:v for k,v in manifest.items() if k!='files'},
            'train_positions_sha256':'1'*64,'train_labels_sha256':'2'*64,'ft_raw_min':24,'ft_raw_max':104,
            'active_feature_min':2,'active_feature_max':40,'maximum_absolute_d':55779,
            'ordered_sfens_sha256':'a'*64,'ft_prefix_sha256':'b'*64}
        return design,manifest,before

    def test_newdesign_original_fourfiles_provenance_without_extraction(self):
        design,manifest,before=self.provenance()
        result=audit.validate_design_provenance(design,manifest,before,self.feature,self.ref03,N=2,dimension=2)
        self.assertTrue(result['original_four_file_identities_verified'])
        self.assertFalse(result['extraction_replayed'])
        for key,val in (('train_labels_sha256','f'*64),('single_numpy_integer_adapter',1),
            ('feature_schema','flat'),('maximum_absolute_d',True),('source_provenance',{})):
            changed=copy.deepcopy(design);changed[key]=val
            with self.assertRaises(audit.InvalidProof):
                audit.validate_design_provenance(changed,manifest,before,self.feature,self.ref03,N=2,dimension=2)

    def test_identity_map_extra_path_bool_and_old_schema_rejected(self):
        for value in ({'/synthetic/file':{'path':'/synthetic/other','bytes':1,'sha256':'a'*64}},
                      {'/synthetic/file':{'bytes':True,'sha256':'a'*64}}):
            with self.assertRaises(audit.InvalidProof):audit.identity_map(value)
        data=fixture();data['design']['schema']='sekirei.paired-linear-train-design.v1'
        with self.assertRaises(audit.InvalidProof):run(data)


    def producer_json_bytes(self, value):
        # Execute only these three pure functions from the unchanged public
        # producer source. The driver module/actual entry is never imported.
        source = REPO/'scripts/fit_white_view_paired_linear.py'
        tree = ast.parse(source.read_text())
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name in ('require', 'rational_json', 'json_bytes')]
        self.assertEqual({node.name for node in selected}, {'require', 'rational_json', 'json_bytes'})
        namespace = {'json': json, 'math': math, 'Fraction': Fraction}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)
        return namespace['json_bytes'](value)

    def float_provenance(self):
        design, manifest, before = self.provenance()
        manifest['seed'] = 42
        manifest['source_corpus_manifest_sha256'] = 'c'*64
        manifest['derivation'] = {'wall_seconds': 1.1, 'complete': True, 'count': 17,
                                  'nested': [0.5, -0.0, {'fractional_seconds': 5e-324}]}
        manifest['games'] = {'train': ['synthetic-game-a'], 'holdout': ['synthetic-game-b']}
        design['source_provenance'] = {key: manifest[key] for key in design['source_provenance']}
        return audit.strict_json(self.producer_json_bytes(design)), manifest, before

    def test_producer_json_bytes_recursive_float_metadata_accepted(self):
        design, manifest, before = self.float_provenance()
        self.assertEqual(design['source_provenance']['derivation']['wall_seconds'],
                         {'numerator': 2476979795053773, 'denominator': 2251799813685248})
        result = audit.validate_design_provenance(design, manifest, before, self.feature, self.ref03,
                                                 N=2, dimension=2)
        self.assertTrue(result['original_source_metadata_equal'])
        self.assertFalse(result['extraction_replayed'])
        self.assertIs(type(design['source_provenance']['derivation']['complete']), bool)
        self.assertIs(type(design['source_provenance']['derivation']['count']), int)

    def test_independent_recursive_recipe_matches_pure_producer(self):
        value = {'finite': [1.1, 0.5, -0.0, 5e-324], 'tuple': (1, True, None, 'text'),
                 'fraction': Fraction(-1, 3), 'int': 2, 'bool': False}
        actual = audit.canonical_producer_metadata(value)
        expected = audit.strict_json(self.producer_json_bytes(value))
        self.assertTrue(audit.exact(actual, expected))
        self.assertTrue(audit.exact(actual['finite'][2], {'numerator': 0, 'denominator': 1}))
        for bad in (float('nan'), float('inf'), -float('inf'), {1: 'bad-key'}, {True: 1}, {1, 2}, b'bytes'):
            with self.subTest(bad=repr(bad)), self.assertRaises(audit.InvalidProof):
                audit.canonical_producer_metadata(bad)

    def test_rational_wall_seconds_full_shape_and_exact_value_required(self):
        original, manifest, before = self.float_provenance()
        mutations = []
        for record in ({'numerator': 1, 'denominator': 1},
                       {'numerator': True, 'denominator': 2},
                       {'numerator': 4953959590107546, 'denominator': 4503599627370496},
                       {'numerator': 2476979795053773, 'denominator': 2251799813685248, 'extra': 0},
                       1.1):
            changed = copy.deepcopy(original)
            changed['source_provenance']['derivation']['wall_seconds'] = record
            mutations.append(changed)
        changed = copy.deepcopy(original)
        changed['source_provenance']['derivation']['wall_seconds']['numerator'] += 1
        mutations.append(changed)
        for changed in mutations:
            with self.assertRaises(audit.InvalidProof):
                audit.validate_design_provenance(changed, manifest, before, self.feature, self.ref03, N=2, dimension=2)

    def test_nonfloat_hash_game_bool_int_and_provenance_keys_still_exact(self):
        original, manifest, before = self.float_provenance()
        mutations = []
        for key in original['source_provenance']:
            changed = copy.deepcopy(original)
            changed['source_provenance'][key] = 'different'
            mutations.append(changed)
        for field, value in (('complete', 1), ('count', True)):
            changed = copy.deepcopy(original)
            changed['source_provenance']['derivation'][field] = value
            mutations.append(changed)
        changed = copy.deepcopy(original)
        changed['source_provenance']['games']['train'][0] = 'changed-game'
        mutations.append(changed)
        changed = copy.deepcopy(original)
        changed['source_provenance']['source_corpus_manifest_sha256'] = 'd'*64
        mutations.append(changed)
        changed = copy.deepcopy(original)
        changed['source_provenance']['extra'] = 'forbidden'
        mutations.append(changed)
        changed = copy.deepcopy(original)
        del changed['source_provenance']['dependencies']
        mutations.append(changed)
        for changed in mutations:
            with self.assertRaises(audit.InvalidProof):
                audit.validate_design_provenance(changed, manifest, before, self.feature, self.ref03, N=2, dimension=2)
        for bad in (float('nan'), float('inf'), -float('inf')):
            changed_manifest = copy.deepcopy(manifest)
            changed_manifest['derivation']['wall_seconds'] = bad
            with self.assertRaises(audit.InvalidProof):
                audit.validate_design_provenance(original, changed_manifest, before, self.feature, self.ref03, N=2, dimension=2)

    def test_v1_all_other_functions_and_classes_ast_unchanged(self):
        old_path = Path(__file__).resolve().parent.parent/'white-view-source-fixtures-v1'/'white-numeric-v1.py.txt'
        old = ast.parse(old_path.read_text()); new = ast.parse(Path(audit.__file__).read_text())
        definitions = lambda tree: {node.name: node for node in tree.body
                                   if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        previous, current = definitions(old), definitions(new)
        self.assertEqual(set(current) - set(previous), {'canonical_producer_metadata'})
        for name in previous:
            if name != 'validate_design_provenance':
                self.assertEqual(ast.dump(previous[name], include_attributes=False),
                                 ast.dump(current[name], include_attributes=False), name)
        class RemoveExpectedSerialization(ast.NodeTransformer):
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id == 'canonical_producer_metadata':
                    self_outer.assertEqual(len(node.args), 1)
                    self_outer.assertFalse(node.keywords)
                    return node.args[0]
                return self.generic_visit(node)
        self_outer = self
        normalized = RemoveExpectedSerialization().visit(copy.deepcopy(current['validate_design_provenance']))
        self.assertEqual(ast.dump(previous['validate_design_provenance'], include_attributes=False),
                         ast.dump(normalized, include_attributes=False))
        assignments = lambda tree: {ast.dump(node.targets[0], include_attributes=False): node for node in tree.body
                                    if isinstance(node, ast.Assign) and len(node.targets) == 1}
        previous, current = assignments(old), assignments(new)
        self.assertEqual(set(previous), set(current))
        # The immutable historical text is the actually enabled v1 producer.
        # Preserve its bytes; explicitly verify only this known flag delta.
        flag = "Name(id='PROTOTYPE_ONLY', ctx=Store())"
        self.assertIs(ast.literal_eval(previous[flag].value), False)
        self.assertIs(ast.literal_eval(current[flag].value), True)
        for name in previous:
            if name not in ("Name(id='SELF', ctx=Store())", "Name(id='OUT', ctx=Store())", flag):
                self.assertEqual(ast.dump(previous[name], include_attributes=False),
                                 ast.dump(current[name], include_attributes=False), name)

    def test_v2_self_output_paths_and_prototype_gate(self):
        self.assertIs(audit.PROTOTYPE_ONLY, True)
        self.assertEqual(audit.SELF, audit.C/'white-view-paired-linear-independent-numeric-audit-worker-v2.py')
        self.assertEqual(audit.OUT, audit.C/'white-view-paired-linear-independent-numeric-audit-v2.json')


if __name__=='__main__':unittest.main()
