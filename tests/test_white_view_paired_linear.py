"""Public synthetic memory fixture; no actual inputs/file generation/fit."""
import ast
from dataclasses import replace
import importlib
import json
from pathlib import Path
import struct
import sys
import unittest

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import material_init as material
import paired_linear as paired
import white_view_paired_linear as w


def binding():
    return {'schema':'sekirei.white-view-paired-linear-feature-binding.v1','mode':w.MODE,'plan_sha256':w.PLAN_SHA256,
        'feature_schema':w.FEATURE_SCHEMA,'compile_feature':w.COMPILE_FEATURE,'native_magic':'SEKIRW03',
        'dimensions':{'input':2420,'l1':256,'l2':32},'stock_initializer_sha256':w.STOCK_INITIALIZER_SHA256,
        'feature_source_sha256':{name:'a'*64 for name in w.SOURCE_NAMES},'core_binary_sha256':'b'*64,'helper_source_sha256':'c'*64}


SFENS=('4k4/9/9/9/9/9/9/9/4K4 b - 1',
       '4k4/9/9/4p4/9/3P5/9/9/4K4 b PLns 1',
       '4k4/9/9/9/4+r4/9/2+B6/9/4K4 w 2Pn 1')


class WhiteViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # In-memory public deterministic construction; no private initializer
        # is read and no candidate/native file is written.
        cls.original=material.encode(material.build_weights(42))
        cls.bound=binding();cls.initial=w.transform_initializer(cls.original,binding=cls.bound)
        cls.zero=b'\0'*1016

    def test_binding_new_mode_and_external_hash_types(self):
        self.assertEqual(w.validate_binding(self.bound),self.bound)
        for edit in (lambda b:b.update(mode=paired.MODE),lambda b:b.update(native_magic='SEKIRW01'),
                     lambda b:b['dimensions'].update(l1=True),lambda b:b.update(plan_sha256='0'*64),
                     lambda b:b.update(core_binary_sha256=False),lambda b:b['feature_source_sha256'].pop(next(iter(w.SOURCE_NAMES)))):
            b=binding();edit(b)
            with self.assertRaises(ValueError):w.validate_binding(b)
    def test_coordinate_covariance_all4536(self):
        count=0
        for square in range(81):
            for kind in range(14):
                for color in range(2):
                    for perspective in range(2):
                        self.assertEqual(w.feature_index(square,kind,color,perspective),w.feature_index(80-square,kind,1-color,1-perspective))
                        expected=(square if perspective==0 else 80-square)*28+2*kind+int(color!=perspective)
                        self.assertEqual(w.feature_index(square,kind,color,perspective),expected);count+=1
        self.assertEqual(count,4536)
    def test_hand_index152_retains_fourbanks(self):
        count=0
        for kind,cap in enumerate(material.HAND_MAX):
            for n in range(1,cap+1):
                for color in range(2):
                    for perspective in range(2):
                        original=w.hand_feature_index(kind,color,perspective,n)
                        flipped=w.hand_feature_index(kind,1-color,1-perspective,n)
                        self.assertEqual((original-2268)//38 ^ 3,(flipped-2268)//38)
                        self.assertEqual((original-2268)%38,(flipped-2268)%38);count+=1
        self.assertEqual(count,152)
    def test_bool_float_feature_coordinates_rejected(self):
        for args in ((True,0,0,0),(0,False,0,0),(0,0,.0,0),(0,0,0,True),(81,0,0,0),(0,14,0,0)):
            with self.assertRaises(ValueError):w.feature_index(*args)
        for args in ((0,0,0,False),(0,0,0,19),(7,0,0,1)):
            with self.assertRaises(ValueError):w.hand_feature_index(*args)
    def test_old01_fixed_transform_input(self):
        self.assertEqual(self.original[:8],b'SEKIRW01');self.assertEqual(w.sha(self.original),w.STOCK_INITIALIZER_SHA256)
        with self.assertRaises(ValueError):w.transform_initializer(self.initial,binding=self.bound)
        changed=bytearray(self.original);changed[8]^=1
        with self.assertRaises(ValueError):w.transform_initializer(bytes(changed),binding=self.bound)
    def test_magic03_not_accepted_by_old_loader(self):
        self.assertEqual(self.initial[:8],b'SEKIRW03')
        with self.assertRaises(ValueError):material.decode(self.initial)
        native=w.serialize_coefficients(self.zero,self.original,binding=self.bound)
        with self.assertRaises(ValueError):paired.validate_native(native,self.zero)
    def test_transform_only_destination_aux_and_magic(self):
        allowed=set(range(8))
        for destination in (2,3):
            for threshold in range(38):
                start=w.ft_row(2268+destination*38+threshold)+4
                allowed.update(range(start,start+508))
        differences={i for i,(a,b) in enumerate(zip(self.original,self.initial)) if a!=b}
        self.assertTrue(differences);self.assertTrue(differences<=allowed)
        self.assertEqual(self.initial[8:w.ft_row(2268)],self.original[8:w.ft_row(2268)])
        self.assertEqual(self.initial[w.FT_END:],self.original[w.FT_END:])
        self.assertEqual(w.protected_bytes(self.initial),w.protected_bytes(self.original))
    def test_hand_full256rows_tied_donors_unchanged(self):
        self.assertTrue(w.validate_hand_ties(self.initial))
        for donor,destination in ((0,3),(1,2)):
            for threshold in range(38):
                a=w.ft_row(2268+donor*38+threshold);b=w.ft_row(2268+destination*38+threshold)
                self.assertEqual(self.initial[a:a+512],self.initial[b:b+512])
                self.assertEqual(self.initial[a:a+512],self.original[a:a+512])
    def test_untied_new_loader_rejected(self):
        changed=bytearray(self.initial);changed[w.ft_row(2268+3*38)+4]^=1
        with self.assertRaises(ValueError):w.validate_hand_ties(bytes(changed))
    def test_new03_independent_parser(self):
        decoded=w.decode_native03(self.initial)
        self.assertEqual(len(decoded['ft']),2420*256)
        self.assertEqual(decoded['ft_bias'].tolist(),[64]*256)
        self.assertEqual(decoded['out'][:4].tolist(),[8192.,8192.,-8192.,-8192.])
    def test_new_serializer_protectedmaterial_and_signed_bits(self):
        coef=struct.pack('<f',-.75)+self.zero[4:]
        native=w.serialize_coefficients(coef,self.original,binding=self.bound)
        validation=w.validate_native(native,coef,self.original,binding=self.bound)
        self.assertEqual(w.protected_bytes(native),w.protected_bytes(self.original))
        self.assertTrue(validation['canonical_new_abi_rebuild_equal'])
        self.assertFalse(validation['actual_core_verified']);self.assertFalse(validation['actual_fitter_verified'])
        raw=struct.unpack('<I',coef[:4])[0]
        for row,col,expected in ((2,4,raw),(2,5,raw^0x80000000),(258,4,raw^0x80000000),(258,5,raw)):
            self.assertEqual(struct.unpack_from('<I',native,paired.l2_cell(row,col))[0],expected)
    def test_oldguard_does_not_change(self):
        old=paired.serialize_coefficients(self.zero,initializer=self.original)
        self.assertTrue(paired.validate_native(old,self.zero)['canonical_full_native_rebuild_equal'])
        self.assertEqual(old[:8],b'SEKIRW01')
    def test_native_modifications_rejected(self):
        native=w.serialize_coefficients(self.zero,self.original,binding=self.bound)
        for offset in (8,w.FT_END,w.L2_OFFSET,w.L2_BIAS_OFFSET,w.OUT_OFFSET,w.OUT_BIAS_OFFSET):
            changed=bytearray(native);changed[offset]^=1
            with self.assertRaises(ValueError):w.validate_native(bytes(changed),self.zero,self.original,binding=self.bound)
    def test_coefficient_nonfinite_negzero_and_cap_rejected(self):
        for v in (float('nan'),float('inf'),-.0,39.50001):
            with self.assertRaises(ValueError):w.serialize_coefficients(struct.pack('<f',v)+self.zero[4:],self.original,binding=self.bound)
    def test_stock_accumulation_physical_order_preserved(self):
        p=material.parse_sfen(SFENS[1]);features=w.active_features(p,1)
        board=[w.feature_index(s,k,c,1) for s,k,c in sorted(p['pieces'])]
        self.assertEqual(features[:len(board)],tuple(board))
        self.assertNotEqual(board,sorted(board))
    def test_position_standard_inventory_guard(self):
        p=material.parse_sfen(SFENS[0]);p['hands'][0][0]=True
        with self.assertRaises(ValueError):w.active_features(p,0)
        p=material.parse_sfen(SFENS[0]);p['pieces'].append(p['pieces'][0])
        with self.assertRaises(ValueError):w.active_features(p,0)
    def test_design_covariance_and_stm_odd_source(self):
        for sfen in SFENS:
            p=material.parse_sfen(sfen);flipped=w.rotated_color_swap(p)
            Z,M,ev=w.design_row(p,self.initial,self.original,binding=self.bound)
            Zf,Mf,_=w.design_row(flipped,self.initial,self.original,binding=self.bound)
            self.assertEqual(Z,Zf);self.assertEqual(M,Mf)
            stmflip={**p,'stm':1-p['stm']}
            Zs,Ms,_=w.design_row(stmflip,self.initial,self.original,binding=self.bound)
            self.assertEqual(tuple(-z for z in Z),Zs);self.assertEqual(-M,Ms)
            self.assertTrue(all(24<=a<=104 for a in ev['raw_us']+ev['raw_them']))
            self.assertTrue(all(type(z) is int and abs(z)<=40 for z in Z))
    def test_design_rejects_arbitrary_prefix(self):
        changed=bytearray(self.initial);changed[8+4]^=2
        with self.assertRaises(ValueError):w.design_row(material.parse_sfen(SFENS[0]),bytes(changed),self.original,binding=self.bound)

    def make_design(self,labels=None,sfens=SFENS):
        if labels is None:labels=tuple({'sfen':sfen,'score_cp':100+i,'teacher_identity':w.ORIGINAL_TEACHER,'label_depth':0,'source_extra':'public synthetic'} for i,sfen in enumerate(sfens))
        return w.build_design(tuple(sfens),labels,self.initial,self.original,binding=self.bound,
            input_digests={'manifest':'a'*64,'train_positions':'b'*64,'train_labels':'c'*64})
    def test_design_order_cache_join_target(self):
        labels=tuple({'sfen':sfen,'score_cp':100+i,'teacher_identity':w.ORIGINAL_TEACHER,'label_depth':0} for i,sfen in enumerate(SFENS))
        design=self.make_design(tuple(reversed(labels)))
        self.assertEqual(len(design.z_bytes),3*254);self.assertEqual(len(design.d_bytes),12)
        expected=[100+i-material.material(material.parse_sfen(sfen)) for i,sfen in enumerate(SFENS)]
        self.assertEqual([v[0] for v in struct.iter_unpack('<i',design.d_bytes)],expected)
        provenance=json.loads(design.provenance_json)
        self.assertTrue(provenance['position_order_preserved']);self.assertFalse(provenance['actual_original_manifest_and_replay_verified'])
    def test_teacher_depth_targettype_reject(self):
        for fields in ({'score_cp':True},{'score_cp':1.0},{'score_cp':30000},{'teacher_identity':'external:other'},
                       {'label_depth':False},{'label_depth':1},{'score_cp':float('nan')}):
            labels=[{'sfen':sfen,'score_cp':100+i,'teacher_identity':w.ORIGINAL_TEACHER,'label_depth':0} for i,sfen in enumerate(SFENS)]
            labels[0].update(fields)
            with self.assertRaises(ValueError):self.make_design(tuple(labels))
    def test_duplicate_missing_extra_semantic_reject(self):
        labels=tuple({'sfen':sfen,'score_cp':0,'teacher_identity':w.ORIGINAL_TEACHER,'label_depth':0} for sfen in SFENS)
        for changed in (labels[:2],(labels[0],labels[0],labels[2]),(labels[0],labels[1],{**labels[2],'sfen':'unmatched'})):
            with self.assertRaises(ValueError):self.make_design(changed)
        duplicate=(SFENS[0],SFENS[0].replace(' 1',' 2'))
        with self.assertRaises(ValueError):self.make_design(sfens=duplicate)
    def test_issued_numeric_design_no_gram_fit(self):
        design=self.make_design();producer=importlib.import_module('verified_design_gram')
        previous_origins=dict(producer._gram_origins)
        issued=w.to_verified_numeric_design(design,producer,binding=self.bound,expected_design_sha256=design.design_sha256,expected_producer_sha256=w.NUMERIC_PRODUCER_SHA256)
        self.assertEqual((issued.samples,issued.dimension),(3,254));self.assertEqual(issued.z_bytes,design.z_bytes)
        self.assertEqual(producer._gram_origins,previous_origins)
        with self.assertRaises(ValueError):w.to_verified_numeric_design(replace(design),producer,binding=self.bound,expected_design_sha256=design.design_sha256,expected_producer_sha256=w.NUMERIC_PRODUCER_SHA256)
        with self.assertRaises(ValueError):w.to_verified_numeric_design(design,producer,binding=self.bound,expected_design_sha256='f'*64,expected_producer_sha256=w.NUMERIC_PRODUCER_SHA256)
    def test_mutated_sealed_design_rejected(self):
        design=self.make_design();producer=importlib.import_module('verified_design_gram')
        object.__setattr__(design,'z_bytes',b'\0'*len(design.z_bytes))
        with self.assertRaises(ValueError):w.to_verified_numeric_design(design,producer,binding=self.bound,expected_design_sha256=design.design_sha256,expected_producer_sha256=w.NUMERIC_PRODUCER_SHA256)
    def test_runtime_closed_and_ast(self):
        with self.assertRaisesRegex(RuntimeError,'SOURCE ONLY'):w.runtime_entry()
        for path in (Path(w.__file__),Path(__file__)):ast.parse(path.read_text())


if __name__=='__main__':unittest.main()
