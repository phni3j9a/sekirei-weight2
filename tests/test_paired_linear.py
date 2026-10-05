"""Synthetic memory-only byte and typed fit control fixtures."""
import copy
import hashlib
import json
import struct
import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from unittest import mock

import material_init
import bounded_material
import paired_linear as p


def fixture():
    coefficients=b'\0'*p.COEFFICIENT_BYTES
    data=p.serialize_coefficients(coefficients)
    ref=lambda path,data:{'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    binding={'plan_sha256':p.CONDITIONAL_PLAN_SHA256,'preregistration_sha256':'a'*64,'activation_sha256':'b'*64,
       'fit_receipt':{},'solver_certificate':ref(p.Q/'solver-certificate.json',b'certificate'),
       'source_helpers_sha256':{'paired_linear.py':'c'*64,'paired_linear_activation.py':'d'*64}}
    run={'schema':'sekirei.paired-linear-fit-run.v1','status':'complete','candidate':p.MODE,'mode':p.MODE,
       'seed':42,'nnue_output':'absolute','optimizer':'projected-FISTA','adam_used':False,'epochs':0,
       'resume_used':False,'trainer_used':False,'inputs_unchanged':True,'source_unchanged':True,'cleanup_verified':True,
       'final_used':False,'adoption_claimed':False,'plan_sha256':p.CONDITIONAL_PLAN_SHA256,
       'preregistration_sha256':binding['preregistration_sha256'],'activation_sha256':binding['activation_sha256'],
       'source_helpers_sha256':binding['source_helpers_sha256'],'solver_certificate':binding['solver_certificate'],
       'outputs':{'weights':ref(p.W,data),'coefficients_f32':ref(p.COEFFICIENTS,coefficients)}}
    raw=json.dumps(run,sort_keys=True).encode();binding['fit_receipt']=ref(p.FIT_RECEIPT,raw)
    return coefficients,data,run,raw,binding


class PairedLinearTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.coefficients,cls.data,cls.fit_run,cls.raw,cls.binding=fixture()

    def test_full_rebuild_and_source_claim_limits(self):
        result=p.validate_native(self.data,self.coefficients,p.reference())
        self.assertTrue(result['canonical_full_native_rebuild_equal'])
        for key in ('actual_core_bound_verified','actual_fitter_verified','legal_dataset_inventory_verified',
                    'native_bit_exact_stm_antisymmetry_verified','activation_verified','adoption_verified','quantization_reexport_claimed'):
            self.assertIs(result[key],False)
        self.assertEqual(p.validate_weights(self.data)['weight_sha256'],hashlib.sha256(self.data).hexdigest())
        self.assertEqual([p.bits(self.data,p.l2_cell(r,c)) for r,c in ((2,4),(2,5),(258,4),(258,5))],[0,p.SIGN_BIT,p.SIGN_BIT,0])

    def test_nonzero_all254_and_wrong_external_coefficients(self):
        coeff=struct.pack('<254f',*([.125,-.125]*127));data=p.serialize_coefficients(coeff)
        self.assertEqual(p.validate_native(data,coeff)['saved_f32_coefficient_l1'],{'numerator':127,'denominator':4})
        with self.assertRaises(p.PairedLinearError):p.validate_native(data,self.coefficients)

    def test_exact_norm_endpoint_and_smallest_subnormal(self):
        coeff=struct.pack('<254f',39.5,*([0.0]*253));p.validate_native(p.serialize_coefficients(coeff),coeff)
        raw=struct.unpack('<I',coeff[:4])[0]
        bad=struct.pack('<I',raw+1)+coeff[4:]
        with self.assertRaises(p.PairedLinearError):p.serialize_coefficients(bad)
        tiny=struct.pack('<254f',19.75,-19.75,*([0.0]*252))
        bad=tiny[:8]+struct.pack('<I',1)+tiny[12:]
        with self.assertRaisesRegex(p.PairedLinearError,'L1 exceeds'):p.serialize_coefficients(bad)

    def test_shape_finite_and_independent_zero_sign(self):
        for coeff in (b'',self.coefficients+b'\0',bytearray(self.coefficients),struct.pack('<I',p.SIGN_BIT)+self.coefficients[4:],
                      struct.pack('<I',0x7f800000)+self.coefficients[4:],struct.pack('<I',0x7fc00001)+self.coefficients[4:]):
            with self.subTest(type=type(coeff)):
                with self.assertRaises(p.PairedLinearError):p.serialize_coefficients(coeff)

    def test_all_frozen_coordinate_groups_and_unused_sign(self):
        offsets=[p.FT_OFFSET,p.FT_END-2,p.FT_END,p.L2_OFFSET-2]
        offsets += [p.l2_cell(r,c) for r in (0,1,2,255,256,257,258,511) for c in range(4)]
        offsets += [p.L2_BIAS_OFFSET+4*c for c in range(4)]+[p.OUT_OFFSET+4*c for c in range(4)]
        offsets += [p.l2_cell(r,c) for r in (0,1,256,257) for c in (4,5)]
        offsets += [p.l2_cell(r,c) for r in (0,2,511) for c in (6,31)]
        offsets += [p.L2_BIAS_OFFSET+4*c for c in (4,5,6,31)]+[p.OUT_OFFSET+4*c for c in (4,5,6,31)]+[p.OUT_BIAS_OFFSET]
        for off in offsets:
            with self.subTest(offset=off):
                data=bytearray(self.data);data[off]^=1
                with self.assertRaises(p.PairedLinearError):p.validate_native(bytes(data),self.coefficients)
        for off in (p.l2_cell(2,5),p.l2_cell(258,4)):
            data=bytearray(self.data);struct.pack_into('<I',data,off,0)
            with self.assertRaises(p.PairedLinearError):p.validate_native(bytes(data),self.coefficients)
        wrong=bytearray(p.reference());wrong[8]^=1
        with self.assertRaises(p.PairedLinearError):p.serialize_coefficients(self.coefficients,bytes(wrong))

    def test_own_sidecar_and_no_old_validator_relaxation(self):
        side=p.sidecar_for_fit(self.data,self.coefficients,self.raw,self.binding)
        self.assertEqual(p.validate_sidecar(side,self.data,self.coefficients,self.raw,self.binding)['status'],'pure-byte-provenance-pass')
        with self.assertRaises(bounded_material.BoundedMaterialError):bounded_material.validate_weights(self.data)
        self.assertEqual(material_init.SOURCE_HASHES,p.EXPECTED_SOURCE_HASHES)
        for key,value in (('nnue_output','residualMaterial'),('checkpoint_hash','0'*16),('extra_success',True)):
            bad=copy.deepcopy(side);bad[key]=value
            with self.assertRaises(p.PairedLinearError):p.validate_sidecar(bad,self.data,self.coefficients,self.raw,self.binding)
        for key,value in (('epochs',False),('adam_used',0),('actual_core_bound_verified',True),('weight_bytes',float(len(self.data))),('mode','bounded-material-v1')):
            bad=copy.deepcopy(side);bad['paired_linear_constrained_ridge_v1'][key]=value
            with self.assertRaises(p.PairedLinearError):p.validate_sidecar(bad,self.data,self.coefficients,self.raw,self.binding)

    def test_fit_flags_type_paths_external_receipt_and_schema(self):
        for key,value in (('epochs',False),('adam_used',0),('resume_used',True),('trainer_used',True),('cleanup_verified',False),
                          ('final_used',True),('optimizer','Adam'),('selected_epoch',3),('schema','sekirei.training-run.v1')):
            bad=copy.deepcopy(self.fit_run);bad[key]=value;raw=json.dumps(bad).encode();binding=copy.deepcopy(self.binding)
            binding['fit_receipt']={'path':str(p.FIT_RECEIPT),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
            with self.assertRaises(p.PairedLinearError):p.validate_fit_run(raw,self.data,self.coefficients,binding)
        with self.assertRaises(p.PairedLinearError):p.validate_fit_run(self.raw+b' ',self.data,self.coefficients,self.binding)
        bad=copy.deepcopy(self.binding);bad['solver_certificate']['path']='/tmp/certificate.json'
        with self.assertRaises(p.PairedLinearError):p.validate_fit_run(self.raw,self.data,self.coefficients,bad)
        bad=copy.deepcopy(self.binding);bad['source_helpers_sha256'].pop('paired_linear_activation.py')
        with self.assertRaises(p.PairedLinearError):p.validate_fit_run(self.raw,self.data,self.coefficients,bad)

    def test_json_duplicate_nonfinite_and_bytes(self):
        for data in (b'{"x":1,"x":2}',b'{"x":NaN}',bytearray(b'{}')):
            with self.assertRaises(p.PairedLinearError):p.strict_json(data)

    def test_no_actual_entry_even_toggle(self):
        with self.assertRaisesRegex(p.PairedLinearError,'PROTOTYPE_ONLY'):p.entry('no-files')
        with mock.patch.object(p,'PROTOTYPE_ONLY',False):
            with self.assertRaises(NotImplementedError):p.entry()
        with mock.patch.dict(material_init.SOURCE_HASHES,{'crates/sekirei-core/src/nnue.rs':'0'*64}):
            with self.assertRaises(p.PairedLinearError):p.validate_native(self.data,self.coefficients)
            with self.assertRaises(p.PairedLinearError):p.serialize_coefficients(self.coefficients)


if __name__=='__main__':unittest.main()
