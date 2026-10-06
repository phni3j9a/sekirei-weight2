import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

ROOT=Path(__file__).resolve().parent
ARTIFACTS=ROOT
spec=importlib.util.spec_from_file_location('quant_sample_request',ROOT/'sample_request.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def bits(value):return struct.pack('>f',value).hex()
def row(index):
    # Shape fixture only. Rust separately parses canonical/legal board semantics.
    return {'schema_version':1,'sfen':f'fixture-board-{index} b - 16',
            'source':{'kind':'gensfen-pack','path':f'game-{index}','ply':16},
            'tags':{'phase':'middlegame','side_to_move':'black'}}
def raw_rows(count):return b''.join((json.dumps(row(i))+'\n').encode() for i in range(count))
def records(selected=2):
    return [{'split':split,'row_index':index,'raw_master_cp_f32_bits':bits(1.5),
             'native_dequant_cp_f32_bits':bits(1.25),'native_core_cp':1,
             'all_state_bits_unchanged':True} for split in ('train','holdout') for index in m.indices(m.COUNTS[split],selected)]

BANNER=b'Training floats: FTZ/DAZ enabled (MXCSR bits 15/6)\n'
TERMINAL=b'complete raw-master/native sampled count512; step338043 unchanged; optimizer_updates0; no teacher/search/export write\n'
def probe_log(values=None):
    fp={'schema':'sekirei.white-view-paired-nonlinear-float-snapshot.v1','float_policy':'x86-ftz-daz',
        'environment_variable':'SEKIREI_TRAIN_FTZ_DAZ','environment_value':'1','mxcsr_raw_bits':0x9fc1,
        'mxcsr_control_bits':0x9fc0,'mxcsr_status_bits':1,'mxcsr_status_mask':63,'required_control_bits':0x9fc0}
    raw=BANNER+b''.join((json.dumps(r)+'\n').encode() for r in (records(256) if values is None else values))
    return raw+b'float_before='+json.dumps(fp).encode()+b'\nfloat_after='+json.dumps(fp).encode()+b'\n'+TERMINAL

class ContractTests(unittest.TestCase):
    def test_fixed512_indices_cover_endpoints_unique(self):
        for count in m.COUNTS.values():
            got=m.indices(count);self.assertEqual(len(got),256)
            self.assertEqual(got,sorted(set(got)));self.assertEqual(got[0],0);self.assertEqual(got[-1],count-1)
            self.assertEqual(got,[i*(count-1)//255 for i in range(256)])
    def test_index_invalid_size_or_bool_rejected(self):
        for count,n in ((1,2),(5,6),(True,2),(5,True),(5,1)):
            with self.subTest(count=count,n=n),self.assertRaises(ValueError):m.indices(count,n)
    def test_sample_original_order_no_label_required(self):
        games={f'game-{i}':'train' for i in range(9)}
        samples=m.sample_rows(raw_rows(9),9,games,'train',3)
        self.assertEqual([x['row_index'] for x in samples],[0,4,8])
        self.assertTrue(all('score_cp' not in x for x in samples))
    def test_only_selected_rows_json_parsed(self):
        raw=raw_rows(9).splitlines(keepends=True);raw[1]=b'not-json\n'
        self.assertEqual([i for i,v in m.select_rows(b''.join(raw),9,3)],[0,4,8])
    def test_all_physical_rows_count_and_lf_required(self):
        for raw,count in ((raw_rows(3),4),(raw_rows(3)[:-1],3),(raw_rows(3)+b'\n',3),(raw_rows(3).replace(b'\n',b'\r\n'),3)):
            with self.assertRaises(ValueError):m.select_rows(raw,count,2)
    def test_holdout_membership_phase_and_ply_not_retagged(self):
        games={f'game-{i}':'train' for i in range(3)}
        with self.assertRaises(ValueError):m.sample_rows(raw_rows(3),3,games,'holdout',2)
        for where,new in (('phase','endgame'),('ply',17),('side_to_move','white')):
            changed=row(0)
            changed['source' if where=='ply' else 'tags'][where]=new
            raw=(json.dumps(changed)+'\n').encode()+raw_rows(3).split(b'\n',1)[1]
            with self.assertRaises(ValueError):m.sample_rows(raw,3,games,'train',2)
    def test_development_final_split_refused(self):
        for split in ('development','final'):
            with self.assertRaises(ValueError):m.sample_rows(raw_rows(3),3,{},split,2)
    def test_duplicate_json_keys_rejected(self):
        for raw in ('{"x":1,"x":2}','{"x":{"z":1,"z":2}}','{"x":NaN}'):
            with self.assertRaises(ValueError):m.unique(raw)
    def test_raw_float_quantization_integer_gaps_distinct_cp(self):
        d=m.aggregate(records(),2)
        for split in m.COUNTS:
            got=d['splits'][split]
            self.assertEqual(got['raw_master_minus_native_core']['mean_absolute_cp'],.5)
            self.assertEqual(got['raw_master_minus_native_dequant']['mean_absolute_cp'],.25)
            self.assertEqual(got['native_dequant_minus_native_core']['mean_absolute_cp'],.25)
        self.assertFalse(d['universal_bound_proved']);self.assertFalse(d['adoption_claimed'])
        self.assertEqual(d['optimizer_updates'],0)
    def test_negative_zero_f32_bits_preserved(self):
        self.assertEqual(struct.pack('>f',m.f32('80000000')).hex(),'80000000')
    def test_combined_supervised_log_requires512_and_actual_fp_metadata(self):
        fp={'schema':'sekirei.white-view-paired-nonlinear-float-snapshot.v1','float_policy':'x86-ftz-daz',
            'environment_variable':'SEKIREI_TRAIN_FTZ_DAZ','environment_value':'1','mxcsr_raw_bits':0x9fc1,
            'mxcsr_control_bits':0x9fc0,'mxcsr_status_bits':1,'mxcsr_status_mask':63,'required_control_bits':0x9fc0}
        raw=b''.join((json.dumps(r)+'\n').encode() for r in records(256))
        terminal=b'complete raw-master/native sampled count512; step338043 unchanged; optimizer_updates0; no teacher/search/export write\n'
        raw+=b'float_before='+json.dumps(fp).encode()+b'\nfloat_after='+json.dumps(fp).encode()+b'\n'+terminal
        raw=BANNER+raw
        got,snapshots=m.records_from_log(raw);self.assertEqual(len(got),512);self.assertEqual(len(snapshots),2)
        for changed in (raw.replace(b'float_before=',b'float_after='),raw[:-len(terminal)],raw+b'unexpected\n',
                        raw.replace(b'"environment_value": "1"',b'"environment_value": "0"')):
            with self.assertRaises(ValueError):m.records_from_log(changed)
    def test_nonfinite_and_boundary_cp_rejected(self):
        for value in ('7f800000','ff800000','7fc00000',bits(899000.0)):
            with self.assertRaises(ValueError):m.f32(value)
    def test_complete_order_split_state_and_types_required(self):
        values=records()
        for key,new in (('split','final'),('row_index',True),('native_core_cp',1.0),('all_state_bits_unchanged',False),('extra',1)):
            changed=copy.deepcopy(values);changed[0][key]=new
            with self.assertRaises(ValueError):m.aggregate(changed,2)
        with self.assertRaises(ValueError):m.aggregate(values[:-1],2)
        with self.assertRaises(ValueError):m.aggregate(list(reversed(values)),2)
    def test_read_ref_pin_regular_symlink_and_modes(self):
        with tempfile.TemporaryDirectory() as name:
            p=Path(name)/'input';p.write_bytes(b'fixture')
            ref={'path':str(p),'bytes':7,'sha256':hashlib.sha256(b'fixture').hexdigest()}
            self.assertEqual(m.read_ref(ref),b'fixture')
            for key,value in (('bytes',True),('bytes',8),('sha256','0'*64),('unknown',1)):
                wrong=dict(ref);wrong[key]=value
                with self.assertRaises(ValueError):m.read_ref(wrong)
            link=Path(name)/'link';link.symlink_to(p)
            with self.assertRaises(ValueError):m.read_ref(dict(ref,path=str(link)))
    def test_prepare_fresh_guard_precedes_input_reads(self):
        with tempfile.TemporaryDirectory() as name:
            output=Path(name)
            for mode in ('foreign-mode',m.MODE):
                with self.assertRaises(ValueError):m.prepare({'mode':mode,'status':'source-only-prepared'},output)
            self.assertEqual(list(output.iterdir()),[])
    def test_frozen_numeric_blocks_verbatim_and_old_functions_unchanged(self):
        meta=json.loads((ROOT/'public-source-manifest.json').read_text())
        base=ROOT.parent/'white-view-paired-nonlinear-rust-v1/compiled-source/source'
        relative='crates/sekirei-train/src/'
        positions=base/(relative+'paired_nonlinear_positions.rs')
        source=positions.read_text()
        self.assertEqual(hashlib.sha256(positions.read_bytes()).hexdigest(),meta['original_numeric_positions_module']['sha256'])
        diag=(ROOT/'source'/relative/'paired_nonlinear_raw_master_diag.rs').read_text()
        anchors=[('// FINITE OBSERVATION:','/// Declaration-only context.'),
                 ('        // FT accumulation','        for &x in relu_us.iter().chain(relu_them.iter())'),
                 ('        // L2 accumulation','        for o in 0..L2 {\n            if relu_l2[o] > 0.0'),
                 ('        // Output\n','        self.output_sum += score as f64;')]
        for i,(start,end) in enumerate(anchors):
            self.assertEqual(source.count(start),1);self.assertEqual(source.count(end),1)
            chunk=source[source.index(start):source.index(end,source.index(start))]
            self.assertIn(chunk,diag);self.assertEqual(hashlib.sha256(chunk.encode()).hexdigest(),meta['numeric_blocks'][i]['sha256'])
        for name,ref in meta['diagnostic_preimages'].items():
            old=(base/name).read_bytes()
            if name.endswith('main.rs'):
                self.assertEqual(hashlib.sha256(old).hexdigest(),meta['baseline_alignment']['main_public_preimage']['sha256'])
                anchor=b'#[cfg(feature = "nnue_white_view_aux_tied")]\nmod paired_nonlinear_sha256;'
                self.assertEqual(old.count(anchor),1)
                old=old.replace(anchor,b'#[cfg(feature = "nnue_white_view_aux_tied")]\nmod weekly_nonlinear_profile;\n'+anchor)
            self.assertEqual(hashlib.sha256(old).hexdigest(),ref['sha256'])
            post=(ROOT/'source'/name).read_bytes()
            if name.endswith('main.rs'):
                begin=post.index(b'        if argv.first().map(String::as_str) == Some("diagnose-raw-master-quantization")')
                end=post.index(b'        if argv.first().map(String::as_str) == Some("verify-paired-nonlinear-inputs")',begin)
                self.assertEqual(post[:begin]+post[end:],old)
            else:self.assertTrue(post.startswith(old))
        self.assertIn('const PROTOTYPE_ONLY:bool=true;',diag)
        self.assertNotIn('&mut TrainWeights',diag);self.assertNotIn('train_paired_nonlinear_position(',diag)
        self.assertNotIn('commit_completed_native03(',diag);self.assertNotIn('Searcher',diag)
        self.assertIn('checkpoint::diagnostic_source_ref(path,info)?',diag)
        self.assertNotIn('Board::from_sfen(',diag)
        self.assertIn('Board::from_sfen_rules_only(',diag)
        self.assertIn('bound::exact(&manifest["files"][&name],&bound::info_value(&declared))',diag)
        self.assertIn('declared.path==manifest_path.parent().ok_or("manifest root")?.join(&name)',diag)
        checkpoint=(ROOT/'source'/relative/'paired_nonlinear_checkpoint_io.rs').read_text()
        self.assertIn('validate_raw_source_identity(path,info)?;',checkpoint)

    def test_startup_banner_missing_rejected(self):
        with self.assertRaises(ValueError):m.records_from_log(probe_log()[len(BANNER):])
    def test_startup_banner_duplicate_rejected(self):
        with self.assertRaises(ValueError):m.records_from_log(BANNER+probe_log())
    def test_startup_banner_variants_rejected(self):
        variants=(BANNER.lower(),b' '+BANNER,BANNER.replace(b'15/6',b'15 / 6'),BANNER[:-1]+b' extra\n')
        for banner in variants:
            with self.subTest(banner=banner),self.assertRaises(ValueError):m.records_from_log(banner+probe_log()[len(BANNER):])
    def test_startup_banner_midstream_rejected(self):
        lines=probe_log().splitlines(keepends=True)
        for changed in (b''.join(lines[:2]+[BANNER]+lines[2:]),b''.join(lines[1:2]+[BANNER]+lines[2:])):
            with self.assertRaises(ValueError):m.records_from_log(changed)
    def test_startup_banner_unknown_extra_rejected(self):
        for changed in (b'unknown\n'+probe_log(),BANNER+b'unknown\n'+probe_log()[len(BANNER):],probe_log()+b'unknown\n',b'\n'+probe_log()):
            with self.assertRaises(ValueError):m.records_from_log(changed)
    def test_startup_banner_retains512_fp_terminal_finite_and_state_checks(self):
        values=records(256);rows,fp=m.records_from_log(probe_log(values))
        self.assertEqual(m.aggregate(rows)['sample_count'],512);self.assertEqual(set(fp),{'float_before','float_after'})
        for raw in (probe_log(values[:-1]),probe_log(values)[:-len(TERMINAL)],probe_log(values)+TERMINAL,
                    probe_log(values).replace(b'"mxcsr_control_bits": 40896',b'"mxcsr_control_bits": 0')):
            with self.assertRaises(ValueError):m.records_from_log(raw)
        for key,value in (('raw_master_cp_f32_bits','7f800000'),('raw_master_cp_f32_bits',bits(899000.0)),
                          ('native_dequant_cp_f32_bits','7fc00000'),('native_core_cp',1.0),
                          ('row_index',True),('all_state_bits_unchanged',False),('extra',1)):
            changed=copy.deepcopy(values);changed[0][key]=value
            rows,_=m.records_from_log(probe_log(changed))
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):m.aggregate(rows)

if __name__=='__main__':unittest.main()
