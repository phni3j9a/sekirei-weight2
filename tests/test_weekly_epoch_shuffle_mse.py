"""Lightweight fixtures: integer row order and genuine raw-trace consumer.

No original SFEN, labels, model arrays, Rust execution or process locks.
Synthetic traces are explicit fixtures, never real training evidence.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
ROOT=Path(__file__).resolve().parents[1]
TREE=ROOT
sys.path.insert(0,str(TREE/'scripts'))
import weekly_nonlinear_epoch_shuffle_mse_order as target

CASES=[]


def record(tag, body): return tag.encode()+b' '+json.dumps(body,separators=(',',':')).encode()


def trace(count=9):
    records=[]
    for epoch in range(1,4):
        values=target.permutation(count,epoch=epoch)
        identity=target.order_identity(values)
        base=(epoch-1)*count
        records.append(('PAIRED_EPOCH_BEGIN', {'epoch':epoch,'start_step':base,'float':{'fixture':True}}))
        records.append(('PAIRED_SHUFFLE_BEGIN', {'epoch':epoch,'seed':20261006,'count':count,
            'algorithm':target.ALGORITHM,'permutation':identity}))
        for start in range(0,count,1024):
            chunk=values[start:start+1024]
            records.append(('PAIRED_SHUFFLE_ROWS', {'epoch':epoch,'start_position':start,'start_step':base+start,
                'end_step':base+start+len(chunk),'indices':chunk}))
        records.append(('PAIRED_SHUFFLE_COMPLETE', {'epoch':epoch,'consumed_count':count,'start_step':base,
            'end_step':base+count,'consumed':identity}))
        records.append(('PAIRED_EPOCH_COMPLETE', {'epoch':epoch,'positions':count,'end_step':base+count,'float':{'fixture':True}}))
    return records


def raw(records): return b'\n'.join(record(tag,body) for tag,body in records)+b'\n'


RECIPE={'schema':'sekirei.white-view-paired-nonlinear-epoch-shuffle-mse-recipe.v1',
    'mode':'white-view-diverse-games-shuffle-seed42-e3-v1','seed':42,'objective':'absolute-cp-mse',
    'shuffle_seed':20261006,'row_order':target.declaration()}


class Cases(unittest.TestCase):
    def accepted(self,name,values,count=9):
        result=target.observations(raw(values),deepcopy(RECIPE),count=count)
        self.assertEqual(len(result),3)
        self.assertEqual(result[-1]['end_step'],3*count)
        CASES.append({'name':name,'expected':'accept','passed':True})
        return result

    def rejected(self,name,values=None,*,count=9,recipe=None,data=None):
        with self.assertRaises(ValueError):
            target.observations(raw(values) if data is None else data, deepcopy(RECIPE) if recipe is None else recipe,count=count)
        CASES.append({'name':name,'expected':'reject','passed':True})

    def test_named_rng_vectors(self):
        rng=target.SplitMix64(0)
        self.assertEqual([rng.next(),rng.next()],[0xe220a8397b1dcdaf,0x6e789e6aa1b965f4])
        # Independent scalar reference uses separate constants and arithmetic.
        for epoch in range(1,4):
            self.assertEqual(target.permutation(8,epoch=epoch), independent(8,epoch))
        CASES.append({'name':'published splitmix64 state0 vectors and independent scalar permutations','expected':'accept','passed':True})

    def test_full_bijection_epoch_stability_and_pins(self):
        seen=[]
        for epoch in range(1,4):
            order=target.permutation(112681,epoch=epoch)
            self.assertEqual(sorted(order),list(range(112681)))
            self.assertEqual(order,target.permutation(112681,epoch=epoch))
            self.assertEqual(order,independent(112681,epoch))
            self.assertEqual(target.order_identity(order),{k:RECIPE['row_order']['epochs'][epoch-1][k] for k in ('bytes','sha256')})
            seen.append(order)
        self.assertNotEqual(seen[0],seen[1]);self.assertNotEqual(seen[1],seen[2]);self.assertNotEqual(seen[0],seen[2])
        CASES.append({'name':'3x112681 full bijection stable independent and distinct epoch orders','expected':'accept','passed':True})

    def test_count_one_and_small_bijections(self):
        for n in (1,2,7,32,1024,1025):
            for e in (1,2,3):self.assertEqual(sorted(target.permutation(n,epoch=e)),list(range(n)))
        CASES.append({'name':'small and chunk-boundary bijections','expected':'accept','passed':True})

    def test_invalid_integer_seed_epoch_count_controls(self):
        for label,kwargs in [('init_seed_as_shuffle',{'seed':42}),('boolean_seed',{'seed':True}),
            ('epoch0',{'epoch':0}),('epoch4',{'epoch':4}),('epochbool',{'epoch':True}),
            ('zero_count',{'count':0}),('negative_count',{'count':-1}),('boolean_count',{'count':True})]:
            with self.subTest(label=label),self.assertRaises(ValueError):
                target.permutation(**({'count':9}|kwargs))
            CASES.append({'name':label,'expected':'reject','passed':True})

    def test_complete_small_trace(self): self.accepted('actual fixture consumed 3x9',trace())

    def test_production_count_trace(self):
        observed=self.accepted('actual fixture consumed 3x112681 and 338043 updates',trace(112681),count=112681)
        self.assertEqual(sum(e['positions'] for e in observed),338043)
        self.assertEqual([e['chunks'] for e in observed],[111,111,111])

    def test_chunk_boundaries(self):
        for n in (1024,1025,2048,2049):self.accepted('contiguous chunk boundary '+str(n),trace(n),count=n)

    def test_missing_duplicate_outside_wrong_order_indices(self):
        for name,mutation in [
            ('missing_index',lambda v:v[2][1]['indices'].pop()),
            ('duplicate_index',lambda v:v[2][1]['indices'].__setitem__(1,v[2][1]['indices'][0])),
            ('outside_index',lambda v:v[2][1]['indices'].__setitem__(0,9)),
            ('negative_index',lambda v:v[2][1]['indices'].__setitem__(0,-1)),
            ('bool_index',lambda v:v[2][1]['indices'].__setitem__(0,True)),
            ('wrong_order',lambda v:v[2][1]['indices'].reverse()),
            ('missing_chunk',lambda v:v.pop(2)),
            ('duplicate_chunk',lambda v:v.insert(3,deepcopy(v[2]))),
        ]:
            values=trace();mutation(values);self.rejected(name,values)

    def test_update_step_and_count_controls(self):
        for index,key,value in [(0,'start_step',1),(2,'start_position',1),(2,'start_step',1),(2,'end_step',8),
            (3,'consumed_count',8),(3,'start_step',1),(3,'end_step',8),(4,'positions',8),(4,'end_step',8),
            (2,'end_step',True)]:
            values=trace();values[index][1][key]=value;self.rejected(f'control_{index}_{key}_{value}',values)

    def test_digest_algorithm_seed_controls(self):
        for name,mutation in [('planned_sha',lambda v:v[1][1]['permutation'].__setitem__('sha256','0'*64)),
            ('consumed_sha',lambda v:v[3][1]['consumed'].__setitem__('sha256','0'*64)),
            ('planned_bytes',lambda v:v[1][1]['permutation'].__setitem__('bytes',71)),
            ('consumed_bytes',lambda v:v[3][1]['consumed'].__setitem__('bytes',71)),
            ('algorithm',lambda v:v[1][1].__setitem__('algorithm','other')),
            ('shuffle_seed',lambda v:v[1][1].__setitem__('seed',42)),
            ('shuffle_seed_bool',lambda v:v[1][1].__setitem__('seed',True))]:
            values=trace();mutation(values);self.rejected(name,values)

    def test_epoch_enclosure_and_order_controls(self):
        for name,mutation in [('rows_before_begin',lambda v:v.insert(0,v.pop(2))),
            ('unclosed_epoch',lambda v:v.pop()),('missing_epoch',lambda v:v.__delitem__(slice(5,10))),
            ('extra_epoch',lambda v:v.extend(deepcopy(v[:5]))),('repeat_epoch',lambda v:v[5][1].__setitem__('epoch',1)),
            ('duplicate_begin',lambda v:v.insert(1,deepcopy(v[0]))),('shuffle_complete_before_rows',lambda v:v.insert(2,v.pop(3))),
            ('epoch_complete_before_shuffle_complete',lambda v:v.insert(3,v.pop(4)))]:
            values=trace();mutation(values);self.rejected(name,values)

    def test_unknown_duplicate_and_nonfinite_json_controls(self):
        values=trace();values.insert(3,('PAIRED_SHUFFLE_UNKNOWN',{}));self.rejected('unknown_record',values)
        base=raw(trace())
        self.rejected('duplicate_json_key',data=base.replace(b'"consumed_count":9',b'"consumed_count":9,"consumed_count":9',1))
        self.rejected('nonfinite_json',data=base.replace(b'"consumed_count":9',b'"consumed_count":NaN',1))
        values=trace();values[2][1]['unexpected']=1;self.rejected('unknown_chunk_key',values)

    def test_recipe_mutations(self):
        for key,value in [('schema','sekirei.white-view-paired-nonlinear-recipe.v1'),('mode','old'),('seed',20261006),
            ('objective','absolute-cp-twice-huber'),('shuffle_seed',None),('shuffle_seed',42),('seed',True)]:
            recipe=deepcopy(RECIPE);recipe[key]=value;self.rejected('recipe_'+key+'_'+str(value),trace(),recipe=recipe)
        recipe=deepcopy(RECIPE);recipe['row_order']['epochs'][0]['sha256']='0'*64
        self.rejected('recipe_recomputable_declaration_changed',trace(),recipe=recipe)

    def test_published_source_variant_matches_row_order_declaration(self):
        p=ROOT/'preparations/white-view-paired-nonlinear-epoch-shuffle-mse-v1'
        variant=json.loads((p/'public-source-manifest.json').read_text())
        self.assertEqual(variant['shuffle_seed'],20261006)
        self.assertEqual(variant['objective'],'absolute-cp-mse')
        self.assertEqual(variant['row_order'],RECIPE['row_order'])
        self.assertEqual(len(variant['changed_files']),3)
        for name,ref in variant['changed_files'].items():
            raw=(p/'source'/name).read_bytes()
            self.assertEqual(len(raw),ref['after']['bytes'])
            self.assertEqual(hashlib.sha256(raw).hexdigest(),ref['after']['sha256'])
        patch=(p/'candidate.patch').read_bytes()
        self.assertEqual(hashlib.sha256(patch).hexdigest(),variant['candidate_patch']['sha256'])

    def test_immutable_numeric_body_and_mutated_outer_epoch(self):
        original=(ROOT/'preparations/white-view-paired-nonlinear-rust-v1/compiled-source/source/crates/sekirei-train/src/paired_nonlinear_positions.rs').read_bytes()
        revised=(ROOT/'preparations/white-view-paired-nonlinear-epoch-shuffle-mse-v1/source/crates/sekirei-train/src/paired_nonlinear_positions.rs').read_bytes()
        prefix=original.split(b'    /// Complete keyed original teacher cache only;')[0]
        self.assertEqual(len(prefix),34791)
        self.assertTrue(revised.startswith(prefix))
        self.assertIn(b'consumption.record(original_index,self.weights.step)?;',revised)
        before=revised.index(b'self.train_paired_nonlinear_position(&sample.board,cache[&sfen],context)?;')
        self.assertGreater(revised.index(b'consumption.record(original_index,self.weights.step)?;'),before)
        self.assertIn(b'consumption.finish(self.weights.step)?;',revised)


def independent(n,e):
    # Separate scalar reference uses rejection boundary as full2^64 modulo.
    mask=2**64-1;gamma=11400714819323198485
    state=20261006 ^ ((e*gamma)%2**64)
    values=list(range(n))
    for k in range(n,1,-1):
        threshold=(2**64)%k
        while True:
            state=(state+gamma)%2**64
            a=((state^(state//2**30))*13787848793156543929)%2**64
            b=((a^(a//2**27))*10723151780598845931)%2**64
            number=b^(b//2**31)
            if number>=threshold:break
        j=number%k;values[k-1],values[j]=values[j],values[k-1]
    return values


if __name__ == "__main__":
    unittest.main()
