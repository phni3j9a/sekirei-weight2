"""SOURCE ONLY memory consumer; no files, processes, training or actual gate IO.

Root must freeze proposed parent/proof body producers before enabling its outer.
This module does not turn child cleanup flags into parent lifecycle evidence.
"""
from array import array
from fractions import Fraction
import hashlib
import json
import math
import re
import struct
import sys

import paired_nonlinear_proof_contract as probe
import white_view_build_contract as build

PROTOTYPE_ONLY=True
MODE=probe.MODE
PLAN_SHA='111f8e3f1bf404c042c9e3db3e46f46659f3f609c405c30234dbf47778e03729'
MANIFEST_SHA='ecc419da180b86b046e1af507e6de0e218d271a5d9eaa9bff07f19ec072719a6'
TEACHER='external:suisho11beta-1m-pack:376d4ef6e503d2ebe687f99e873103845b6eddc0c04d0d08e9b9c785ea061b8d'
FEATURE='flat_white_view_aux_tied_v1'
N,H,STEPS=112681,5895,338043
INPUT,L1,L2=2420,256,32
NATIVE_BYTES=1305356
CHECKPOINT_SCHEMA='sekirei.white-view-paired-nonlinear-fullstate-bits-checkpoint.v1'
EXPORT_SCHEMA='sekirei.white-view-paired-nonlinear-export-stage.v1'
ROLES={'plan','source_binding','parent_preflight','training_build','engine_manifest','engine_identity','recipe',
       'manifest','train_positions','train_labels','holdout_positions','holdout_labels','reference03',
       'checkpoint','native03','export_stage','training_completion','metadata','core','incremental','fixtures'}
VECTORS={'ft':INPUT*L1,'ft_bias':L1,'l2':2*L1*L2,'l2_bias':L2,'out':L2}
MOMENTS={'ft':('ft_m','ft_v'),'ft_bias':('bias_m','bias_v'),'l2':('l2_m','l2_v'),
         'l2_bias':('l2bias_m','l2bias_v'),'out':('out_m','out_v')}
STATE_KEYS={*VECTORS,*(k for pair in MOMENTS.values() for k in pair),'out_bias','obias_m','obias_v','step'}
PROOF_SOURCE_SHA={'core':'ba362852b1e86a765ff06ab1036158eb06604bfbfeb04f8d3f831df9c116cf9b',
    'incremental':'5d6c1fede3e686841ff74d16e7c40bc822e394415d6ff14c37a69ba09a266e8b',
    'native_contract':'95507c5360cdd1b0929369b8a15ebaf3ea621c2c1ffaf50f69f2580c9e27a5b3'}
VALUES=(100,430,470,640,680,890,1040,0,600,600,600,640,1150,1300)
KINDS=('P','L','N','S','G','B','R','K','+P','+L','+N','+S','+B','+R')

def require(ok,message):
    if not ok:raise ValueError(message)

def runtime_guard():
    raise RuntimeError('SOURCE ONLY nonlinear gate outer disabled before I/O')

def exact(a,b):
    if type(a)is not type(b):return False
    if type(a)is dict:return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if type(a)is list:return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b

def obj(v,keys):
    require(type(v)is dict and set(v)==set(keys),'exact typed object keys required')
    return v

def fixed(v,want):
    require(type(v)is dict and all(k in v and exact(v[k],x) for k,x in want.items()),'typed fixed fields differ')

def uint(v,maximum=(1<<64)-1):
    require(type(v)is int and 0<=v<=maximum,'strict nonnegative integer required')
    return v

def sha(v):
    require(type(v)is str and len(v)==64 and all(c in '0123456789abcdef' for c in v),'lower SHA64 required')
    return v

def absolute(v):
    require(type(v)is str and v.startswith('/') and len(v)>1 and '\0' not in v
            and all(p not in ('','.','..') for p in v.split('/')[1:]),'lexical canonical absolute path required')
    return v

def info(v):
    obj(v,{'bytes','sha256'});uint(v['bytes']);sha(v['sha256']);return v

def fullref(v):
    obj(v,{'path','bytes','sha256'});absolute(v['path']);info({k:v[k] for k in ('bytes','sha256')});return v

def identity_map(v):
    require(type(v)is dict and v,'nonempty identity map required')
    for p,i in v.items():absolute(p);info(i)
    return v

def strict_json(raw):return probe.strict_json(raw)

def digest(raw):
    require(type(raw)is bytes,'raw immutable bytes required')
    return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

class Bytes:
    """All raw bytes are caller supplied; this class performs no filesystem IO."""
    def __init__(self,raw,expected):
        identity_map(expected);require(type(raw)is dict and set(raw)==set(expected),'whole frozen raw inventory required')
        for p,b in raw.items():require(exact(digest(b),expected[p]),'raw input identity differs')
        self.raw=raw;self.expected=expected
    def read(self,r):
        fullref(r);require(exact(self.expected.get(r['path']),{k:r[k] for k in ('bytes','sha256')}),'transitive fullref absent/rebound')
        b=self.raw[r['path']];require(exact(digest(b),{k:r[k] for k in ('bytes','sha256')}),'raw byte SHA differs before parse');return b
    def json(self,r):return strict_json(self.read(r))
    def map(self,v):
        identity_map(v)
        for p,i in v.items():self.read({'path':p,**i})
    def refs_in(self,v,required_map=None):
        if type(v)is dict:
            if set(v)=={'path','bytes','sha256'}:
                self.read(v)
                if required_map is not None:require(exact(required_map.get(v['path']),{k:v[k] for k in ('bytes','sha256')}),'transitive reference lost required predecessor closure')
            else:
                for x in v.values():self.refs_in(x,required_map)
        elif type(v)is list:
            for x in v:self.refs_in(x,required_map)

def bits(x):return struct.unpack('<I',struct.pack('<f',x))[0]

def frombits(x):
    uint(x,(1<<32)-1);f=struct.unpack('<f',struct.pack('<I',x))[0]
    require(math.isfinite(f),'nonfinite binary32 state');return f

def group(k):return None if k==7 else (0 if k in (0,8) else 1)

def canonical_reference():
    """Independent seed42 initializer03; mirrors the frozen Rust draw order."""
    rng=42
    def sign():
        nonlocal rng
        rng=(rng*6364136223846793005+1442695040888963407)&((1<<64)-1)
        return 1 if rng>>63 else -1
    ft=array('h',[0])*(INPUT*L1)
    offsets=(0,18,22,26,30,34,36);sizes=(18,4,4,4,4,2,2)
    for f in range(INPUT):
        for j in range(2,L1):ft[f*L1+j]=sign()
        if f<2268:
            k,opp=(f%28)//2,f%2;g=group(k)
            if opp==0 and g is not None:ft[f*L1+g]=VALUES[k]//2
        else:
            bank,t=divmod(f-2268,38);color,perspective=divmod(bank,2)
            k=next(k for k in range(7) if offsets[k]<=t<offsets[k]+sizes[k])
            if color==perspective:ft[f*L1+group(k)]=VALUES[k]//2
    l2=array('f',[0.0])*(2*L1*L2)
    for row,col in ((0,0),(1,1),(256,2),(257,3)):l2[row*L2+col]=1.0
    for row in range(2*L1):
        if row%L1>=2:
            for col in range(4,L2):l2[row*L2+col]=sign()/256.0
    for donor in range(2):
        for t in range(38):
            a=(2268+donor*38+t)*L1;b=(2268+(3-donor)*38+t)*L1
            ft[b:b+L1]=ft[a:a+L1]
    return {'ft':ft,'ft_bias':array('h',[64])*L1,'l2':l2,
            'l2_bias':array('f',[0.0]*4+[4.0]*28),
            'out':array('f',[8192.0,8192.0,-8192.0,-8192.0]+[0.0]*28),'out_bias':0.0}

def encode_native(w):
    parts=[b'SEKIRW03']
    for name in ('ft','ft_bias','l2','l2_bias','out'):
        a=array(w[name].typecode,w[name])
        if sys.byteorder!='little':a.byteswap()
        parts.append(a.tobytes())
    parts.append(struct.pack('<f',w['out_bias']));raw=b''.join(parts)
    require(len(raw)==NATIVE_BYTES,'native03 full layout size differs');return raw

def up(x):
    require(math.isfinite(x) and x>=0,'finite positive bound required')
    raw=struct.unpack('<Q',struct.pack('<d',x))[0]
    require(raw>>63==0,'positive-sign finite bound required')
    y=struct.unpack('<d',struct.pack('<Q',raw+1))[0];require(math.isfinite(y),'bound overflow');return y

def ftz64(x):
    """Pure bit oracle for 9fc0 DAZ operands / FTZ arithmetic results."""
    raw=struct.unpack('<Q',struct.pack('<d',x))[0];magnitude=raw&0x7fffffffffffffff
    if 0<magnitude<0x0010000000000000:
        return struct.unpack('<d',struct.pack('<Q',raw&0x8000000000000000))[0]
    return x

def abs32_as64(raw):
    """Native abs(f32) -> as f64 under DAZ; parameter bits stay untouched."""
    value=frombits(raw);magnitude=raw&0x7fffffff
    return 0.0 if 0<magnitude<0x00800000 else abs(value)

def add_up(a,b):return up(ftz64(ftz64(a)+ftz64(b)))
def mul_up(a,b):return up(ftz64(ftz64(a)*ftz64(b)))
def div_up(a,b):
    require(math.isfinite(b) and b>0,'finite positive divisor required')
    return up(ftz64(ftz64(a)/ftz64(b)))

def forward_bounds(s):
    q=0.0
    for p in range(14):q=add_up(q,abs32_as64(s['out'][4+2*p]))
    require(q<=32768.0,'conservative output L1 exceeds selected budget')
    raw=add_up(1682688.0,mul_up(254.0,q));factor=div_up(1.0,1.0-2.0**-18)
    upper=add_up(div_up(mul_up(raw,factor),64.0),1.0);require(upper<899000,'output reaches ordinary/mate boundary')
    factor=div_up(1.0,1.0-2.0**-14)
    for col in range(4,L2):
        total=0.0
        for row in range(2*L1):total=add_up(total,abs32_as64(s['l2'][row*L2+col]))
        c=add_up(abs32_as64(s['l2_bias'][col]),mul_up(127.0,total))
        require(add_up(mul_up(c,factor),1e-30)<frombits(0x7f7fffff),'L2 partial sum may overflow before clamp')
    return {'q_l1_upper':q,'absolute_cp_upper':upper}

def validate_state(s,reference):
    obj(s,STATE_KEYS);require(type(s['step'])is int and s['step']==STEPS,'exact E3 optimizer step required')
    for name,count in VECTORS.items():
        for k in (name,*MOMENTS[name]):
            require(type(s[k])is list and len(s[k])==count,'whole parameter/m/v vector shape differs')
            for x in s[k]:frombits(x)
        require(all(frombits(x)>=0 for x in s[MOMENTS[name][1]]),'negative Adam variance')
    for k in ('out_bias','obias_m','obias_v'):frombits(s[k])
    require(frombits(s['obias_v'])>=0,'negative output bias variance')
    def protected(name,i,want):
        m,v=MOMENTS[name]
        require(s[name][i]==bits(want) and s[m][i]==0 and s[v][i]==0,'protected parameter/m/v changed')
    def tie(name,a,b,negative=False):
        m,v=MOMENTS[name];mask=0x80000000 if negative else 0
        require(s[name][b]==(s[name][a]^mask) and s[m][b]==(s[m][a]^mask) and s[v][b]==s[v][a],'exact signed parameter/m/v tie differs')
    for f in range(INPUT):
        for j in range(L1):
            i=f*L1+j
            if j<2:protected('ft',i,reference['ft'][i]/64.0)
            else:require(abs(frombits(s['ft'][i]))<=797.0/64.0,'aux FT797 box violated')
    for j in range(L1):protected('ft_bias',j,1.0)
    for row in range(2*L1):
        for col in range(L2):
            i=row*L2+col
            if col<4 or row%L1<2:protected('l2',i,reference['l2'][i])
    for col in range(4):
        protected('l2_bias',col,reference['l2_bias'][col]);protected('out',col,reference['out'][col])
    require(s['out_bias']==s['obias_m']==s['obias_v']==0,'protected output bias state differs')
    for donor in range(2):
        for t in range(38):
            for j in range(L1):tie('ft',(2268+donor*38+t)*L1+j,(2268+(3-donor)*38+t)*L1+j)
    for pair in range(14):
        a=4+2*pair;b=a+1
        for j in range(2,L1):
            tie('l2',j*L2+a,(L1+j)*L2+b);tie('l2',(L1+j)*L2+a,j*L2+b)
        tie('l2_bias',a,b);tie('out',a,b,True)
    return forward_bounds(s)

def nearest_from_state(s):
    ft=array('h',[0])*(INPUT*L1)
    for i,b in enumerate(s['ft']):
        x=frombits(b)*64.0
        if i%L1<2:require(x==math.trunc(x) and abs(x)<=32767,'protected material not exact native integer');q=int(x)
        else:q=round(x);require(abs(q)<=797,'nearest aux prefix domain exceeded')
        ft[i]=q
    w={'ft':ft,'ft_bias':array('h',[64])*L1}
    for name in ('l2','l2_bias','out'):w[name]=array('f',(frombits(b) for b in s[name]))
    w['out_bias']=frombits(s['out_bias']);return encode_native(w)

def checkpoint(raw,context,reference_raw,native_raw):
    v=strict_json(raw);obj(v,{'schema','version','feature_schema','encoding','resume_allowed','epochs_completed','positions_per_epoch','context','state'})
    fixed(v,{'schema':CHECKPOINT_SCHEMA,'version':1,'feature_schema':FEATURE,'encoding':'all-f32-u32-bits-le-values',
             'resume_allowed':False,'epochs_completed':3,'positions_per_epoch':N})
    require(exact(v['context'],context),'checkpoint context raw identities differ')
    ref=canonical_reference();require(reference_raw==encode_native(ref),'reference03 differs from independent seed42 allbytes')
    bounds=validate_state(v['state'],ref)
    require(type(native_raw)is bytes and native_raw==nearest_from_state(v['state']),'checkpoint nearest03 full native bytes differ')
    return bounds

def lifecycle(v,argv=None,limit=None):
    obj(v,{'command','pid','pgid','returncode','waited','reaped','timed_out','group_empty_scans','log','wall_seconds'})
    require(type(v['command'])is list and v['command'] and all(type(x)is str and x and '\0' not in x for x in v['command']),'literal argv required')
    if argv is not None:require(exact(v['command'],argv),'executed argv differs')
    require(type(v['pid'])is int and v['pid']>0 and type(v['pgid'])is int and v['pgid']==v['pid'],'new-session PID/PGID differs')
    fixed(v,{'returncode':0,'waited':True,'reaped':True,'timed_out':False,'group_empty_scans':[True,True]})
    fullref(v['log']);require(type(v['wall_seconds'])in (int,float) and math.isfinite(v['wall_seconds']) and v['wall_seconds']>=0,'finite typed numeric wall required')
    if limit is not None:require(v['wall_seconds']<limit,'whole training/proof wall budget exceeded')

def float_snapshot(v):
    obj(v,{'schema','float_policy','environment_variable','environment_value','mxcsr_raw_bits','mxcsr_control_bits','mxcsr_status_bits','mxcsr_status_mask','required_control_bits'})
    fixed(v,{'schema':'sekirei.white-view-paired-nonlinear-float-snapshot.v1','float_policy':'x86-ftz-daz',
             'environment_variable':'SEKIREI_TRAIN_FTZ_DAZ','environment_value':'1','mxcsr_control_bits':0x9fc0,'mxcsr_status_mask':0x3f,'required_control_bits':0x9fc0})
    raw=uint(v['mxcsr_raw_bits'],0xffffffff);uint(v['mxcsr_status_bits'],0x3f)
    require((raw&~0x3f)==0x9fc0 and (raw&0x3f)==v['mxcsr_status_bits'],'actual raw/control/status float bits differ')

def float_evidence(v,reader,execution_log,export_float):
    obj(v,{'schema','guard_source','epoch_log','epoch_observations','export'})
    fixed(v,{'schema':'sekirei.white-view-paired-nonlinear-float-policy-evidence.v1','epoch_log':execution_log,'export':export_float})
    reader.read(v['guard_source']);raw=reader.read(v['epoch_log'])
    observed=[];begun=None
    for line in raw.splitlines():
        if line.startswith(b'PAIRED_EPOCH_BEGIN '):
            require(begun is None,'duplicate/unclosed epoch begin');begun=strict_json(line[len(b'PAIRED_EPOCH_BEGIN '):])
            obj(begun,{'epoch','start_step','float'});e=len(observed)+1
            fixed(begun,{'epoch':e,'start_step':(e-1)*N});float_snapshot(begun['float'])
        elif line.startswith(b'PAIRED_EPOCH_COMPLETE '):
            require(begun is not None,'epoch complete before begin');rec=strict_json(line[len(b'PAIRED_EPOCH_COMPLETE '):])
            obj(rec,{'epoch','positions','end_step','float'});e=len(observed)+1
            fixed(rec,{'epoch':e,'positions':N,'end_step':e*N});float_snapshot(rec['float'])
            observed.append({'epoch':e,'start_step':begun['start_step'],'positions':N,'end_step':rec['end_step'],'before':begun['float'],'after':rec['float']});begun=None
    require(begun is None and exact(v['epoch_observations'],observed) and len(observed)==3,'float epoch evidence must be reparsed from bound actual log')
    float_snapshot(v['export'])

def fnv(raw):
    x=14695981039346656037
    for b in raw:x=((x^b)*1099511628211)&((1<<64)-1)
    return f'{x:016x}'

def export_stage(v,refs,context,inputs,native):
    obj(v,{'schema','status','epochs_completed','positions_per_epoch','global_step','resume_allowed','feature_schema','context','outputs','native_fnv1a',
           'inputs_before','inputs_after','checkpoint_all_state_bits_equal','nearest_all_bytes_equal','cleanup_verified','parent_reap_verified','core_fullrows_verified','incremental_verified','adoption_claimed','final_used','float_export'})
    fixed(v,{'schema':EXPORT_SCHEMA,'status':'three-epochs-export-readback-complete','epochs_completed':3,'positions_per_epoch':N,'global_step':STEPS,
             'resume_allowed':False,'feature_schema':FEATURE,'context':context,'outputs':{'checkpoint':refs['checkpoint'],'nearest_native03':refs['native03']},
             'native_fnv1a':fnv(native),'inputs_before':inputs,'inputs_after':inputs,'checkpoint_all_state_bits_equal':True,'nearest_all_bytes_equal':True,
             'cleanup_verified':False,'parent_reap_verified':False,'core_fullrows_verified':False,'incremental_verified':False,'adoption_claimed':False,'final_used':False})
    float_snapshot(v['float_export'])

def parent_completion(v,refs,reader,context,native):
    obj(v,{'schema','status','mode','plan','source_binding','parent_preflight','training_build','engine_build','recipe','source_head','original_inputs','reference03',
           'execution','epoch_completion','outputs','fp_evidence','inputs_before','inputs_after','source_unchanged','build_unchanged','poisoned','resume_used','shuffle_used','final_used','adoption_claimed'})
    fixed(v,{'schema':'sekirei.white-view-paired-nonlinear-training-completion.v1','status':'complete','mode':MODE,'plan':refs['plan'],
             'source_binding':refs['source_binding'],'parent_preflight':refs['parent_preflight'],'training_build':refs['training_build'],
             'engine_build':refs['engine_manifest'],'recipe':refs['recipe'],'reference03':refs['reference03'],'source_unchanged':True,'build_unchanged':True,
             'poisoned':False,'resume_used':False,'shuffle_used':False,'final_used':False,'adoption_claimed':False})
    require(type(v['source_head'])is str and len(v['source_head'])==40 and all(c in '0123456789abcdef' for c in v['source_head']),'public Git40 required')
    require(exact(v['inputs_before'],v['inputs_after']),'parent input maps changed');reader.map(v['inputs_before'])
    require(all(exact(v['inputs_before'].get(p),i) for p,i in context['source_files'].items()),'parent lost child/source closure')
    fixed(v,{'outputs':{'checkpoint':refs['checkpoint'],'native03':refs['native03'],'export_stage':refs['export_stage']},
             'epoch_completion':[{'epoch':e,'positions':N,'end_step':e*N} for e in range(1,4)]})
    lifecycle(v['execution'],limit=86400);reader.read(v['execution']['log'])
    command=v['execution']['command'];sb=reader.json(refs['source_binding'])
    require(len(command)==40 and command[:2]==[sb['training_binary']['path'],'train-paired-nonlinear'],'actual dedicated training binary/route required')
    options={}
    for flag,value in zip(command[2::2],command[3::2]):require(flag not in options,'duplicate actual command option');options[flag]=value
    wanted={'--output':refs['native03']['path'].rsplit('/',1)[0]}
    for flag,role in [('recipe','recipe'),('source-binding','source_binding'),('manifest','manifest'),('reference03','reference03'),('positions','train_positions'),('labels','train_labels')]:
        wanted['--'+flag]=refs[role]['path'];wanted['--'+flag+'-bytes']=str(refs[role]['bytes']);wanted['--'+flag+'-sha256']=refs[role]['sha256']
    require(options==wanted,'actual CLI fullrefs/output differ; initialized diagnosis is not E3 training')
    float_evidence(v['fp_evidence'],reader,v['execution']['log'],reader.json(refs['export_stage'])['float_export']);reader.refs_in(v)

def original_five(refs):
    return {name:refs[role] for name,role in [('manifest.json','manifest'),('train.positions.jsonl','train_positions'),('train.labels.jsonl','train_labels'),
          ('holdout.positions.jsonl','holdout_positions'),('holdout.labels.jsonl','holdout_labels')]}

def prior_parent_bodies(reader,parent,sb,refs):
    """Consume Root typed provenance bodies; source/membership/process execution is Root's job."""
    keys={'schema','status','mode','selected_plan','training_build','engine_build','original_inputs','reference03','prior_original_origin_proof',
          'origin_proof_policy','origin_counts','source_inputs','inputs_before','inputs_after','checks','process_evidence','source_head'}
    obj(parent,keys)
    fixed(parent,{'origin_proof_policy':'reuse-immutable-prior-full-origin-proof-plus-fresh-O-byte-and-semantic-validation'})
    checks={'prior_origin_proof_raw_and_transitive_refs_valid','fresh_original_O_schema_semantics_and_join_valid','original_pool_exclusion_split_unchanged',
        'reference03_fullbytes_reconstructed','new_training_source_build_compiler_abi_bound','fixed_white_view_engine_unchanged','inputs_before_after_equal','source_membership_equal','no_conflicting_heavy_process'}
    obj(parent['checks'],checks);fixed(parent['checks'],dict.fromkeys(checks,True))
    forbidden={refs[k]['path'] for k in ('recipe','source_binding','parent_preflight')}
    require(not forbidden.intersection(parent['source_inputs']),'predecessor DAG has future/self back reference')
    for p,i in parent['source_inputs'].items():require(exact(reader.expected.get(p),i),'parent raw closure absent/rebound')
    e=parent['process_evidence'];obj(e,{'schema','status','producer_source','process_reader','allowlist','scan_records','lock_records','source_head'})
    fixed(e,{'schema':'sekirei.white-view-paired-nonlinear-preflight-process-evidence.v1','status':'observed-clear','source_head':parent['source_head']})
    for k in ('producer_source','process_reader','allowlist'):reader.read(e[k])
    require(type(e['scan_records'])is list and len(e['scan_records'])==2,'two Root process scans required')
    for scan in e['scan_records']:
        obj(scan,{'conflicts','excluded_preexisting_services'});fixed(scan,{'conflicts':[]})
        require(type(scan['excluded_preexisting_services'])is list,'typed exclusion descriptors required')
    require(type(e['lock_records'])is list and e['lock_records'],'acquired Root lock records required');paths=set()
    for lock in e['lock_records']:
        obj(lock,{'path','exclusive','nonblocking','acquired'});absolute(lock['path']);fixed(lock,{'nonblocking':True,'acquired':True})
        require(type(lock['exclusive'])is bool and lock['path'] not in paths,'strict unique lock role');paths.add(lock['path'])
    prior=reader.json(parent['prior_original_origin_proof'])
    obj(prior,{'schema','status','producer_source','prior_source_preflight','prior_preregistration','validator_sources','original_inputs','counts','checks','inputs_before','inputs_after','fresh_full_replay_performed'})
    fixed(prior,{'schema':'sekirei.white-view-paired-nonlinear-reused-original-proof.v2','status':'validated','original_inputs':original_five(refs),
                 'counts':parent['origin_counts'],'fresh_full_replay_performed':False})
    require(prior['producer_source']['sha256']=='5afb42458f96ed5272479ec59a8d9503959ac28e31e850d417f455b02df07d51','fixed fresh origin producer SOURCE differs')
    checks={'prior_preflight_body_valid_under_original_producer_contract','original_producer_and_validator_sources_bound','all_prior_transitive_raw_references_valid',
        'fresh_original_O_schema_semantics_and_full_join_valid','original_pool_exclusion_and_frozen_split_unchanged','inputs_before_after_equal'}
    obj(prior['checks'],checks);fixed(prior['checks'],dict.fromkeys(checks,True))
    require(exact(prior['inputs_before'],prior['inputs_after']),'prior origin raw maps changed');reader.map(prior['inputs_before']);reader.map(prior['validator_sources']);reader.refs_in(prior,parent['source_inputs'])
    for filemap in (prior['inputs_before'],prior['validator_sources']):
        require(all(exact(parent['source_inputs'].get(p),i) for p,i in filemap.items()),'parent lost inherited prior raw/validator map')
    for name,want in [('white_view_fit_contract.py','7ae7d4cddecd09c143de8bc97ea0e1851f25e33760b992cb87b6b34bea7ad649'),
                      ('functional_anchor.py','e704e153e6752abe1f3a6b686ac5e9eddea688216cd28626fffb598beef8130b')]:
        found=[v for p,v in prior['validator_sources'].items() if p.rsplit('/',1)[-1]==name]
        require(found and all(v['sha256']==want for v in found),'unchanged prior semantic validator source missing/changed')
    for doc in ('prior_source_preflight','prior_preregistration'):reader.json(prior[doc])

def validate_recipe(v,refs):
    keys={'schema','mode','feature_schema','seed','epochs','train_count','objective','optimizer','learning_rate_f32_bits','head_init_width_f32_bits',
          'head_bias_init_f32_bits','output_native_l1_budget_f32_bits','shuffle_seed','teacher_identity','manifest','reference03','positions','labels','source_binding','resume_allowed','ft_saved_q_max','ft_bias_q'}
    obj(v,keys);fixed(v,{'schema':'sekirei.white-view-paired-nonlinear-recipe.v1','mode':MODE,'feature_schema':FEATURE,'seed':42,'epochs':3,'train_count':N,
          'objective':'absolute-cp-mse','optimizer':'fresh-adam-tied-masters','shuffle_seed':None,'teacher_identity':TEACHER,'manifest':refs['manifest'],
          'reference03':refs['reference03'],'positions':refs['train_positions'],'labels':refs['train_labels'],'source_binding':refs['source_binding'],
          'resume_allowed':False,'ft_saved_q_max':797,'ft_bias_q':64,**{k+'_f32_bits':x for k,x in probe.RECIPE_BITS.items()}})

def training_commands(commands):
    """Same deterministic Cargo command requirements as frozen parent Rust v3."""
    require(type(commands)is list and commands,'nonempty actual build commands required');seen=set()
    for command in commands:
        require(type(command)is list and command and all(type(x)is str and x and '\0' not in x for x in command),'literal Cargo argv required')
        require(command[0].rsplit('/',1)[-1]=='cargo' and {'--release','--offline','--locked'}<=set(command),'actual cargo release/offline/locked required')
        pairs=list(zip(command,command[1:]))
        jobs=any(a in ('--jobs','-j') and b=='2' for a,b in pairs) or any(x in ('--jobs=2','-j2') for x in command)
        features=[b for a,b in pairs if a=='--features']+[x[len('--features='):] for x in command if x.startswith('--features=')]
        require(jobs and any('nnue_white_view_aux_tied' in x.split(',') for x in features),'actual jobs2/new white feature required')
        names=set(command)&{'test','build'};require(len(names)==1,'actual test or build Cargo route required');seen|=names
    require(seen=={'test','build'},'both actual test and build commands required')

def rows(raw):
    require(type(raw)is bytes and raw and raw.endswith(b'\n') and b'\r' not in raw,'canonical nonempty LF JSONL required')
    return [strict_json(b) for b in raw[:-1].split(b'\n')]

def material(sfen):
    require(type(sfen)is str,'SFEN string required');parts=sfen.split(' ')
    require(len(parts)==4 and parts[1]in ('b','w') and parts[3].isascii() and parts[3].isdigit() and int(parts[3])>0,'canonical SFEN fields')
    ranks=parts[0].split('/');require(len(ranks)==9,'nine SFEN ranks');value=0;count=0
    for rank in ranks:
        cells=0;promoted=False
        for c in rank:
            if c in '123456789':require(not promoted,'promotion before gap');cells+=int(c);continue
            if c=='+':require(not promoted,'duplicate promotion');promoted=True;continue
            kind=('+' if promoted else '')+c.upper();require(kind in KINDS,'invalid SFEN piece');promoted=False;cells+=1;count+=1
            value+=(1 if c.isupper() else -1)*VALUES[KINDS.index(kind)]
        require(cells==9 and not promoted,'SFEN rank shape')
    if parts[2]!='-':
        num=''
        for c in parts[2]:
            if c.isascii() and c.isdigit():num+=c;continue
            require(c.upper()in KINDS[:7],'invalid hand kind');n=int(num) if num else 1;require(n>0,'positive hand count');num='';count+=n
            value+=(1 if c.isupper() else -1)*n*VALUES[KINDS.index(c.upper())]
        require(not num,'dangling hand count')
    require(count<=40,'legal inventory/prefix domain exceeded');return value if parts[1]=='b' else -value

def original_rows(reader,refs):
    require(refs['manifest']['sha256']==MANIFEST_SHA,'fixed original O manifest differs')
    m=reader.json(refs['manifest']);fixed(m,{'schema_version':1,'teacher_identity':TEACHER,'positions':{'train':N,'holdout':H}})
    result={}
    for split,count in [('train',N),('holdout',H)]:
        p=rows(reader.read(refs[split+'_positions']));labels=rows(reader.read(refs[split+'_labels']));require(len(p)==len(labels)==count,'full original split counts differ')
        sfens=[];cache={}
        for row in labels:
            obj(row,{'sfen','score_cp','label_depth','teacher_identity'});fixed(row,{'label_depth':0,'teacher_identity':TEACHER})
            require(type(row['score_cp'])is int and abs(row['score_cp'])<30000,'teacher ordinary integer required');material(row['sfen'])
            require(row['sfen'] not in cache,'duplicate cache SFEN');cache[row['sfen']]=row['score_cp']
        for row in p:
            obj(row,{'schema_version','sfen','source','tags'});fixed(row,{'schema_version':1});material(row['sfen']);sfens.append(row['sfen'])
        require(len(set(sfens))==count and set(sfens)==set(cache),'complete original position/cache join differs')
        result[split]=sfens
    f=reader.json(refs['fixtures']);require(type(f)is list and len(f)==15,'public15 fixtures required')
    result['fixtures']=[v['sfen'] for v in f]
    for v in f:require(type(v['expected_cp_stm'])is int and material(v['sfen'])==v['expected_cp_stm'],'public fixture M differs')
    return result

def core_stderr(raw,count,bounds=None):
    text=raw.decode('utf-8');lines=text.splitlines()
    require(len(lines)==3,'three dedicated core stderr records required')
    match=re.fullmatch(r'float_subnormal_policy=x86-ftz-daz; mxcsr_control=0x9fc0; mxcsr_status=0x([0-9a-f]+)',lines[0])
    require(match is not None and int(match[1],16)<=0x3f,'dedicated core FP stderr differs')
    prefix='nonlinear_native_structure=verified; mode='+MODE+'; epochs=3; q_l1_upper='
    require(lines[1].startswith(prefix),'native structural stderr missing')
    values=lines[1][len(prefix):].split('; absolute_cp_upper=');require(len(values)==2,'native bound stderr syntax')
    q,cp=(float(x) for x in values);require(math.isfinite(q) and 0<=q<=32768 and math.isfinite(cp) and 0<cp<899000,'finite selected native bound stderr required')
    if bounds is not None:require(q==bounds['q_l1_upper'] and cp==bounds['absolute_cp_upper'],'actual native bounds differ from independent state reconstruction')
    require(lines[2]=='complete count='+str(count)+'; nonlinear absolute STM; both stored-native float bridges, FT prefixes, every L2/output product/add finite before clamp','core completion stderr differs')

def proof_compilation(value,kind,reader,refs,body):
    """Embedded producer proposal; actual compilation still belongs to Root."""
    obj(value,{'engine_manifest','engine_identity','core_rlib','release_dependencies_before','release_dependencies_after','sources','compiler',
               'binary','command','execution','inputs_before','inputs_after'})
    fixed(value,{'engine_manifest':refs['engine_manifest'],'engine_identity':refs['engine_identity'],'binary':body['probe']})
    manifest=reader.json(refs['engine_manifest']);identity=reader.json(refs['engine_identity'])
    fixed(value,{'core_rlib':manifest['core_link']['rlib'],'release_dependencies_before':manifest['release_dependencies_after_probes'],
                'release_dependencies_after':manifest['release_dependencies_after_probes']})
    obj(value['sources'],{'probe','native_contract'});fixed(value['sources'],{'probe':body['probe_source']})
    require(value['sources']['native_contract']['sha256']==PROOF_SOURCE_SHA['native_contract'],'fixed nonlinear native contract source changed')
    reader.refs_in(value);reader.map(value['release_dependencies_before'])
    source=reader.read(value['sources']['probe']);marker=b'const PROTOTYPE_ONLY:bool=false;'
    require(source.count(marker)==1 and digest(source.replace(marker,b'const PROTOTYPE_ONLY:bool=true;',1))['sha256']==PROOF_SOURCE_SHA[kind],
            'enabled proof source must be frozen prototype plus exactly the one false barrier assignment')
    require(value['sources']['native_contract']['path']==value['sources']['probe']['path'].rsplit('/',1)[0]+'/paired_nonlinear_native_contract.rs','actual Rust adjacent module source differs')
    fullref(value['compiler']);require(exact(identity['compiler_files'].get(value['compiler']['path']),{k:value['compiler'][k] for k in ('bytes','sha256')}),'actual compiler not in fixed new engine identity')
    deps=manifest['core_link']['rlib']['path'].rsplit('/',1)[0]
    command=[value['compiler']['path'],'--edition=2024','-C','target-cpu=x86-64-v3','-C','opt-level=3','-C','panic=abort',
        '--extern','sekirei_core='+value['core_rlib']['path'],'-L','dependency='+deps,value['sources']['probe']['path'],'-o',body['probe']['path']]
    fixed(value,{'command':command});lifecycle(value['execution'],command);reader.read(value['execution']['log'])
    require(exact(value['inputs_before'],value['inputs_after']),'proof compilation inputs changed');reader.map(value['inputs_before'])
    for r in (value['engine_manifest'],value['engine_identity'],value['core_rlib'],value['compiler'],*value['sources'].values()):
        require(exact(value['inputs_before'].get(r['path']),{k:r[k] for k in ('bytes','sha256')}),'compile lost source/compiler/rlib predecessor closure')
    for p,i in value['release_dependencies_before'].items():require(exact(value['inputs_before'].get(p),i),'compile lost full release dependency closure')

def validate_proofs(reader,refs,positions,training,context,bounds):
    core=reader.json(refs['core']);inc=reader.json(refs['incremental'])
    common={'status':'complete','mode':MODE,'plan':refs['plan'],'training_completion':refs['training_completion'],'native03':refs['native03'],
            'checkpoint':refs['checkpoint'],'reference03':refs['reference03'],'engine_manifest':refs['engine_manifest'],'engine_identity':refs['engine_identity'],
            'training_build':refs['training_build'],'recipe_bits':probe.RECIPE_BITS,'adam_used':True,'epochs':3,'selected_epoch':3,'float_policy':'x86-ftz-daz',
            'final_used':False,'adoption_claimed':False}
    basekeys={*common,'schema','worker','probe','probe_source','compilation','inputs_before','inputs_after','cleanup_verified'}
    obj(core,basekeys|{'results','counts','total_rows','pair_argument_order'});fixed(core,{**common,'schema':'sekirei.white-view-paired-nonlinear-candidate-core-proof.v1',
                'counts':probe.COUNTS,'total_rows':118591,'pair_argument_order':['candidate03','reference03'],'cleanup_verified':True})
    require(exact(core['inputs_before'],core['inputs_after']),'core source/inputs changed');reader.map(core['inputs_before']);reader.refs_in(core)
    proof_compilation(core['compilation'],'core',reader,refs,core)
    obj(core['results'],{'train','holdout','fixtures'});summaries={}
    for split,sfens in positions.items():
        r=core['results'][split];obj(r,{'count','positions','stdin','stdout','stderr','execution','maximum_material_difference_cp','maximum_stored_float_core_bridge_cp'})
        fixed(r,{'count':len(sfens),'positions':refs['fixtures' if split=='fixtures' else split+'_positions']})
        inp=reader.read(r['stdin']);out=reader.read(r['stdout']);err=reader.read(r['stderr']);core_stderr(err,len(sfens),bounds)
        require(inp==''.join(s+'\n' for s in sfens).encode(),'core stdin order differs from original O/public15')
        lifecycle(r['execution'],[core['probe']['path'],refs['native03']['path'],refs['reference03']['path'],'x86-ftz-daz'],600)
        lines=out.splitlines();require(len(lines)==len(sfens),'core missing/extra output rows');maximum=0;bridge=0.0
        for index,(raw,sfen) in enumerate(zip(lines,sfens)):
            row=probe.core_row(raw,index,material(sfen));maximum=max(maximum,abs(row['native_core_cp']-row['material_cp']))
            bridge=max(bridge,abs(row['native_quantized_float_cp']-row['native_core_cp']))
            require(abs(row['native_quantized_float_cp'])<=bounds['absolute_cp_upper'],'observed candidate exceeds selected bound')
        fixed(r,{'maximum_material_difference_cp':maximum,'maximum_stored_float_core_bridge_cp':bridge})
        summaries[split]={'count':len(sfens),'maximum_material_difference_cp':maximum,'maximum_stored_float_core_bridge_cp':bridge}
    obj(inc,basekeys|{'execution','stdout','stderr','stdin','fixture_tsv','result'});fixed(inc,{**common,'schema':'sekirei.white-view-paired-nonlinear-candidate-incremental-proof.v1','cleanup_verified':True})
    require(exact(inc['inputs_before'],inc['inputs_after']),'incremental source/inputs changed');reader.map(inc['inputs_before']);reader.refs_in(inc)
    proof_compilation(inc['compilation'],'incremental',reader,refs,inc)
    lifecycle(inc['execution'],[inc['probe']['path'],refs['native03']['path'],refs['reference03']['path'],inc['fixture_tsv']['path']],1200)
    require(reader.read(inc['stderr'])==reader.read(inc['stdin'])==b'','incremental empty stdin/stderr differs')
    fixtures=reader.json(refs['fixtures']);tsv=''.join(str(v['expected_cp_stm'])+'\t'+v['sfen']+'\n' for v in fixtures).encode()
    require(reader.read(inc['fixture_tsv'])==tsv,'ordered public15 incremental TSV differs')
    result=probe.incremental_result(reader.json(inc['stdout']));require(exact(result,inc['result']),'incremental aggregate differs from raw reparse')
    require(result['native_q_l1_upper']<=bounds['q_l1_upper'] and result['native_absolute_cp_upper']<=bounds['absolute_cp_upper'],'incremental actual native bounds differ')
    return summaries,result

def reconstruct_gate(raw_inputs,binding_raw,expected_binding_sha256):
    """Pure proposed Root outer contract. No actual Reader/lock/write implementation."""
    sha(expected_binding_sha256);require(digest(binding_raw)['sha256']==expected_binding_sha256,'external binding SHA differs before parse')
    b=strict_json(binding_raw);obj(b,{'schema','refs','inputs','source_helpers','source_head'})
    fixed(b,{'schema':'sekirei.white-view-paired-nonlinear-candidate-gate-binding.v1'});obj(b['refs'],ROLES)
    reader=Bytes(raw_inputs,b['inputs']);refs=b['refs']
    for r in refs.values():reader.read(r)
    reader.map(b['source_helpers'])
    require(refs['plan']['bytes']==7360 and refs['plan']['sha256']==PLAN_SHA,'fixed selected model plan raw identity differs')
    plan=reader.json(refs['plan']);fixed(plan,{'schema':'sekirei.white-view-paired-nonlinear-selected-plan.v1','mode':MODE})
    probe.selected_recipe_bits({k:plan['training']['scalars'][k]['f32_bits'] for k in probe.RECIPE_BITS})
    require(exact(plan['original_inputs'],original_five(refs)),'selected plan original five differ')
    recipe=reader.json(refs['recipe']);validate_recipe(recipe,refs)
    sb=reader.json(refs['source_binding']);parent=reader.json(refs['parent_preflight']);tb=reader.json(refs['training_build'])
    obj(sb,{'schema','status','mode','selected_plan','parent_preflight','training_build','training_binary','training_source_files','compiler_files','engine_build','original_inputs','reference03','float_policy'})
    fixed(sb,{'schema':'sekirei.white-view-paired-nonlinear-source-binding.v2','status':'ready','mode':MODE,'selected_plan':refs['plan'],'parent_preflight':refs['parent_preflight'],
             'training_build':refs['training_build'],'engine_build':refs['engine_manifest'],'original_inputs':original_five(refs),'reference03':refs['reference03'],'float_policy':'x86-ftz-daz'})
    reader.refs_in(sb);reader.map(sb['training_source_files']);reader.map(sb['compiler_files'])
    fixed(parent,{'schema':'sekirei.white-view-paired-nonlinear-parent-preflight.v2','status':'complete','mode':MODE,'selected_plan':refs['plan'],'training_build':refs['training_build'],
                  'engine_build':refs['engine_manifest'],'original_inputs':original_five(refs),'reference03':refs['reference03'],'source_head':b['source_head'],
                  'origin_counts':{'train':N,'holdout':H,'prior_raw_games':1000,'prior_replay_source_files':1042}})
    require(exact(parent['source_inputs'],parent['inputs_before']) and exact(parent['source_inputs'],parent['inputs_after']),'parent source maps changed')
    reader.map(parent['source_inputs']);reader.refs_in(parent,parent['source_inputs'])
    prior_parent_bodies(reader,parent,sb,refs)
    obj(tb,{'schema','status','mode','producer_source','source_root','upstream_commit','source_head','source_files','compiler_files','dependency_files','engine_build','training_binary',
            'commands','environment','tests','process_outcomes','inputs_before','inputs_after','scalar_power_cache_used'})
    fixed(tb,{'schema':'sekirei.white-view-paired-nonlinear-training-build.v1','status':'complete','mode':MODE,'source_head':b['source_head'],'scalar_power_cache_used':False,
              'source_files':sb['training_source_files'],'compiler_files':sb['compiler_files'],'engine_build':refs['engine_manifest'],'training_binary':sb['training_binary'],
              'upstream_commit':'f09c13026e9485a19b4ba41b91ed2e1bbdf5e1c9'})
    require(exact(tb['inputs_before'],tb['inputs_after']),'training build input map changed');reader.map(tb['inputs_before']);reader.refs_in(tb,parent['source_inputs'])
    require(all(exact(parent['source_inputs'].get(p),i) for p,i in tb['inputs_before'].items()),'parent lost full training build predecessor map')
    obj(tb['environment'],{'CARGO_TARGET_DIR','RUSTC','RUSTFLAGS','CARGO_BUILD_JOBS'});fixed(tb['environment'],{'RUSTFLAGS':'-C target-cpu=x86-64-v3','CARGO_BUILD_JOBS':'2'})
    for p in ('CARGO_TARGET_DIR','RUSTC'):absolute(tb['environment'][p])
    require(tb['environment']['RUSTC'] in sb['compiler_files'],'actual rustc missing from compiler map')
    require(type(tb['commands'])is list and len(tb['commands'])==len(tb['process_outcomes']) and tb['commands'],'actual build commands/outcomes required')
    training_commands(tb['commands'])
    for cmd,outcome in zip(tb['commands'],tb['process_outcomes']):lifecycle(outcome,cmd);reader.read(outcome['log'])
    reader.map(tb['dependency_files'])
    for filemap in (tb['source_files'],tb['compiler_files'],tb['dependency_files']):
        require(all(exact(tb['inputs_before'].get(p),i) for p,i in filemap.items()),'build predecessor source/compiler/dependency closure absent')
    obj(tb['tests'],{'log','passed','failed','ignored','required_test_names'});fixed(tb['tests'],{'failed':0})
    require(uint(tb['tests']['passed'])>0,'positive actual Rust tests required');uint(tb['tests']['ignored'])
    names=tb['tests']['required_test_names'];require(type(names)is list and names and len(set(names))==len(names) and all(type(x)is str and x for x in names),'unique actual test names required')
    testlog=reader.read(tb['tests']['log']).decode('utf-8')
    passed={line.strip()[len('test '):-len(' ... ok')] for line in testlog.splitlines() if line.strip().startswith('test ') and line.strip().endswith(' ... ok')}
    for name in names:require(name in passed,'named Rust test not observed passing')
    for needles in [('reader',),('sha',),('unique',),('native',),('update','adapter'),('ftz','float_policy','mxcsr')]:
        require(any(any(n in name.lower() for n in needles) for name in names),'actual typed Reader/SHA/JSON/native/update/FP test category missing')
    identity=reader.json(refs['engine_identity']);manifest=reader.json(refs['engine_manifest'])
    build.validate_identity(identity);build.validate_manifest(manifest,identity,refs['engine_identity'])
    reader.map(build.immutable_inputs_for_runtime({'manifest':manifest,'identity':identity},refs['engine_manifest']))
    context={'recipe':refs['recipe'],'source_binding':refs['source_binding'],'manifest':refs['manifest'],'reference03':refs['reference03'],
             'positions':refs['train_positions'],'labels':refs['train_labels'],'source_files':reader.json(refs['checkpoint'])['context']['source_files']}
    reader.map(context['source_files'])
    for r in context.values():
        if type(r)is dict and set(r)=={'path','bytes','sha256'}:require(exact(context['source_files'].get(r['path']),{k:r[k] for k in ('bytes','sha256')}),'checkpoint context missing mandatory input')
    require(all(exact(context['source_files'].get(p),i) for p,i in parent['source_inputs'].items()),'checkpoint lost original parent raw closure')
    native=reader.read(refs['native03']);bounds=checkpoint(reader.read(refs['checkpoint']),context,reader.read(refs['reference03']),native)
    export_stage(reader.json(refs['export_stage']),refs,context,context['source_files'],native)
    completed=reader.json(refs['training_completion']);parent_completion(completed,refs,reader,context,native)
    fixed(completed,{'source_head':b['source_head'],'original_inputs':original_five(refs)})
    meta=reader.json(refs['metadata']);obj(meta,{'schema','mode','feature_schema','native_magic','native','native_fnv1a','checkpoint','training_completion','recipe','source_binding','selected_epoch','epochs','adam_used','global_step','resume_used','shuffle_used','scalar_power_cache_used'})
    fixed(meta,{'schema':'sekirei.white-view-paired-nonlinear-native-sidecar.v1','mode':MODE,'feature_schema':FEATURE,'native_magic':'SEKIRW03','native':refs['native03'],'native_fnv1a':fnv(native),
                'checkpoint':refs['checkpoint'],'training_completion':refs['training_completion'],'recipe':refs['recipe'],'source_binding':refs['source_binding'],'selected_epoch':3,'epochs':3,'adam_used':True,
                'global_step':STEPS,'resume_used':False,'shuffle_used':False,'scalar_power_cache_used':False})
    positions=original_rows(reader,refs);core,inc=validate_proofs(reader,refs,positions,completed,context,bounds)
    return {'schema':'sekirei.white-view-paired-nonlinear-model-technical-gate.v1','status':'complete','mode':MODE,'plan':refs['plan'],'source_head':b['source_head'],
        'model':{'kind':'nnue',**refs['native03']},'metadata':refs['metadata'],'reference_weight':refs['reference03'],'training_build':refs['training_build'],'engine_manifest':refs['engine_manifest'],'engine_identity':refs['engine_identity'],
        'proofs':{k:refs[k] for k in ('training_completion','checkpoint','export_stage','core','incremental')},'recipe_bits':probe.RECIPE_BITS,'adam_used':True,'epochs':3,'selected_epoch':3,'global_step':STEPS,
        'native_magic':'SEKIRW03','feature_schema':FEATURE,'state_encoding':'all-f32-u32-bits-le-values','fullstate_fields':19,'core_total_rows':118591,'incremental_observations':8185,
        'checkpoint_native_reexport_allbytes_equal':True,'protected_material_params_m_v_unchanged':True,'full_hand_and_head_state_ties_verified':True,'aux_ft_saved_q_limit':797,'max_prefix_features':41,
        'native_functional_bounds':bounds,'core_results':core,'incremental_result':inc,'float_policy':'x86-ftz-daz','inputs_before':dict(b['inputs']),'inputs_after':dict(b['inputs']),
        'raw_training_to_nearest_lt_1p001_claimed':False,'linear100cp_residual_cap_claimed':False,'scalar_power_cache_used':False,'search_benchmark_verified':False,'final_used':False,'adoption_verified':False}

def actual_entry(*_args,**_kwargs):runtime_guard()
