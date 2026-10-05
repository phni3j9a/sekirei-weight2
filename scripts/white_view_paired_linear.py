"""SOURCE ONLY memory contracts for white-view flat auxiliary tied FT.

The original stock seed42 initializer is a transform input, never a new ABI
model. New native03 is independently parsed and rebuilt; it is never retagged
to native01 and accepted by an old loader. Existing paired/native/numeric APIs
are unchanged. Runtime source files, new core binary, legal replay, actual
112681-row fit and adoption proof belong to future externally pinned drivers.
"""
from array import array
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import re
import struct
import sys

import material_init as material
import paired_linear as paired

PROTOTYPE_ONLY=True
MODE='white-view-paired-linear-constrained-ridge1-l1-39p5-v1'
PLAN_SHA256='1f274b187a96b16674814c51f77cb402ca8a80e6a7e5d738a5caf9354127e836'
FEATURE_SCHEMA='flat_white_view_aux_tied_v1'
COMPILE_FEATURE='nnue_white_view_aux_tied'
MAGIC=b'SEKIRW03'
OLD_MAGIC=b'SEKIRW01'
INPUT,L1,L2=2420,256,32
BOARD_INPUT,HAND_THRESHOLDS=2268,38
CHANNELS=tuple(range(2,256))
FT_OFFSET=8
FT_END=8+INPUT*L1*2
L2_OFFSET=FT_END+L1*2
L2_BIAS_OFFSET=L2_OFFSET+2*L1*L2*4
OUT_OFFSET=L2_BIAS_OFFSET+L2*4
OUT_BIAS_OFFSET=OUT_OFFSET+L2*4
WEIGHT_BYTES=OUT_BIAS_OFFSET+4
STOCK_INITIALIZER_SHA256='bbe9fbea4c943d69d605190f9ef8c6e9c7a4b9aa7c3a6be970d93e3405334e40'
ORIGINAL_TEACHER='external:suisho11beta-1m-pack:376d4ef6e503d2ebe687f99e873103845b6eddc0c04d0d08e9b9c785ea061b8d'
NUMERIC_PRODUCER_SHA256='ca65fc08248cb4327b1fb0ec733a65af303673010217e952ea49c878498a3b68'
SOURCE_NAMES={'crates/sekirei-core/src/nnue.rs','crates/sekirei-core/Cargo.toml','crates/sekirei-usi/Cargo.toml'}


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(data):
    require(type(data) is bytes,'immutable bytes required')
    return hashlib.sha256(data).hexdigest()


def strict_sha(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}',value),'strict lowercase SHA256 required');return value


def integer(value,low,high,message):
    require(type(value) is int and low<=value<=high,message);return value


def exact(a,b):
    if type(a) is not type(b):return False
    if type(a) is dict:return a.keys()==b.keys() and all(exact(a[k],v) for k,v in b.items())
    if type(a) in (list,tuple):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b


def canonical_json(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')


def validate_binding(binding):
    """Pure typed declaration, NOT verification of runtime files or binary."""
    expected={'schema':'sekirei.white-view-paired-linear-feature-binding.v1','mode':MODE,
        'plan_sha256':PLAN_SHA256,'feature_schema':FEATURE_SCHEMA,'compile_feature':COMPILE_FEATURE,
        'native_magic':MAGIC.decode(),'dimensions':{'input':INPUT,'l1':L1,'l2':L2},
        'stock_initializer_sha256':STOCK_INITIALIZER_SHA256}
    names=set(expected)|{'feature_source_sha256','core_binary_sha256','helper_source_sha256'}
    require(type(binding) is dict and set(binding)==names,'dedicated white feature binding fields required')
    require(all(exact(binding.get(k),v) for k,v in expected.items()),'white feature ABI/mode/plan/initializer differs')
    sources=binding['feature_source_sha256']
    require(type(sources) is dict and set(sources)==SOURCE_NAMES,'new explicit three-file feature source bindings required')
    for digest in sources.values():strict_sha(digest)
    strict_sha(binding['core_binary_sha256']);strict_sha(binding['helper_source_sha256'])
    return json.loads(canonical_json(binding))


def shape(data,magic):
    require(type(data) is bytes and len(data)==WEIGHT_BYTES and data[:8]==magic,'exact immutable native shape and ABI magic required')


def validate_original_initializer(original01):
    shape(original01,OLD_MAGIC)
    require(sha(original01)==STOCK_INITIALIZER_SHA256,'exact original stock seed42 initializer required')
    require(original01==paired.reference(),'original initializer full public seed42 reconstruction differs')
    return original01


def ft_row(feature):return FT_OFFSET+feature*L1*2


def _tie_aux_and_tag(native01):
    """Private transform after original or old-head producer validation."""
    shape(native01,OLD_MAGIC)
    result=bytearray(native01);result[:8]=MAGIC
    for donor,destination in ((0,3),(1,2)):
        for threshold in range(HAND_THRESHOLDS):
            start=ft_row(BOARD_INPUT+donor*HAND_THRESHOLDS+threshold)+4
            target=ft_row(BOARD_INPUT+destination*HAND_THRESHOLDS+threshold)+4
            result[target:target+254*2]=native01[start:start+254*2]
    return bytes(result)


def protected_bytes(native):
    """All FT material cols0/1 and four material L2/output units, in order."""
    pieces=[native[ft_row(feature):ft_row(feature)+4] for feature in range(INPUT)]
    for row in range(2*L1):
        start=L2_OFFSET+row*L2*4;pieces.append(native[start:start+16])
    pieces.extend((native[FT_END:L2_OFFSET],native[L2_BIAS_OFFSET:L2_BIAS_OFFSET+16],
        native[OUT_OFFSET:OUT_OFFSET+16],native[OUT_BIAS_OFFSET:]))
    return b''.join(pieces)


def validate_hand_ties(native03):
    shape(native03,MAGIC)
    for donor,destination in ((0,3),(1,2)):
        for threshold in range(HAND_THRESHOLDS):
            start=ft_row(BOARD_INPUT+donor*HAND_THRESHOLDS+threshold)
            target=ft_row(BOARD_INPUT+destination*HAND_THRESHOLDS+threshold)
            require(native03[start:start+L1*2]==native03[target:target+L1*2],'new loader requires complete i16 hand rows0=3/1=2')
    return True


def transform_initializer(original01,*,binding):
    validate_binding(binding);validate_original_initializer(original01)
    result=_tie_aux_and_tag(original01);validate_hand_ties(result)
    require(result[FT_OFFSET:ft_row(BOARD_INPUT)]==original01[FT_OFFSET:ft_row(BOARD_INPUT)],'board FT changed')
    require(result[FT_END:]==original01[FT_END:],'FT bias or downstream initializer bytes changed')
    require(protected_bytes(result)==protected_bytes(original01),'protected material bytes changed')
    return result


def serialize_coefficients(coefficients_f32,original01,*,binding):
    """Pure new03 head serializer; old01 is only the fixed transform input."""
    validate_binding(binding);validate_original_initializer(original01)
    paired.validate_coefficients(coefficients_f32)
    old_head=paired.serialize_coefficients(coefficients_f32,initializer=original01)
    result=_tie_aux_and_tag(old_head)
    validate_hand_ties(result)
    require(protected_bytes(result)==protected_bytes(original01),'paired white head changed protected material bytes')
    return result


def validate_native(native03,coefficients_f32,original01,*,binding):
    binding=validate_binding(binding);shape(native03,MAGIC)
    validate_hand_ties(native03)
    norm=paired.validate_coefficients(coefficients_f32)
    require(native03==serialize_coefficients(coefficients_f32,original01,binding=binding),'new03 full canonical reconstruction differs')
    excursion=Fraction(5,4)*norm
    return {'schema':'sekirei.white-view-paired-linear-pure-native-validation.v1','status':'pure_native_bytes_pass',
        'mode':MODE,'plan_sha256':PLAN_SHA256,'feature_schema':FEATURE_SCHEMA,'native_magic':MAGIC.decode(),
        'weight_sha256':sha(native03),'weight_bytes':len(native03),'stock_initializer_sha256':sha(original01),
        'coefficient_bits_sha256':sha(coefficients_f32),'binding':binding,
        'saved_f32_coefficient_l1':{'numerator':norm.numerator,'denominator':norm.denominator},
        'coefficient_l1_cap':{'numerator':79,'denominator':2},'board_ft_bytes_preserved':True,
        'ft_bias_bytes_preserved':True,'material_bytes_preserved':True,'hand_full_row_ties_verified':True,
        'canonical_new_abi_rebuild_equal':True,'signed_mirror_bits_verified':True,'adam_used':False,'epochs':0,
        'source_conditional_real_residual_bound_cp':{'numerator':(2*excursion).numerator,'denominator':(2*excursion).denominator},
        'actual_source_files_verified':False,'actual_core_verified':False,'native_bit_exact_covariance_verified':False,
        'actual_design_replay_verified':False,'actual_fitter_verified':False,'adoption_verified':False}


def decode_native03(data):
    """Independent field parser; no old loader/header-retag acceptance path."""
    shape(data,MAGIC);validate_hand_ties(data)
    result={};cursor=8
    for key,code,count in (('ft','h',INPUT*L1),('ft_bias','h',L1),('l2','f',2*L1*L2),('l2_bias','f',L2),('out','f',L2)):
        size=2 if code=='h' else 4
        value=array(code);value.frombytes(data[cursor:cursor+size*count])
        if sys.byteorder!='little':value.byteswap()
        result[key]=value;cursor+=size*count
    result['out_bias']=struct.unpack_from('<f',data,cursor)[0]
    require(all(math.isfinite(v) for key in ('l2','l2_bias','out') for v in result[key]) and math.isfinite(result['out_bias']),'finite new-native floats required')
    return result


def feature_index(square,kind,color,perspective):
    integer(square,0,80,'strict square0..80 required');integer(kind,0,13,'strict piece kind0..13 required')
    integer(color,0,1,'strict color0/1 required');integer(perspective,0,1,'strict perspective0/1 required')
    view=square if perspective==0 else 80-square
    return view*28+kind*2+int(color!=perspective)


def hand_feature_index(kind,color,perspective,count):
    integer(kind,0,6,'strict hand kind0..6 required');integer(color,0,1,'strict hand color0/1 required')
    integer(perspective,0,1,'strict perspective0/1 required')
    integer(count,1,material.HAND_MAX[kind],'strict bounded hand threshold required')
    return BOARD_INPUT+(2*color+perspective)*HAND_THRESHOLDS+material.HAND_OFFSETS[kind]+count-1


def validate_position(position):
    require(type(position) is dict and set(position)=={'pieces','hands','stm'},'typed position fields required')
    integer(position['stm'],0,1,'strict STM required')
    pieces=position['pieces'];hands=position['hands']
    require(type(pieces) in (list,tuple) and type(hands) in (list,tuple) and len(hands)==2,'piece/hand sequences required')
    squares=set();totals=[0]*7;kings=[0,0]
    for piece in pieces:
        require(type(piece) in (list,tuple) and len(piece)==3,'strict piece triple required')
        square,kind,color=piece
        feature_index(square,kind,color,0);require(square not in squares,'duplicate board square');squares.add(square)
        if kind==7:kings[color]+=1
        else:totals[material.BASE[kind]]+=1
    for hand in hands:
        require(type(hand) in (list,tuple) and len(hand)==7,'exact seven hand counts required')
        for kind,count in enumerate(hand):totals[kind]+=integer(count,0,material.HAND_MAX[kind],'strict hand inventory count required')
    require(kings==[1,1] and all(total<=cap for total,cap in zip(totals,material.HAND_MAX)),'standard inventory and both kings required')
    return position


def active_features(position,perspective):
    validate_position(position);integer(perspective,0,1,'strict perspective required')
    # Stock accumulation walks PHYSICAL square order; only the feature index
    # changes. Do not sort by rotated q and silently alter accumulation order.
    result=[feature_index(square,kind,color,perspective) for square,kind,color in sorted(position['pieces'])]
    for color in range(2):
        for kind,count in enumerate(position['hands'][color]):
            result.extend(hand_feature_index(kind,color,perspective,n) for n in range(1,count+1))
    require(2<=len(result)<=40 and len(set(result))==len(result),'standard40 unique active features required')
    return tuple(result)


def rotated_color_swap(position):
    validate_position(position)
    return {'pieces':sorted((80-square,kind,1-color) for square,kind,color in position['pieces']),
        'hands':[list(position['hands'][1]),list(position['hands'][0])],'stm':1-position['stm']}


def _design_row(position,initializer03):
    """Exact integer Z/M from caller-bound transformed initializer, no fit."""
    validate_position(position);shape(initializer03,MAGIC);validate_hand_ties(initializer03)
    us=active_features(position,position['stm']);them=active_features(position,1-position['stm'])
    require(len(us)==len(them),'perspective feature count differs')
    raw=[]
    for features in (us,them):
        values=[64]*254
        for feature in features:
            weights=struct.unpack_from('<254h',initializer03,ft_row(feature)+4)
            require(all(value in (-1,1) for value in weights),'auxiliary FT must be +/-1')
            for index,value in enumerate(weights):values[index]+=value
        require(all(24<=value<=104 for value in values),'aux raw24..104 required');raw.append(tuple(values))
    bias=struct.unpack_from('<254h',initializer03,FT_END+4)
    require(all(value==64 for value in bias),'auxiliary FT bias64 required')
    difference=tuple(a-b for a,b in zip(*raw))
    require(all(value%2==0 and abs(value)<=80 for value in difference),'equal feature parity/rawdifference80 required')
    Z=tuple(value//2 for value in difference)
    M=material.material(position);integer(M,-25780,25780,'fixed material STM bound required')
    return Z,M,{'raw_us':raw[0],'raw_them':raw[1],'active_features':len(us)}


def design_row(position,initializer03,original01,*,binding):
    """Strict public row adapter; arbitrary new03 FT prefixes are rejected."""
    require(initializer03==transform_initializer(original01,binding=binding),'fixed transformed initializer03 required')
    return _design_row(position,initializer03)


@dataclass(frozen=True)
class WhiteDesign:
    z_bytes:bytes
    d_bytes:bytes
    count:int
    dimension:int
    binding_json:bytes
    provenance_json:bytes
    design_sha256:str


_issued_designs={}


def _design_digest(z_bytes,d_bytes,count,binding_json,provenance_json):
    return sha(b'sekirei.white-view-design.v1\0'+struct.pack('<QQ',count,254)+binding_json+b'\0'+provenance_json+b'\0'+z_bytes+d_bytes)


def build_design(ordered_sfens,labels,initializer03,original01,*,binding,input_digests,check_deadline=lambda:None):
    """Pure SFEN-keyed TRAIN adapter; external manifest/replay not claimed.

    Position order is authoritative. Label order may differ. Future actual
    driver must first validate the original manifest and all four source file
    hashes/provenance with existing guards; this function only consumes those
    caller-verified TRAIN SFEN rows and original O label objects in memory.
    """
    binding=validate_binding(binding);validate_original_initializer(original01)
    require(initializer03==transform_initializer(original01,binding=binding),'exact dedicated transformed initializer03 required')
    require(type(ordered_sfens) is tuple and type(labels) is tuple and 1<=len(ordered_sfens)<=1_000_000 and len(labels)==len(ordered_sfens),'immutable bounded ordered SFEN/label tuples required')
    require(type(input_digests) is dict and set(input_digests)=={'manifest','train_positions','train_labels'},'external three original TRAIN input digests required')
    for value in input_digests.values():strict_sha(value)
    require(callable(check_deadline),'deadline callback required');check_deadline()
    cache={}
    for label in labels:
        require(type(label) is dict and {'sfen','score_cp','teacher_identity','label_depth'}<=set(label),'original teacher storage fields required')
        sfen=label['sfen'];require(type(sfen) is str and sfen not in cache,'duplicate or invalid teacher SFEN')
        integer(label['score_cp'],-29999,29999,'original T must be signed integer abs<30000, never bool/float')
        require(label['teacher_identity']==ORIGINAL_TEACHER and type(label['teacher_identity']) is str and exact(label['label_depth'],0),'original O teacher/depth0 required')
        cache[sfen]=label['score_cp']
    require(set(ordered_sfens)==set(cache) and len(set(ordered_sfens))==len(ordered_sfens),'missing/extra/duplicate exact SFEN cache join')
    zb=bytearray();db=bytearray();boards=set();ordered=hashlib.sha256()
    raw_min,raw_max=104,24
    for index,sfen in enumerate(ordered_sfens):
        if index%256==0:check_deadline()
        require(type(sfen) is str and sfen and '\n' not in sfen and '\r' not in sfen,'single-line exact SFEN required')
        position=material.parse_sfen(sfen);validate_position(position)
        semantic=(tuple(sorted(position['pieces'])),tuple(tuple(hand) for hand in position['hands']),position['stm'])
        require(semantic not in boards,'duplicate semantic board with different SFEN ply');boards.add(semantic)
        Z,M,evidence=_design_row(position,initializer03)
        target=cache[sfen]-M;integer(target,-55779,55779,'bounded original teacher minus fixed material required')
        zb.extend(value&255 for value in Z);db.extend(struct.pack('<i',target))
        raw_min=min(raw_min,min(evidence['raw_us']),min(evidence['raw_them']))
        raw_max=max(raw_max,max(evidence['raw_us']),max(evidence['raw_them']))
        ordered.update(sfen.encode('utf-8')+b'\n')
    check_deadline();z_bytes,d_bytes=bytes(zb),bytes(db)
    provenance={'schema':'sekirei.white-view-paired-linear-train-design.v1','mode':MODE,'feature_schema':FEATURE_SCHEMA,
        'count':len(ordered_sfens),'dimension':254,'input_digests':dict(input_digests),'original_teacher_identity':ORIGINAL_TEACHER,
        'stock_initializer_sha256':sha(original01),'transformed_initializer_sha256':sha(initializer03),
        'ordered_sfens_sha256':ordered.hexdigest(),'z_sha256':sha(z_bytes),'d_sha256':sha(d_bytes),
        'z_definition':'(raw_us-raw_them)//2,auxchannels2:256;int8C-row-major','x_definition':'Z/16',
        'd_definition':'original absolute STM T minus fixed material M;signed-i32LE','position_order_preserved':True,
        'exact_sfen_join_verified':True,'labels_extra_metadata_retained_in_original_input_only':True,
        'ft_raw_min':raw_min,'ft_raw_max':raw_max,'holdout_rows_read':False,'development_used':False,'final_used':False,
        'actual_original_manifest_and_replay_verified':False,'actual_source_files_and_core_verified':False,'actual_fit_verified':False}
    bj,pj=canonical_json(binding),canonical_json(provenance)
    result=WhiteDesign(z_bytes,d_bytes,len(ordered_sfens),254,bj,pj,_design_digest(z_bytes,d_bytes,len(ordered_sfens),bj,pj))
    _issued_designs[id(result)]=(result,result.design_sha256)
    return result


def to_verified_numeric_design(design,producer,*,binding,expected_design_sha256,expected_producer_sha256):
    """Only design issuance; never Gram, FISTA, source read or actual fit."""
    require(type(design) is WhiteDesign and id(design) in _issued_designs and _issued_designs[id(design)][0] is design,'exact issued white design object required')
    digest=_design_digest(design.z_bytes,design.d_bytes,design.count,design.binding_json,design.provenance_json)
    require(digest==_issued_designs[id(design)][1]==design.design_sha256==strict_sha(expected_design_sha256),'white design content/origin digest differs')
    require(design.binding_json==canonical_json(validate_binding(binding)),'external white design/source/core binding differs')
    require(strict_sha(expected_producer_sha256)==NUMERIC_PRODUCER_SHA256 and producer.__name__=='verified_design_gram','fixed unchanged numeric producer required')
    return producer.verified_design_bytes(design.z_bytes,design.d_bytes,design.count,design.dimension)


def runtime_entry(*args,**kwargs):
    raise RuntimeError('SOURCE ONLY prototype; actual data/fit/build/probe/file entry is blocked')


if __name__=='__main__':runtime_entry()
