"""Pure SOURCE contract. No paths, model/file reads, imports of actual controls."""
import json, math, struct
PROTOTYPE_ONLY=True
MODE='white-view-paired-nonlinear-adam-e3-v1'
RECIPE_BITS={'learning_rate':'3a83126f','head_init_width':'3b800000','head_bias_init':'40800000','output_native_l1_budget':'47000000'}
COUNTS={'train':112681,'holdout':5895,'public_fixtures':15}
ROW_KEYS={'index','native_core_cp','nearest_core_cp','native_quantized_float_cp','nearest_quantized_float_cp','material_cp','ft_prefixes_checked','all_preclamp_intermediates_finite',*(role+'_'+operation for role in ('candidate','reference') for operation in ('l2_products','l2_additions','output_products','output_additions'))}

def require(ok,reason):
    if not ok:raise ValueError(reason)

def runtime_guard():
    raise RuntimeError('SOURCE ONLY; caller provenance/source/build/lifecycle integration not activated')

def selected_recipe_bits(value):
    require(type(value)is dict and set(value)==set(RECIPE_BITS),'recipe exact four bit fields')
    require(all(type(value[k])is str and value[k]==v for k,v in RECIPE_BITS.items()),'selected nonlinear four f32bits differ')
    return dict(value)

def training_completion(*,mode,adam_used,epochs_completed,positions_per_epoch,global_step,resume_used,shuffle_used):
    require(type(mode)is str and mode==MODE,'new nonlinear mode required')
    for actual,want in [(adam_used,True),(resume_used,False),(shuffle_used,False)]:
        require(type(actual)is bool and actual is want,'Adam/fresh/noShuffle flags differ')
    for actual,want in [(epochs_completed,3),(positions_per_epoch,112681),(global_step,338043)]:
        require(type(actual)is int and actual==want,'three complete epochs/count/global step required')
    return True

def strict_json(raw):
    def pairs(items):
        out={}
        for k,v in items:
            require(k not in out,'duplicate JSON key');out[k]=v
        return out
    def bad(value):raise ValueError('nonfinite JSON constant')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=bad)

def f32(value):
    require(type(value)is float and math.isfinite(value),'typed finite probe float required')
    try:data=struct.pack('<f',value)
    except (OverflowError,struct.error) as e:raise ValueError('probe float outside binary32')from e
    restored=struct.unpack('<f',data)[0];require(math.isfinite(restored),'nonfinite restored binary32')
    return restored

def core_row(raw,index,material):
    row=strict_json(raw)
    require(type(row)is dict and set(row)==ROW_KEYS,'dedicated nonlinear row fields differ')
    require(type(index)is int and index>=0 and type(row['index'])is int and row['index']==index,'row order/index differs')
    require(type(material)is int,'fixed Python material integer required')
    for k in ('native_core_cp','nearest_core_cp','material_cp'):
        require(type(row[k])is int and abs(row[k])<899000,'ordinary typed integer cp required')
    require(row['material_cp']==material and row['nearest_core_cp']==material,'reference03/compiled material/Python M differs')
    for k in ('ft_prefixes_checked','all_preclamp_intermediates_finite'):
        require(type(row[k])is bool and row[k]is True,'preclamp/FT checking missing')
    for role in ('candidate','reference'):
        for op,want in [('l2_products',16384),('l2_additions',16384),('output_products',32),('output_additions',32)]:
            actual=row[role+'_'+op];require(type(actual)is int and actual==want,'every L2/output operation count differs')
    for prefix in ('native','nearest'):
        k=prefix+'_quantized_float_cp';row[k]=f32(row[k])
        require(abs(row[k])<899000 and abs(row[k]-row[prefix+'_core_cp'])<1.001,'stored nearest native float/core bridge differs')
    require(struct.pack('<f',row['nearest_quantized_float_cp'])==struct.pack('<f',float(material)),'reference quantized float not fixed M')
    # Deliberately no old linear material residual <=100cp condition.
    return row


INCREMENTAL_KEYS={'positions_checked','fixtures','walks','search_walks','captures','promotions','drops','undos','null_undos','max_material_difference_cp','incremental_refresh_error_cp','accumulator_refresh_equal','parent_restoration_equal','observed_float_intermediates_finite','all_preclamp_intermediates_finite','mxcsr_control','material_bound_enabled','normal_score_domain_verified','native_structure_verified','native_loaded_fullbytes_equal','max_abs_core_cp','max_abs_quantized_float_cp','maximum_stored_float_core_bridge_cp','l2_products_checked','l2_additions_checked','output_products_checked','output_additions_checked','native_q_l1_upper','native_absolute_cp_upper'}
def incremental_result(value):
    require(type(value)is dict and set(value)==INCREMENTAL_KEYS,'dedicated nonlinear incremental exact fields required')
    for k,want in [('positions_checked',8185),('fixtures',15),('walks',16),('search_walks',8),('incremental_refresh_error_cp',0),('l2_products_checked',8185*16384),('l2_additions_checked',8185*16384),('output_products_checked',8185*32),('output_additions_checked',8185*32)]:
        require(type(value[k])is int and value[k]==want,'fixed incremental scope/count differs')
    for k in ('captures','promotions','drops','undos','null_undos'):
        require(type(value[k])is int and value[k]>0,'required transition was not observed')
    for k in ('accumulator_refresh_equal','parent_restoration_equal','observed_float_intermediates_finite','all_preclamp_intermediates_finite','normal_score_domain_verified','native_structure_verified','native_loaded_fullbytes_equal'):
        require(type(value[k])is bool and value[k]is True,'incremental finite/native/restoration property missing')
    require(type(value['material_bound_enabled'])is bool and value['material_bound_enabled']is False,'old linear100cap is not this contract')
    require(type(value['mxcsr_control'])is str and value['mxcsr_control']=='9fc0','fixed float control differs')
    require(type(value['max_material_difference_cp'])is int and value['max_material_difference_cp']>=0,'material delta is an observed statistic')
    require(type(value['max_abs_core_cp'])is int and 0<=value['max_abs_core_cp']<899000,'integer ordinary-score domain differs')
    maximum=f32(value['max_abs_quantized_float_cp']);require(0<=maximum<899000,'float ordinary-score domain differs')
    for k in ('maximum_stored_float_core_bridge_cp','native_q_l1_upper','native_absolute_cp_upper'):
        require(type(value[k])is float and math.isfinite(value[k]) and value[k]>=0,'finite native statistic required')
    require(value['maximum_stored_float_core_bridge_cp']<1.001,'stored native incremental float/core bridge differs')
    require(value['native_q_l1_upper']<=32768 and 0<value['native_absolute_cp_upper']<899000,'selected native functional budget differs')
    require(maximum<=value['native_absolute_cp_upper'],'observed native prediction exceeds computed upper bound')
    return dict(value)
