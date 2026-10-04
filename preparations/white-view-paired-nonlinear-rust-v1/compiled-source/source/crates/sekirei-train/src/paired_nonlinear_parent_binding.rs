//! SOURCE ONLY v2 typed parent/source-binding validator. No FS/process I/O.
//! The caller supplies immutable raw snapshots verified by guarded Reader.
//! Implements Root-declared typed predecessor bodies; no bool skips/callbacks.
use std::{collections::{BTreeMap,BTreeSet},path::PathBuf};
use serde::Deserialize;
use serde_json::Value;
use crate::paired_nonlinear_cli::{Args,FullRef};
use crate::paired_nonlinear_bound_io::{self as io,Snapshot};
use crate::paired_nonlinear_sha256 as sha;
const MODE:&str="white-view-paired-nonlinear-adam-e3-v1";
const PLAN_BYTES:u64=7360;
const PLAN_SHA:&str="111f8e3f1bf404c042c9e3db3e46f46659f3f609c405c30234dbf47778e03729";
const TEACHER:&str="external:suisho11beta-1m-pack:376d4ef6e503d2ebe687f99e873103845b6eddc0c04d0d08e9b9c785ea061b8d";
const MANIFEST_SHA:&str="ecc419da180b86b046e1af507e6de0e218d271a5d9eaa9bff07f19ec072719a6";
const FLOAT_POLICY:&str="x86-ftz-daz";
const PLAN_SHAPE:&str=r##"{"schema":{"type":"str"},"status":{"type":"str"},"mode":{"type":"str"},"created_utc":{"type":"str"},"goal_thread_id":{"type":"str"},"previous_candidate":{"mode":{"type":"str"},"valid_negative":{"type":"bool"},"comparison":{"path":{"type":"str"},"bytes":{"type":"int"},"sha256":{"type":"str"}},"stopped":{"path":{"type":"str"},"bytes":{"type":"int"},"sha256":{"type":"str"}},"nas":{"path":{"type":"str"},"bytes":{"type":"int"},"sha256":{"type":"str"}}},"hypothesis":{"type":"str"},"architecture":{"feature":{"type":"str"},"dimensions":{"type":"array","item_shapes":[{"type":"int"},{"type":"int"},{"type":"int"}]},"native_magic":{"type":"str"},"aux_ft_channels":{"type":"int"},"paired_heads":{"type":"int"},"protected_material_units":{"type":"array","item_shapes":[{"type":"int"},{"type":"int"},{"type":"int"},{"type":"int"}]},"all_ft_bias_fixed_q":{"type":"int"},"hand_ties":{"type":"array","item_shapes":[{"type":"array","item_shapes":[{"type":"int"},{"type":"int"}]},{"type":"array","item_shapes":[{"type":"int"},{"type":"int"}]}]},"residual_exact_formula":{"type":"str"}},"training":{"seed":{"type":"int"},"epochs":{"type":"int"},"selected_epoch":{"type":"int"},"train_count":{"type":"int"},"holdout_count":{"type":"int"},"optimizer":{"type":"str"},"objective":{"type":"str"},"learning_rate_schedule":{"type":"str"},"shuffle_seed":{"type":"NoneType"},"resume_allowed":{"type":"bool"},"scalars":{"learning_rate":{"f32_bits":{"type":"str"},"f32_value":{"type":"float"}},"head_init_width":{"f32_bits":{"type":"str"},"f32_value":{"type":"float"}},"head_bias_init":{"f32_bits":{"type":"str"},"f32_value":{"type":"float"}},"output_native_l1_budget":{"f32_bits":{"type":"str"},"f32_value":{"type":"float"}}},"logical_masters_per_position":{"type":"int"},"total_positions":{"type":"int"},"scalar_updates":{"type":"int"},"all_inactive_moments_updated":{"type":"bool"},"ft_saved_q_limit_aux":{"type":"int"},"ft_protected_material_not_projected":{"type":"bool"},"nearest_native03":{"type":"str"}},"parameter_rationale":{"learning_rate":{"type":"str"},"head_init_width":{"type":"str"},"head_bias_init":{"type":"str"},"output_budget":{"type":"str"}},"numeric_safety":{"ordinary_score_abs_strict_limit":{"type":"int"},"mate_score":{"type":"int"},"outward_forward_bound_exact":{"numerator":{"type":"int"},"denominator":{"type":"int"}},"outward_forward_bound_cp":{"type":"float"},"quantized_native_to_core_max_abs_difference_cp":{"type":"float"},"raw_training_to_nearest_is_diagnostic_only":{"type":"bool"},"every_forward_intermediate_finite_required":{"type":"bool"},"every_gradient_moment_parameter_finite_required":{"type":"bool"},"ft_prefix_i16_range":{"type":"array","item_shapes":[{"type":"int"},{"type":"int"}]}},"resources":{"heavy_serial":{"type":"bool"},"build_jobs":{"type":"int"},"analysis_jobs":{"type":"int"},"threads":{"type":"int"},"training_wall_limit_seconds":{"type":"int"},"added_storage_budget_bytes":{"type":"int"},"minimum_ssd_remaining_bytes":{"type":"int"},"float_policy":{"type":"str"},"rounding":{"type":"str"},"capacity_evidence":{"path":{"type":"str"},"bytes":{"type":"int"},"sha256":{"type":"str"}},"capacity_evidence_float_policy":{"type":"str"},"capacity_projection_is_full_wall_guarantee":{"type":"bool"},"ftz_daz_capacity_measurement_required_before_activation":{"type":"bool"}},"activation_requirements":{"type":"array","item_shapes":[{"type":"str"},{"type":"str"},{"type":"str"},{"type":"str"},{"type":"str"},{"type":"str"},{"type":"str"}]},"adoption":{"development_games":{"type":"int"},"go_nodes":{"type":"int"},"hash_mib":{"type":"int"},"mae_multipv":{"type":"int"},"top3_multipv":{"type":"int"},"rule":{"type":"str"},"fixed_existing_white_view_runtime":{"type":"bool"},"final_used_for_daily_selection":{"type":"bool"},"learning_loss_is_adoption_criterion":{"type":"bool"}},"original_inputs":{"manifest.json":{"path":{"type":"str"},"bytes":{"type":"int"},"sha256":{"type":"str"}},"train.positions.jsonl":{"path":{"type":"str"},"bytes":{"type":"int"},"sha256":{"type":"str"}},"train.labels.jsonl":{"path":{"type":"str"},"bytes":{"type":"int"},"sha256":{"type":"str"}},"holdout.positions.jsonl":{"path":{"type":"str"},"bytes":{"type":"int"},"sha256":{"type":"str"}},"holdout.labels.jsonl":{"path":{"type":"str"},"bytes":{"type":"int"},"sha256":{"type":"str"}}},"teacher_identity":{"type":"str"},"production_training_started":{"type":"bool"},"model_adopted":{"type":"bool"},"best_model_updated":{"type":"bool"},"goal_completed":{"type":"bool"}}"##;
const PLAN_VALUES:&str=r##"{"schema":"sekirei.white-view-paired-nonlinear-selected-plan.v1","status":"selected-activation-pending","mode":"white-view-paired-nonlinear-adam-e3-v1","architecture":{"feature":"nnue_white_view_aux_tied","dimensions":[2420,256,32],"native_magic":"SEKIRW03","aux_ft_channels":254,"paired_heads":14,"protected_material_units":[0,1,2,3],"all_ft_bias_fixed_q":64,"hand_ties":[[0,3],[1,2]],"residual_exact_formula":"sum_j q_j*(a_j(A,B)-a_j(B,A))/64"},"training":{"seed":42,"epochs":3,"selected_epoch":3,"train_count":112681,"holdout_count":5895,"optimizer":"fresh-adam-tied-masters","objective":"absolute-cp-mse","learning_rate_schedule":"constant","shuffle_seed":null,"resume_allowed":false,"scalars":{"learning_rate":{"f32_bits":"3a83126f","f32_value":0.0010000000474974513},"head_init_width":{"f32_bits":"3b800000","f32_value":0.00390625},"head_bias_init":{"f32_bits":"40800000","f32_value":4.0},"output_native_l1_budget":{"f32_bits":"47000000","f32_value":32768.0}},"logical_masters_per_position":602516,"total_positions":338043,"scalar_updates":203676316188,"all_inactive_moments_updated":true,"ft_saved_q_limit_aux":797,"ft_protected_material_not_projected":true,"nearest_native03":"round-to-nearest-ties-even"},"numeric_safety":{"ordinary_score_abs_strict_limit":899000,"mate_score":900000,"outward_forward_bound_exact":{"numerator":40983855103,"denominator":262143},"outward_forward_bound_cp":156341.59639204555,"quantized_native_to_core_max_abs_difference_cp":1.001,"raw_training_to_nearest_is_diagnostic_only":true,"every_forward_intermediate_finite_required":true,"every_gradient_moment_parameter_finite_required":true,"ft_prefix_i16_range":[-32613,32741]},"resources":{"heavy_serial":true,"build_jobs":2,"analysis_jobs":1,"threads":1,"training_wall_limit_seconds":86400,"added_storage_budget_bytes":2147483648,"minimum_ssd_remaining_bytes":8589934592,"float_policy":"x86-ftz-daz","rounding":"nearest","capacity_evidence":{"path":"/home/server/.local/share/sekirei-weight2/training-17-v1/white-view-paired-nonlinear-cpu-probe-v1/capacity-outcome-v2.json","bytes":181377,"sha256":"a63c621368634a6971b1c6812fdb689ae815e050bedb63c6dcebed7aa352eca0"},"capacity_evidence_float_policy":"inherited-not-recorded","capacity_projection_is_full_wall_guarantee":false,"ftz_daz_capacity_measurement_required_before_activation":true},"activation_requirements":["fully typed original O/source/replay/legal/exclusion preflight","selected source/core/compiler/build/reader/checkpoint binding","Rust fullshape tests and actual IO readback proof","explicit FTZ/DAZ and round-nearest evidence","if scalar power caching is used, fullstate bit equivalence against original under the fixed float policy","poisoned or partial state cannot be saved or resumed","new nonlinear proof/gate; preserve old linear/noAdam/100cp gate"]}"##;

#[derive(Clone,Deserialize,PartialEq,Eq,Debug)]
#[serde(deny_unknown_fields)]
struct Ref {path:String,bytes:u64,sha256:String}
impl Ref {
    fn validate(&self)->Result<(),String>{io::absolute(&self.path)?;io::sha_string(&self.sha256)?;Ok(())}
    fn cli(&self)->FullRef{FullRef{path:PathBuf::from(&self.path),bytes:self.bytes,sha256:self.sha256.clone()}}
    fn from_cli(v:&FullRef)->Result<Self,String>{let r=Self{path:v.path.to_str().ok_or("UTF8 path required")?.into(),bytes:v.bytes,sha256:v.sha256.clone()};r.validate()?;Ok(r)}
}
#[derive(Clone,Deserialize,PartialEq,Eq,Debug)]
#[serde(deny_unknown_fields)]
struct Identity{bytes:u64,sha256:String}
type IdentityMap=BTreeMap<String,Identity>;
fn identity_map(v:&IdentityMap)->Result<(),String>{
    io::require(!v.is_empty(),"nonempty typed identity map required")?;
    for (path,info) in v{io::absolute(path)?;io::sha_string(&info.sha256)?;}Ok(())
}
fn contains(map:&IdentityMap,r:&Ref)->Result<(),String>{
    io::require(map.get(&r.path)==Some(&Identity{bytes:r.bytes,sha256:r.sha256.clone()}),"mandatory raw identity absent/rebound")
}
#[derive(Clone,Deserialize,PartialEq,Eq,Debug)]
#[serde(deny_unknown_fields)]
struct OriginalFive{
    #[serde(rename="manifest.json")]manifest:Ref,
    #[serde(rename="train.positions.jsonl")]train_positions:Ref,
    #[serde(rename="train.labels.jsonl")]train_labels:Ref,
    #[serde(rename="holdout.positions.jsonl")]holdout_positions:Ref,
    #[serde(rename="holdout.labels.jsonl")]holdout_labels:Ref,
}
impl OriginalFive{
    fn refs(&self)->[&Ref;5]{[&self.manifest,&self.train_positions,&self.train_labels,&self.holdout_positions,&self.holdout_labels]}
    fn validate(&self,args:&Args)->Result<(),String>{
        let root=args.manifest.path.parent().ok_or("O manifest parent required")?;
        let names=["manifest.json","train.positions.jsonl","train.labels.jsonl","holdout.positions.jsonl","holdout.labels.jsonl"];
        for (r,name) in self.refs().into_iter().zip(names){r.validate()?;io::require(PathBuf::from(&r.path)==root.join(name),"O five sibling paths differ")?;}
        io::require(self.manifest==Ref::from_cli(&args.manifest)?&&self.train_positions==Ref::from_cli(&args.positions)?
            &&self.train_labels==Ref::from_cli(&args.labels)?&&self.manifest.sha256==MANIFEST_SHA,"fixed O/CLI identities differ")
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceBinding{
    schema:String,status:String,mode:String,selected_plan:Ref,parent_preflight:Ref,training_build:Ref,training_binary:Ref,
    training_source_files:IdentityMap,compiler_files:IdentityMap,engine_build:Ref,original_inputs:OriginalFive,reference03:Ref,float_policy:String,
}
#[derive(Deserialize,PartialEq,Eq)]
#[serde(deny_unknown_fields)]
struct OriginCounts{train:u64,holdout:u64,prior_raw_games:u64,prior_replay_source_files:u64}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Checks{
    prior_origin_proof_raw_and_transitive_refs_valid:bool,fresh_original_O_schema_semantics_and_join_valid:bool,
    original_pool_exclusion_split_unchanged:bool,reference03_fullbytes_reconstructed:bool,
    new_training_source_build_compiler_abi_bound:bool,fixed_white_view_engine_unchanged:bool,
    inputs_before_after_equal:bool,source_membership_equal:bool,no_conflicting_heavy_process:bool,
}
impl Checks{fn all(&self)->bool{
    self.prior_origin_proof_raw_and_transitive_refs_valid&&self.fresh_original_O_schema_semantics_and_join_valid
    &&self.original_pool_exclusion_split_unchanged&&self.reference03_fullbytes_reconstructed
    &&self.new_training_source_build_compiler_abi_bound&&self.fixed_white_view_engine_unchanged
    &&self.inputs_before_after_equal&&self.source_membership_equal&&self.no_conflicting_heavy_process
}}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Parent{
    schema:String,status:String,mode:String,selected_plan:Ref,training_build:Ref,engine_build:Ref,
    original_inputs:OriginalFive,reference03:Ref,prior_original_origin_proof:Ref,origin_proof_policy:String,
    origin_counts:OriginCounts,source_inputs:IdentityMap,inputs_before:IdentityMap,inputs_after:IdentityMap,
    checks:Checks,process_evidence:Value,source_head:String,
}
fn parse<T:serde::de::DeserializeOwned>(raw:&[u8])->Result<T,String>{
    serde_json::from_value(io::unique_json(raw)?).map_err(|e|e.to_string())
}
fn strict_sha40(s:&str)->Result<(),String>{io::require(s.len()==40&&s.bytes().all(|b|b.is_ascii_digit()||(b'a'..=b'f').contains(&b)),"source Git SHA40 required")}
fn check_plan_shape(v:&Value,shape:&Value)->Result<(),String>{
    if let Some(typ)=shape.get("type"){
        match typ.as_str().ok_or("shape type")?{
            "str"=>io::require(v.is_string(),"plan strict string required"),
            "int"=>io::require(v.as_i64().is_some()||v.as_u64().is_some(),"plan strict integer required"),
            "float"=>io::require(v.as_number().is_some_and(|n|n.is_f64())&&v.as_f64().is_some_and(|x|x.is_finite()),"plan strict finite float required"),
            "bool"=>io::require(v.is_boolean(),"plan strict bool required"),
            "NoneType"=>io::require(v.is_null(),"plan explicit null required"),
            "array"=>{
                let items=v.as_array().ok_or("plan array required")?;
                let expected=shape["item_shapes"].as_array().ok_or("shape array")?;
                io::require(items.len()==expected.len(),"plan array exact size differs")?;
                for (x,s) in items.iter().zip(expected){check_plan_shape(x,s)?;}Ok(())
            },
            _=>Err("unknown source shape type".into()),
        }
    }else{
        let schema=shape.as_object().ok_or("source shape object")?;
        let object=v.as_object().ok_or("plan object required")?;
        io::require(object.len()==schema.len()&&object.keys().all(|k|schema.contains_key(k)),"unknown/missing plan fields")?;
        for (k,s) in schema{check_plan_shape(&object[k],s)?;}Ok(())
    }
}
/// Strict memory-only plan parsing. Actual raw plan SHA is checked separately.
fn validate_plan_value(v:&Value,args:&Args)->Result<OriginalFive,String>{
    let shape:Value=serde_json::from_str(PLAN_SHAPE).map_err(|e|e.to_string())?;check_plan_shape(v,&shape)?;
    let fixed:Value=serde_json::from_str(PLAN_VALUES).map_err(|e|e.to_string())?;
    for (key,value) in fixed.as_object().ok_or("fixed plan values")?{
        io::require(io::exact(&v[key],value),"selected plan normative value/strict type differs")?;
    }
    io::require(v["teacher_identity"].as_str()==Some(TEACHER),"selected plan O teacher differs")?;
    for key in ["production_training_started","model_adopted","best_model_updated","goal_completed"]{
        io::require(v[key].as_bool()==Some(false),"selected plan premature success/activation")?;
    }
    for key in ["comparison","stopped","nas"]{
        let r:Ref=serde_json::from_value(v["previous_candidate"][key].clone()).map_err(|e|e.to_string())?;r.validate()?;
    }
    io::require(v["previous_candidate"]["valid_negative"].as_bool()==Some(true),"selected prior valid negative required")?;
    let capacity:Ref=serde_json::from_value(v["resources"]["capacity_evidence"].clone()).map_err(|e|e.to_string())?;capacity.validate()?;
    let original:OriginalFive=serde_json::from_value(v["original_inputs"].clone()).map_err(|e|e.to_string())?;
    original.validate(args)?;Ok(original)
}

/// Immutable raw byte container. Digest/type binding is not filesystem or
/// provenance validation; caller must derive it from Reader's pinned snapshots.
pub struct RawInputs{entries:BTreeMap<String,(Ref,Vec<u8>)>}
impl RawInputs{
    pub fn from_reader_snapshots(snapshots:&BTreeMap<PathBuf,Snapshot>)->Result<Self,String>{
        let mut entries=BTreeMap::new();
        for snapshot in snapshots.values(){
            let reference=Ref::from_cli(&snapshot.reference)?;
            io::require(reference.bytes==snapshot.bytes.len()as u64&&sha::hex(&snapshot.bytes)?==reference.sha256,"snapshot raw byte identity changed")?;
            io::require(entries.insert(reference.path.clone(),(reference,snapshot.bytes.clone())).is_none(),"duplicate raw snapshot path")?;
        }Ok(Self{entries})
    }
    fn raw(&self,r:&Ref)->Result<&[u8],String>{
        r.validate()?;let (expected,raw)=self.entries.get(&r.path).ok_or("raw snapshot missing")?;
        io::require(expected==r&&raw.len()as u64==r.bytes&&sha::hex(raw)?==r.sha256,"raw expected SHA/size differs before JSON parse")?;
        Ok(raw)
    }
    fn typed<T:serde::de::DeserializeOwned>(&self,r:&Ref)->Result<T,String>{parse(self.raw(r)?) }
    fn map(&self,map:&IdentityMap)->Result<(),String>{
        identity_map(map)?;
        for (path,info) in map{self.raw(&Ref{path:path.clone(),bytes:info.bytes,sha256:info.sha256.clone()})?;}Ok(())
    }
}
/// Opaque structural declaration. A header is never a verified-parent token;
/// predecessor body/source/whole raw closure validation is mandatory.
pub struct DeclaredBinding{binding:SourceBinding}
impl DeclaredBinding{
    pub fn require_current_exe(&self,current:&std::path::Path)->Result<(),String>{
        io::require(current==PathBuf::from(&self.binding.training_binary.path),"actual current_exe differs from bound training binary")
    }
    pub fn original_refs(&self)->BTreeMap<String,FullRef>{
        ["manifest.json","train.positions.jsonl","train.labels.jsonl","holdout.positions.jsonl","holdout.labels.jsonl"]
            .into_iter().zip(self.binding.original_inputs.refs()).map(|(name,r)|(name.into(),r.cli())).collect()
    }
    pub fn raw_refs_required(&self)->Vec<FullRef>{
        let b=&self.binding;let mut refs=vec![b.selected_plan.cli(),b.parent_preflight.cli(),b.training_build.cli(),b.training_binary.cli(),b.engine_build.cli(),b.reference03.cli()];
        refs.extend(b.original_inputs.refs().into_iter().map(Ref::cli));
        for map in [&b.training_source_files,&b.compiler_files]{for (path,info) in map{refs.push(Ref{path:path.clone(),bytes:info.bytes,sha256:info.sha256.clone()}.cli());}}
        refs
    }
    pub fn prior_proof_raw_ref(&self,inputs:&RawInputs)->Result<FullRef,String>{
        let p:Parent=inputs.typed(&self.binding.parent_preflight)?;Ok(p.prior_original_origin_proof.cli())
    }
    pub fn additional_parent_input_refs(&self,inputs:&RawInputs)->Result<Vec<FullRef>,String>{
        let p:Parent=inputs.typed(&self.binding.parent_preflight)?;identity_map(&p.source_inputs)?;
        Ok(p.source_inputs.into_iter().map(|(path,i)|Ref{path,bytes:i.bytes,sha256:i.sha256}.cli()).collect())
    }
    /// Consume typed Root outputs and all bound raw predecessor identities.
    /// No True field is accepted without its exact producer/body/source binding.
    pub fn validate_parent_and_source(&self,inputs:&RawInputs,args:&Args,run_request_path:Option<&str>)->Result<(),String>{
        let b=&self.binding;let p:Parent=inputs.typed(&b.parent_preflight)?;
        io::require(p.schema=="sekirei.white-view-paired-nonlinear-parent-preflight.v2"&&p.status=="complete"&&p.mode==MODE,"parent exact schema/status/mode differ")?;
        io::require(p.origin_proof_policy=="reuse-immutable-prior-full-origin-proof-plus-fresh-O-byte-and-semantic-validation","parent origin proof policy differs")?;
        io::require(p.selected_plan==b.selected_plan&&p.training_build==b.training_build&&p.engine_build==b.engine_build
            &&p.original_inputs==b.original_inputs&&p.reference03==b.reference03,"parent/source binding refs differ")?;
        p.original_inputs.validate(args)?;strict_sha40(&p.source_head)?;
        io::require(p.origin_counts==OriginCounts{train:112681,holdout:5895,prior_raw_games:1000,prior_replay_source_files:1042},"exact original/replay counts differ")?;
        identity_map(&p.source_inputs)?;
        io::require(p.source_inputs==p.inputs_before&&p.source_inputs==p.inputs_after&&p.checks.all(),"parent identity maps/check fields differ")?;
        let mut forbidden=BTreeSet::from([args.recipe.path.to_string_lossy().into_owned(),args.source_binding.path.to_string_lossy().into_owned(),b.parent_preflight.path.clone()]);
        if let Some(path)=run_request_path{io::absolute(path)?;forbidden.insert(path.into());}
        for path in p.source_inputs.keys(){io::require(!forbidden.contains(path),"DAG predecessor has recipe/SB/request/parent-self back reference")?;}
        for r in self.raw_refs_required(){
            let reference=Ref::from_cli(&r)?;
            if reference!=b.parent_preflight{contains(&p.source_inputs,&reference)?;}
        }
        contains(&p.source_inputs,&p.prior_original_origin_proof)?;
        for map in [&b.training_source_files,&b.compiler_files]{
            identity_map(map)?;
            for (path,i) in map{io::require(p.source_inputs.get(path)==Some(i),"source/compiler map absent or rebound")?;}
        }
        inputs.map(&p.source_inputs)?;inputs.raw(&p.prior_original_origin_proof)?;
        io::require(b.selected_plan.bytes==PLAN_BYTES&&b.selected_plan.sha256==PLAN_SHA,"fixed selected plan raw identity differs")?;
        let plan=io::unique_json(inputs.raw(&b.selected_plan)?)?;
        let original=validate_plan_value(&plan,args)?;
        io::require(original==b.original_inputs,"plan/source/parent original five refs differ")?;
        for key in ["comparison","stopped","nas"]{
            let r:Ref=serde_json::from_value(plan["previous_candidate"][key].clone()).map_err(|e|e.to_string())?;
            contains(&p.source_inputs,&r)?;inputs.raw(&r)?;
        }
        let capacity:Ref=serde_json::from_value(plan["resources"]["capacity_evidence"].clone()).map_err(|e|e.to_string())?;
        contains(&p.source_inputs,&capacity)?;inputs.raw(&capacity)?;
        // Exact typed Root body consumers, not injectable callbacks/trust skips.
        // Root alone executes the original proof producers before this stage.
        validate_training_build_body(inputs,b,&p)?;
        validate_process_evidence_body(inputs,b,&p)?;
        validate_prior_origin_body(inputs,b,&p)?;
        Ok(())
    }
}
/// Parse externally pinned source-binding raw bytes before staging predecessor
/// refs; does not accept parent evidence or claim filesystem/provenance success.
pub fn declared_binding(raw:&[u8],args:&Args)->Result<DeclaredBinding,String>{
    let expected=Ref::from_cli(&args.source_binding)?;
    io::require(raw.len()as u64==expected.bytes&&sha::hex(raw)?==expected.sha256,"source-binding raw external SHA differs")?;
    let b:SourceBinding=parse(raw)?;
    io::require(b.schema=="sekirei.white-view-paired-nonlinear-source-binding.v2"&&b.status=="ready"&&b.mode==MODE
        &&b.float_policy==FLOAT_POLICY,"source-binding schema/status/mode/float policy differ")?;
    io::require(b.selected_plan.bytes==PLAN_BYTES&&b.selected_plan.sha256==PLAN_SHA,"selected plan raw identity differs")?;
    for r in [&b.selected_plan,&b.parent_preflight,&b.training_build,&b.training_binary,&b.engine_build,&b.reference03]{r.validate()?;}
    b.original_inputs.validate(args)?;
    io::require(b.reference03==Ref::from_cli(&args.reference03)?,"source-binding CLI reference03 differs")?;
    identity_map(&b.training_source_files)?;identity_map(&b.compiler_files)?;
    let forbidden=BTreeSet::from([args.recipe.path.to_string_lossy().into_owned(),args.source_binding.path.to_string_lossy().into_owned()]);
    for r in [&b.selected_plan,&b.parent_preflight,&b.training_build,&b.training_binary,&b.engine_build,&b.reference03]{io::require(!forbidden.contains(&r.path),"binding predecessor back reference")?;}
    for map in [&b.training_source_files,&b.compiler_files]{for path in map.keys(){io::require(!forbidden.contains(path),"source/compiler backward recipe/SB reference")?;}}
    Ok(DeclaredBinding{binding:b})
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BuildEnvironment{
    #[serde(rename="CARGO_TARGET_DIR")]target:String,
    #[serde(rename="RUSTC")]rustc:String,
    #[serde(rename="RUSTFLAGS")]flags:String,
    #[serde(rename="CARGO_BUILD_JOBS")]jobs:String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BuildTests{log:Ref,passed:u64,failed:u64,ignored:u64,required_test_names:Vec<String>}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProcessOutcome{command:Vec<String>,pid:u64,pgid:u64,returncode:i64,waited:bool,reaped:bool,timed_out:bool,group_empty_scans:[bool;2],log:Ref,wall_seconds:Value}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TrainingBuild{
    schema:String,status:String,mode:String,producer_source:Ref,source_root:String,upstream_commit:String,source_head:String,
    source_files:IdentityMap,compiler_files:IdentityMap,dependency_files:IdentityMap,engine_build:Ref,training_binary:Ref,
    commands:Vec<Vec<String>>,environment:BuildEnvironment,tests:BuildTests,process_outcomes:Vec<ProcessOutcome>,
    inputs_before:IdentityMap,inputs_after:IdentityMap,scalar_power_cache_used:bool,
}
fn included_map(parent:&IdentityMap,child:&IdentityMap)->Result<(),String>{
    identity_map(child)?;
    for (path,i) in child{io::require(parent.get(path)==Some(i),"transitive map absent/rebound")?;}Ok(())
}
fn raw_ref(inputs:&RawInputs,parent:&Parent,r:&Ref)->Result<(),String>{r.validate()?;contains(&parent.source_inputs,r)?;inputs.raw(r)?;Ok(())}
fn check_strings(v:&[String])->Result<(),String>{io::require(!v.is_empty()&&v.iter().all(|x|!x.is_empty()&&!x.contains('\0')),"nonempty literal argv/name strings required")}
fn validate_training_build_body(inputs:&RawInputs,b:&SourceBinding,p:&Parent)->Result<(),String>{
    let t:TrainingBuild=inputs.typed(&b.training_build)?;
    io::require(t.schema=="sekirei.white-view-paired-nonlinear-training-build.v1"&&t.status=="complete"&&t.mode==MODE
        &&t.upstream_commit=="f09c13026e9485a19b4ba41b91ed2e1bbdf5e1c9"&&!t.scalar_power_cache_used,"fixed original scalar build differs")?;
    io::absolute(&t.source_root)?;strict_sha40(&t.source_head)?;
    io::require(t.source_head==p.source_head&&t.source_files==b.training_source_files&&t.compiler_files==b.compiler_files
        &&t.engine_build==b.engine_build&&t.training_binary==b.training_binary,"build/binding/parent source or refs differ")?;
    io::require(t.inputs_before==t.inputs_after,"build immutable before/after differ")?;
    included_map(&p.source_inputs,&t.inputs_before)?;inputs.map(&t.inputs_before)?;
    for map in [&t.source_files,&t.compiler_files,&t.dependency_files]{included_map(&t.inputs_before,map)?;inputs.map(map)?;}
    contains(&t.inputs_before,&t.producer_source)?;contains(&t.inputs_before,&t.engine_build)?;
    for r in [&t.producer_source,&t.engine_build,&t.training_binary,&t.tests.log]{raw_ref(inputs,p,r)?;}
    io::absolute(&t.environment.target)?;io::absolute(&t.environment.rustc)?;
    io::require(t.environment.flags=="-C target-cpu=x86-64-v3"&&t.environment.jobs=="2","fixed build environment differs")?;
    io::require(t.compiler_files.contains_key(&t.environment.rustc),"actual RUSTC not in bound compiler map")?;
    io::require(!t.commands.is_empty()&&t.commands.len()==t.process_outcomes.len(),"one process outcome per build command required")?;
    let mut test_seen=false;let mut build_seen=false;
    for (command,outcome) in t.commands.iter().zip(&t.process_outcomes){
        check_strings(command)?;
        let cargo=PathBuf::from(&command[0]);
        io::require(cargo.file_name().is_some_and(|name|name=="cargo"),"cargo command required")?;
        io::require(command.iter().any(|x|x=="--release")&&command.iter().any(|x|x=="--offline")
            &&command.iter().any(|x|x=="--locked"),"release/offline/locked required")?;
        let jobs=command.windows(2).any(|a|(a[0]=="--jobs"||a[0]=="-j")&&a[1]=="2")
            ||command.iter().any(|x|x=="--jobs=2"||x=="-j2");
        let feature=command.windows(2).any(|a|a[0]=="--features"&&a[1].split(',').any(|x|x=="nnue_white_view_aux_tied"))
            ||command.iter().any(|x|x.strip_prefix("--features=").is_some_and(|v|v.split(',').any(|x|x=="nnue_white_view_aux_tied")));
        io::require(jobs&&feature,"jobs2 and explicit new white feature required")?;
        let is_test=command.iter().any(|x|x=="test");let is_build=command.iter().any(|x|x=="build");
        io::require(is_test^is_build,"test or build cargo command required")?;
        test_seen|=is_test;build_seen|=is_build;
        io::require(outcome.command==*command&&outcome.pid>0&&outcome.pid==outcome.pgid&&outcome.returncode==0
            &&outcome.waited&&outcome.reaped&&!outcome.timed_out&&outcome.group_empty_scans==[true,true],"build process lifecycle/command differs")?;
        let wall=outcome.wall_seconds.as_f64().ok_or("numeric wall seconds required")?;
        io::require(wall.is_finite()&&wall>=0.0,"finite nonnegative wall seconds required")?;
        raw_ref(inputs,p,&outcome.log)?;
    }
    io::require(test_seen&&build_seen,"both actual tests and release build required")?;
    io::require(t.tests.passed>0&&t.tests.failed==0,"actual positive tests/no failures required")?;
    check_strings(&t.tests.required_test_names)?;
    let names:BTreeSet<_>=t.tests.required_test_names.iter().collect();
    io::require(names.len()==t.tests.required_test_names.len(),"required test names must be unique")?;
    let log=std::str::from_utf8(inputs.raw(&t.tests.log)?).map_err(|e|e.to_string())?;
    let passed:BTreeSet<&str>=log.lines().filter_map(|line|line.trim().strip_prefix("test ").and_then(|r|r.strip_suffix(" ... ok"))).collect();
    for name in &t.tests.required_test_names{io::require(passed.contains(name.as_str()),"required test not observed passing in bound actual log")?;}
    // Category names are only a necessary source/log linkage check. The Root
    // producer validates exact expected test bodies; names alone prove nothing.
    let lowered:Vec<String>=t.tests.required_test_names.iter().map(|v|v.to_lowercase()).collect();
    for needles in [&["reader"][..],&["sha"][..],&["unique"][..],&["native"][..],&["update","adapter"][..],&["ftz","float_policy","mxcsr"][..]]{
        io::require(lowered.iter().any(|name|needles.iter().any(|needle|name.contains(needle))),"required typed Reader/SHA/uniqueJSON/native/update/FTZ category missing")?;
    }
    Ok(())
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ScanRecord{conflicts:Vec<Value>,excluded_preexisting_services:Vec<Value>}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LockRecord{path:String,exclusive:bool,nonblocking:bool,acquired:bool}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProcessEvidence{schema:String,status:String,producer_source:Ref,process_reader:Ref,allowlist:Ref,
    scan_records:[ScanRecord;2],lock_records:Vec<LockRecord>,source_head:String}
fn validate_process_evidence_body(inputs:&RawInputs,_b:&SourceBinding,p:&Parent)->Result<(),String>{
    let e:ProcessEvidence=serde_json::from_value(p.process_evidence.clone()).map_err(|e|e.to_string())?;
    io::require(e.schema=="sekirei.white-view-paired-nonlinear-preflight-process-evidence.v1"&&e.status=="observed-clear"
        &&e.source_head==p.source_head,"Root process evidence schema/status/head differ")?;strict_sha40(&e.source_head)?;
    for r in [&e.producer_source,&e.process_reader,&e.allowlist]{raw_ref(inputs,p,r)?;}
    // Opaque descriptors are checked by the pinned unchanged Root-reader and
    // exact allowlist producer. Rust does not create a new EACCES exception.
    for scan in &e.scan_records{io::require(scan.conflicts.is_empty(),"Root scan conflicts not empty")?;}
    io::require(!e.lock_records.is_empty(),"nonempty acquired lock evidence required")?;
    let mut locks=BTreeSet::new();
    for lock in &e.lock_records{io::absolute(&lock.path)?;
        io::require(lock.nonblocking&&lock.acquired&&locks.insert(lock.path.clone()),"duplicate/unacquired/blocking lock evidence")?;
        // exclusive is a strict bool; mixed EX/SH coordination roles are fixed
        // by Root's producer contract, never coerced by a truthy value here.
    }
    Ok(())
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PriorChecks{prior_preflight_body_valid_under_original_producer_contract:bool,original_producer_and_validator_sources_bound:bool,
    all_prior_transitive_raw_references_valid:bool,fresh_original_O_schema_semantics_and_full_join_valid:bool,
    original_pool_exclusion_and_frozen_split_unchanged:bool,inputs_before_after_equal:bool}
impl PriorChecks{fn all(&self)->bool{self.prior_preflight_body_valid_under_original_producer_contract
    &&self.original_producer_and_validator_sources_bound&&self.all_prior_transitive_raw_references_valid
    &&self.fresh_original_O_schema_semantics_and_full_join_valid&&self.original_pool_exclusion_and_frozen_split_unchanged&&self.inputs_before_after_equal}}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReusedOrigin{schema:String,status:String,producer_source:Ref,prior_source_preflight:Ref,prior_preregistration:Ref,
    validator_sources:IdentityMap,original_inputs:OriginalFive,counts:OriginCounts,checks:PriorChecks,
    inputs_before:IdentityMap,inputs_after:IdentityMap,fresh_full_replay_performed:bool}
fn validate_prior_origin_body(inputs:&RawInputs,b:&SourceBinding,p:&Parent)->Result<(),String>{
    let prior:ReusedOrigin=inputs.typed(&p.prior_original_origin_proof)?;
    io::require(prior.schema=="sekirei.white-view-paired-nonlinear-reused-original-proof.v2"&&prior.status=="validated"
        &&!prior.fresh_full_replay_performed&&prior.original_inputs==b.original_inputs&&prior.counts==p.origin_counts
        &&prior.checks.all(),"Root reused-origin typed body differs")?;
    io::require(prior.inputs_before==prior.inputs_after,"prior wrapper immutable input maps differ")?;
    included_map(&p.source_inputs,&prior.inputs_before)?;inputs.map(&prior.inputs_before)?;
    included_map(&prior.inputs_before,&prior.validator_sources)?;inputs.map(&prior.validator_sources)?;
    for r in [&prior.producer_source,&prior.prior_source_preflight,&prior.prior_preregistration]{
        contains(&prior.inputs_before,r)?;raw_ref(inputs,p,r)?;
    }
    for r in prior.original_inputs.refs(){contains(&prior.inputs_before,r)?;raw_ref(inputs,p,r)?;}
    // Historical documents stay under their unchanged Root producer contract;
    // nevertheless duplicate/nonfinite raw JSON cannot be hidden at this boundary.
    io::unique_json(inputs.raw(&prior.prior_source_preflight)?)?;
    io::unique_json(inputs.raw(&prior.prior_preregistration)?)?;
    for (name,expected) in [("white_view_fit_contract.py","7ae7d4cddecd09c143de8bc97ea0e1851f25e33760b992cb87b6b34bea7ad649"),
        ("functional_anchor.py","e704e153e6752abe1f3a6b686ac5e9eddea688216cd28626fffb598beef8130b")]{
        let found:Vec<_>=prior.validator_sources.iter().filter(|(path,_)|PathBuf::from(path.as_str()).file_name().is_some_and(|v|v==name)).collect();
        io::require(!found.is_empty()&&found.iter().all(|(_,i)|i.sha256==expected),"unchanged original public validator SHA missing/changed")?;
    }
    // Root executes unchanged original producer/body validators + fresh full O
    // semantics, and closes all transitive refs in these raw maps. The child
    // validates this dedicated Root output; it does not spawn Python or claim
    // that status/check names alone reproduced the original replay here.
    Ok(())
}
#[cfg(test)]mod tests{
    use super::*;
    #[test]fn reader_fullref_strict_types(){
        let good=br#"{"path":"/tmp/public-fixture/empty","bytes":0,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}"#;
        let r:Ref=parse(good).unwrap();r.validate().unwrap();assert_eq!(r.bytes,0);
        for bad in [br#"{"path":"/tmp/a","bytes":true,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}"#.as_slice(),
            br#"{"path":"/tmp/a","bytes":1.0,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}"#.as_slice(),
            br#"{"path":"/tmp/a","bytes":1,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","extra":0}"#.as_slice()]{assert!(parse::<Ref>(bad).is_err());}
    }
    #[test]fn reader_plan_numeric_types(){
        let shape=serde_json::json!({"type":"float"});
        assert!(check_plan_shape(&serde_json::json!(4.0),&shape).is_ok());
        assert!(check_plan_shape(&serde_json::json!(4),&shape).is_err());
        assert!(check_plan_shape(&serde_json::json!(true),&serde_json::json!({"type":"int"})).is_err());
    }
    #[test]fn reader_recursive_unknown(){
        assert!(parse::<Ref>(br#"{"path":"/tmp/a","path":"/tmp/b","bytes":1,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}"#).is_err());
        let v=serde_json::json!({"x":true,"extra":false});
        let s=serde_json::json!({"x":{"type":"bool"}});assert!(check_plan_shape(&v,&s).is_err());
    }
}
