//! SOURCE ONLY v2 typed parent/source-binding validator. No FS/process I/O.
//! The caller supplies immutable raw snapshots verified by guarded Reader.
//! Implements Root-declared typed predecessor bodies; no bool skips/callbacks.
use std::{collections::{BTreeMap,BTreeSet},path::PathBuf};
use serde::Deserialize;
use serde_json::Value;
use crate::paired_nonlinear_cli::{Args,FullRef};
use crate::paired_nonlinear_bound_io::{self as io,Snapshot};
use crate::paired_nonlinear_sha256 as sha;
use crate::weekly_nonlinear_profile::{MODE, PLAN_BYTES, PLAN_SHA, MANIFEST_SHA};
use crate::weekly_nonlinear_profile as weekly;
const TEACHER:&str="external:suisho11beta-1m-pack:376d4ef6e503d2ebe687f99e873103845b6eddc0c04d0d08e9b9c785ea061b8d";
const FLOAT_POLICY:&str="x86-ftz-daz";

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
    training_source_files:IdentityMap,compiler_files:IdentityMap,engine_build:Ref,#[serde(rename="dataset_inputs")]original_inputs:OriginalFive,reference03:Ref,float_policy:String,
}
#[derive(Deserialize,PartialEq,Eq)]
#[serde(deny_unknown_fields)]
struct OriginCounts{train:u64,holdout:u64,acquired_pool_games:u64,selected_pack_games:u64}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Checks{
    fresh_dataset_origin_raw_and_transitive_refs_valid:bool,fresh_dataset_schema_semantics_and_join_valid:bool,
    frozen_holdout_and_pool_exclusion_verified:bool,reference03_fullbytes_reconstructed:bool,
    new_training_source_build_compiler_abi_bound:bool,fixed_white_view_engine_unchanged:bool,
    inputs_before_after_equal:bool,source_membership_equal:bool,no_conflicting_heavy_process:bool,
}
impl Checks{fn all(&self)->bool{
    self.fresh_dataset_origin_raw_and_transitive_refs_valid&&self.fresh_dataset_schema_semantics_and_join_valid
    &&self.frozen_holdout_and_pool_exclusion_verified&&self.reference03_fullbytes_reconstructed
    &&self.new_training_source_build_compiler_abi_bound&&self.fixed_white_view_engine_unchanged
    &&self.inputs_before_after_equal&&self.source_membership_equal&&self.no_conflicting_heavy_process
}}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Parent{
    schema:String,status:String,mode:String,selected_plan:Ref,training_build:Ref,engine_build:Ref,
    #[serde(rename="dataset_inputs")]original_inputs:OriginalFive,reference03:Ref,
    #[serde(rename="dataset_origin_proof")]prior_original_origin_proof:Ref,origin_proof_policy:String,
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
    let o=io::object(v,&["schema","status","mode","hypothesis","created_utc","architecture","training",
        "dataset_inputs","teacher_identity","selection_profile","resources","adoption","incumbent_model"])?;
    io::require(o["schema"].as_str()==Some("sekirei.weekly-nonlinear-epoch-shuffle-mse-selected-plan.v1")
        &&o["status"].as_str()==Some("frozen-before-build")&&o["mode"].as_str()==Some(MODE)
        &&o["teacher_identity"].as_str()==Some(TEACHER),"weekly selected plan header differs")?;
    for key in ["hypothesis","created_utc"]{io::require(o[key].as_str().is_some_and(|s|!s.is_empty()),"nonempty plan text required")?;}
    let architecture=serde_json::json!({"feature_schema":"flat_white_view_aux_tied_v1","dimensions":[2420,256,32],
        "native_magic":"SEKIRW03","protected_material_units":[0,1,2,3],"all_ft_bias_fixed_q":64,
        "hand_ties":[[0,3],[1,2]],"aux_ft_channels":254,"paired_heads":14});
    io::require(io::exact(&o["architecture"],&architecture),"weekly architecture differs")?;
    let training=serde_json::json!({"seed":42,"epochs":3,"selected_epoch":3,"train_count":112681,"holdout_count":5895,
        "objective":"absolute-cp-mse","optimizer":"fresh-adam-tied-masters","learning_rate_schedule":"constant",
        "learning_rate_f32_bits":"3a83126f","head_init_width_f32_bits":"3b800000","head_bias_init_f32_bits":"40800000",
        "output_native_l1_budget_f32_bits":"47000000","shuffle_seed":20261006,
        "row_order":crate::trainer::paired_nonlinear_positions::row_order_declaration(),"resume_allowed":false,
        "ft_saved_q_max":797,"ft_bias_q":64});
    io::require(io::exact(&o["training"],&training),"weekly fixed numeric recipe differs")?;
    let profile:Ref=serde_json::from_value(o["selection_profile"].clone()).map_err(|e|e.to_string())?;profile.validate()?;
    io::require(profile.sha256==weekly::PROFILE_SHA,"weekly preregistered selection profile differs")?;
    let incumbent:Ref=serde_json::from_value(o["incumbent_model"].clone()).map_err(|e|e.to_string())?;incumbent.validate()?;
    let resources=io::object(&o["resources"],&["heavy_serial","build_jobs","analysis_jobs","threads",
        "training_wall_limit_seconds","added_storage_budget_bytes","minimum_ssd_remaining_bytes"])?;
    for (key,value) in [("heavy_serial",serde_json::json!(true)),("build_jobs",serde_json::json!(2)),
        ("analysis_jobs",serde_json::json!(1)),("threads",serde_json::json!(1))]{
        io::require(io::exact(&resources[key],&value),"weekly fixed resource condition differs")?;
    }
    for key in ["training_wall_limit_seconds","added_storage_budget_bytes","minimum_ssd_remaining_bytes"]{
        io::require(resources[key].as_u64().is_some_and(|x|x>0),"positive typed resource budget required")?;
    }
    io::require(resources["training_wall_limit_seconds"].as_u64().unwrap()<=86400,"training attempt exceeds one-day limit")?;
    let adoption=serde_json::json!({"development_games":5,"go_nodes":1000000,"mae_multipv":1,"top3_multipv":3,
        "rule":"strict MAE decrease and no Top3 decrease against latest incumbent",
        "final_used_for_daily_selection":false,"learning_loss_is_adoption_criterion":false});
    io::require(io::exact(&o["adoption"],&adoption),"fixed development adoption conditions differ")?;
    let dataset:OriginalFive=serde_json::from_value(o["dataset_inputs"].clone()).map_err(|e|e.to_string())?;
    dataset.validate(args)?;Ok(dataset)
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
    pub fn frozen_manifest_ref(&self,inputs:&RawInputs)->Result<FullRef,String>{
        let p:Parent=inputs.typed(&self.binding.parent_preflight)?;
        let origin:FreshOrigin=inputs.typed(&p.prior_original_origin_proof)?;
        Ok(origin.frozen_dataset_inputs.manifest.cli())
    }
    pub fn additional_parent_input_refs(&self,inputs:&RawInputs)->Result<Vec<FullRef>,String>{
        let p:Parent=inputs.typed(&self.binding.parent_preflight)?;identity_map(&p.source_inputs)?;
        Ok(p.source_inputs.into_iter().map(|(path,i)|Ref{path,bytes:i.bytes,sha256:i.sha256}.cli()).collect())
    }
    /// Consume typed Root outputs and all bound raw predecessor identities.
    /// No True field is accepted without its exact producer/body/source binding.
    pub fn validate_parent_and_source(&self,inputs:&RawInputs,args:&Args,run_request_path:Option<&str>)->Result<(),String>{
        let b=&self.binding;let p:Parent=inputs.typed(&b.parent_preflight)?;
        io::require(p.schema=="sekirei.weekly-nonlinear-epoch-shuffle-mse-parent-preflight.v1"&&p.status=="complete"&&p.mode==MODE,"parent exact schema/status/mode differ")?;
        io::require(p.origin_proof_policy=="fresh-whole-pack-ranked-dataset-with-frozen-holdout","parent origin proof policy differs")?;
        io::require(p.selected_plan==b.selected_plan&&p.training_build==b.training_build&&p.engine_build==b.engine_build
            &&p.original_inputs==b.original_inputs&&p.reference03==b.reference03,"parent/source binding refs differ")?;
        p.original_inputs.validate(args)?;strict_sha40(&p.source_head)?;
        io::require(p.origin_counts==OriginCounts{train:112681,holdout:5895,acquired_pool_games:1000,selected_pack_games:weekly::TOTAL_SELECTED_GAMES},"exact original/replay counts differ")?;
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
        for key in ["selection_profile","incumbent_model"]{
            let r:Ref=serde_json::from_value(plan[key].clone()).map_err(|e|e.to_string())?;
            contains(&p.source_inputs,&r)?;inputs.raw(&r)?;
        }
        // Exact typed Root body consumers, not injectable callbacks/trust skips.
        // Root alone executes the original proof producers before this stage.
        validate_training_build_body(inputs,b,&p)?;
        validate_process_evidence_body(inputs,b,&p)?;
        validate_fresh_origin_body(inputs,b,&p)?;
        Ok(())
    }
}
/// Parse externally pinned source-binding raw bytes before staging predecessor
/// refs; does not accept parent evidence or claim filesystem/provenance success.
pub fn declared_binding(raw:&[u8],args:&Args)->Result<DeclaredBinding,String>{
    let expected=Ref::from_cli(&args.source_binding)?;
    io::require(raw.len()as u64==expected.bytes&&sha::hex(raw)?==expected.sha256,"source-binding raw external SHA differs")?;
    let b:SourceBinding=parse(raw)?;
    io::require(b.schema=="sekirei.weekly-nonlinear-epoch-shuffle-mse-source-binding.v1"&&b.status=="ready"&&b.mode==MODE
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
    io::require(t.schema=="sekirei.weekly-nonlinear-epoch-shuffle-mse-training-build.v1"&&t.status=="complete"&&t.mode==MODE
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
    for suffix in ["shuffle_stable_integer_vectors","shuffle_full_permutation_bijection_and_epoch_pins","shuffle_rejects_seed_epoch_missing_duplicate_and_order","shuffle_consumption_success_counts_and_exact_updates","shuffle_consumption_rejects_wrong_step_order_duplicate_and_incomplete","shuffle_declaration_has_exact_separate_init_seed_and_holdout_order"]{
        io::require(t.tests.required_test_names.iter().any(|name|name.ends_with(suffix)),"actual bound shuffle test missing")?;
    }
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
    io::require(e.schema=="sekirei.weekly-nonlinear-epoch-shuffle-mse-preflight-process-evidence.v1"&&e.status=="observed-clear"
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
struct FreshChecks{selection_profile_hash_body_valid:bool,selected_replay_producer_source_bound:bool,
    whole_pack_index_membership_valid:bool,frozen_holdout_bytes_preserved:bool,complete_pool_excluded:bool,
    split_and_boards_disjoint:bool,raw_input_hashes_unchanged:bool,exact_train_label_join_valid:bool}
impl FreshChecks{fn all(&self)->bool{self.selection_profile_hash_body_valid&&self.selected_replay_producer_source_bound
    &&self.whole_pack_index_membership_valid&&self.frozen_holdout_bytes_preserved&&self.complete_pool_excluded
    &&self.split_and_boards_disjoint&&self.raw_input_hashes_unchanged&&self.exact_train_label_join_valid}}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FreshOrigin{schema:String,status:String,producer_source:Ref,dataset_inputs:OriginalFive,frozen_dataset_inputs:OriginalFive,
    profile:Ref,pack_manifest:Ref,independent_pool_manifest:Ref,input_hashes:BTreeMap<String,String>,
    checks:FreshChecks,counts:OriginCounts,inputs_before:IdentityMap,inputs_after:IdentityMap,
    actual_replay_repeated_by_verifier:bool,final_used:bool}
fn validate_fresh_origin_body(inputs:&RawInputs,b:&SourceBinding,p:&Parent)->Result<(),String>{
    let fresh:FreshOrigin=inputs.typed(&p.prior_original_origin_proof)?;
    io::require(fresh.schema=="sekirei.weekly-nonlinear-epoch-shuffle-mse-dataset-origin.v1"&&fresh.status=="verified"
        &&fresh.dataset_inputs==b.original_inputs&&fresh.counts==p.origin_counts&&fresh.checks.all()
        &&!fresh.actual_replay_repeated_by_verifier&&!fresh.final_used,"fresh origin typed body differs")?;
    io::require(fresh.inputs_before==fresh.inputs_after,"fresh origin immutable input maps differ")?;
    included_map(&p.source_inputs,&fresh.inputs_before)?;inputs.map(&fresh.inputs_before)?;
    for r in [&fresh.producer_source,&fresh.profile,&fresh.pack_manifest,&fresh.independent_pool_manifest]{
        contains(&fresh.inputs_before,r)?;raw_ref(inputs,p,r)?;
    }
    for r in fresh.frozen_dataset_inputs.refs(){contains(&fresh.inputs_before,r)?;raw_ref(inputs,p,r)?;}
    for r in fresh.dataset_inputs.refs(){raw_ref(inputs,p,r)?;}
    io::require(fresh.profile.sha256==weekly::PROFILE_SHA&&fresh.pack_manifest.sha256==weekly::CORPUS_SHA
        &&fresh.frozen_dataset_inputs.manifest.sha256==weekly::FROZEN_MANIFEST_SHA,"fresh immutable source identities differ")?;
    io::require(!fresh.input_hashes.is_empty(),"fresh origin input hash map required")?;
    for (path,digest) in &fresh.input_hashes{
        io::absolute(path)?;io::sha_string(digest)?;
        io::require(fresh.inputs_before.get(path).is_some_and(|i|i.sha256==*digest),"fresh derivation raw input absent/rebound")?;
    }
    let manifest=io::unique_json(inputs.raw(&fresh.dataset_inputs.manifest)?)?;
    let derivation=io::object(&manifest["derivation"],&["kind","profile","profile_sha256","producer_sources","input_sha256","script_sha256",
        "ordering","indexes","frozen_dataset_manifest_sha256","train_histogram","old_train_histogram",
        "old_train_board_overlap","input_unchanged","wall_seconds"])?;
    io::require(io::exact(&derivation["input_sha256"],&serde_json::to_value(&fresh.input_hashes).map_err(|e|e.to_string())?)
        &&derivation["script_sha256"].as_str()==Some(fresh.producer_source.sha256.as_str()),"fresh producer/manifest/input hash linkage differs")?;
    let profile=io::unique_json(inputs.raw(&fresh.profile)?)?;
    io::require(io::exact(&derivation["profile"],&profile),"manifest profile differs from preregistered actual raw body")?;
    let producers=derivation["producer_sources"].as_object().ok_or("fresh producer closure required")?;
    io::require(io::exact(&derivation["producer_sources"],&profile["producer_sources"]),"fresh producer closure differs from preregistered profile")?;
    for (name,info) in producers{
        let matching:Vec<_>=fresh.input_hashes.iter().filter(|(path,_)|path.ends_with(&format!("/{name}"))).collect();
        io::require(matching.len()==1,"producer closure input membership differs")?;
        let identity:Identity=serde_json::from_value(info.clone()).map_err(|e|e.to_string())?;
        io::require(fresh.inputs_before.get(matching[0].0)==Some(&identity),"producer closure size/SHA differs")?;
    }
    let frozen=io::unique_json(inputs.raw(&fresh.frozen_dataset_inputs.manifest)?)?;
    for name in ["holdout.positions.jsonl","holdout.labels.jsonl"]{
        io::require(io::exact(&manifest["files"][name],&frozen["files"][name]),"fixed holdout file bytes or SHA changed")?;
    }
    // Selected-game legality is established by the pinned dataset producer.
    // This consumer binds its fresh manifest/source/raw closure and never claims
    // to have rerun the producer's cshogi replay in the Rust training child.
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
