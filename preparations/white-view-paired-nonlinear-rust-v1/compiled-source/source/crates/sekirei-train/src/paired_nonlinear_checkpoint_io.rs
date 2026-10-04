//! SOURCE ONLY trainer child. Dedicated full-state bits checkpoint; no legacy save.
//! Uses the frozen full-state/native helper; actual I/O lives in the guarded Reader.
use super::{TrainWeights,Trainer};
use super::paired_nonlinear_adapter::{self as update,DeclaredRecipe};
use super::paired_nonlinear_checkpoint_native as native;
use crate::paired_nonlinear_bound_io as bound;
use serde::{Deserialize,Serialize};
use serde_json::Value;

#[derive(Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
struct StateBits {
    ft:Vec<u32>,
    ft_bias:Vec<u32>,
    l2:Vec<u32>,
    l2_bias:Vec<u32>,
    out:Vec<u32>,
    ft_m:Vec<u32>,
    ft_v:Vec<u32>,
    bias_m:Vec<u32>,
    bias_v:Vec<u32>,
    l2_m:Vec<u32>,
    l2_v:Vec<u32>,
    l2bias_m:Vec<u32>,
    l2bias_v:Vec<u32>,
    out_m:Vec<u32>,
    out_v:Vec<u32>,
    out_bias:u32,
    obias_m:u32,
    obias_v:u32,
    step:u64,
}
impl StateBits {
    fn capture(w:&TrainWeights)->Self {Self {
        ft:w.ft.iter().map(|x|x.to_bits()).collect(),
        ft_bias:w.ft_bias.iter().map(|x|x.to_bits()).collect(),
        l2:w.l2.iter().map(|x|x.to_bits()).collect(),
        l2_bias:w.l2_bias.iter().map(|x|x.to_bits()).collect(),
        out:w.out.iter().map(|x|x.to_bits()).collect(),
        ft_m:w.ft_m.iter().map(|x|x.to_bits()).collect(),
        ft_v:w.ft_v.iter().map(|x|x.to_bits()).collect(),
        bias_m:w.bias_m.iter().map(|x|x.to_bits()).collect(),
        bias_v:w.bias_v.iter().map(|x|x.to_bits()).collect(),
        l2_m:w.l2_m.iter().map(|x|x.to_bits()).collect(),
        l2_v:w.l2_v.iter().map(|x|x.to_bits()).collect(),
        l2bias_m:w.l2bias_m.iter().map(|x|x.to_bits()).collect(),
        l2bias_v:w.l2bias_v.iter().map(|x|x.to_bits()).collect(),
        out_m:w.out_m.iter().map(|x|x.to_bits()).collect(),
        out_v:w.out_v.iter().map(|x|x.to_bits()).collect(),
        out_bias:w.out_bias.to_bits(),
        obias_m:w.obias_m.to_bits(),
        obias_v:w.obias_v.to_bits(),
        step:w.step,
    }}
    fn restore(self)->TrainWeights {TrainWeights {
        ft:self.ft.into_iter().map(f32::from_bits).collect(),
        ft_bias:self.ft_bias.into_iter().map(f32::from_bits).collect(),
        l2:self.l2.into_iter().map(f32::from_bits).collect(),
        l2_bias:self.l2_bias.into_iter().map(f32::from_bits).collect(),
        out:self.out.into_iter().map(f32::from_bits).collect(),
        ft_m:self.ft_m.into_iter().map(f32::from_bits).collect(),
        ft_v:self.ft_v.into_iter().map(f32::from_bits).collect(),
        bias_m:self.bias_m.into_iter().map(f32::from_bits).collect(),
        bias_v:self.bias_v.into_iter().map(f32::from_bits).collect(),
        l2_m:self.l2_m.into_iter().map(f32::from_bits).collect(),
        l2_v:self.l2_v.into_iter().map(f32::from_bits).collect(),
        l2bias_m:self.l2bias_m.into_iter().map(f32::from_bits).collect(),
        l2bias_v:self.l2bias_v.into_iter().map(f32::from_bits).collect(),
        out_m:self.out_m.into_iter().map(f32::from_bits).collect(),
        out_v:self.out_v.into_iter().map(f32::from_bits).collect(),
        out_bias:f32::from_bits(self.out_bias),
        obias_m:f32::from_bits(self.obias_m),
        obias_v:f32::from_bits(self.obias_v),
        step:self.step,
    }}
}
#[derive(Clone,Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointContext {
    pub recipe:Value,pub source_binding:Value,pub manifest:Value,pub reference03:Value,
    pub positions:Value,pub labels:Value,pub source_files:Value,
}
impl CheckpointContext {
    fn validate(&self)->Result<(),String>{
        for r in [&self.recipe,&self.source_binding,&self.manifest,&self.reference03,&self.positions,&self.labels]{bound::fullref(r)?;}
        let files=self.source_files.as_object().ok_or("source map required")?;
        bound::require(!files.is_empty(),"full source map required")?;
        for (path,info) in files {
            validate_raw_source_identity(path,info)?;
        }Ok(())
    }
}
#[derive(Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
struct Checkpoint {
    schema:String,version:u32,feature_schema:String,encoding:String,resume_allowed:bool,
    epochs_completed:u32,positions_per_epoch:u64,context:CheckpointContext,state:StateBits,
}
const SCHEMA:&str="sekirei.white-view-paired-nonlinear-fullstate-bits-checkpoint.v1";
const STEPS:u64=3*112681;
fn validate_end(w:&TrainWeights,recipe:DeclaredRecipe)->Result<(),String>{
    native::validate_checkpoint_state(w,recipe,native::SnapshotStage::AfterPosition{expected_step:STEPS},native::FEATURE_TAG)
        .map_err(str::to_string)
}
/// No public mutable weight access is added. Initialized state is independently rebuilt.
pub fn fresh_trainer(recipe:DeclaredRecipe)->Result<(Trainer,TrainWeights),String>{
    update::validate_recipe(recipe).map_err(str::to_string)?;
    let reference=TrainWeights::from_nnue_weights(&native::canonical_reference_native03().map_err(str::to_string)?);
    let mut w=reference.clone();update::initialize_head(&mut w,&reference,recipe).map_err(str::to_string)?;
    native::validate_checkpoint_state(&w,recipe,native::SnapshotStage::Initialized,native::FEATURE_TAG).map_err(str::to_string)?;
    let mut trainer=Trainer::new(42,recipe.head_bias_init);
    trainer.weights=w;trainer.lr=recipe.learning_rate;
    trainer.teacher_score_cap=30000.0;trainer.exclude_mate_labels=true;
    trainer.search_target_weight=1.0;trainer.residual_material_target=false;
    // Remaining original config defaults are rejected by frozen per-position guard
    // if they ever acquire an incompatible meaning; no fallback teacher is called.
    Ok((trainer,reference))
}
pub fn completed_checkpoint_bytes(t:&Trainer,recipe:DeclaredRecipe,context:&CheckpointContext)->Result<Vec<u8>,String>{
    context.validate()?;validate_end(&t.weights,recipe)?;
    let v=serde_json::to_value(Checkpoint{schema:SCHEMA.into(),version:1,
        feature_schema:native::FEATURE_TAG.into(),encoding:"all-f32-u32-bits-le-values".into(),
        resume_allowed:false,epochs_completed:3,positions_per_epoch:112681,
        context:context.clone(),state:StateBits::capture(&t.weights)}).map_err(|e|e.to_string())?;
    bound::json_bytes(&v)
}
fn restored_completed(raw:&[u8],recipe:DeclaredRecipe,context:&CheckpointContext)->Result<TrainWeights,String>{
    context.validate()?;
    let v=bound::unique_json(raw)?;
    let c:Checkpoint=serde_json::from_value(v).map_err(|e|e.to_string())?;
    bound::require(c.schema==SCHEMA&&c.version==1&&c.feature_schema==native::FEATURE_TAG
        &&c.encoding=="all-f32-u32-bits-le-values"&&!c.resume_allowed&&c.epochs_completed==3
        &&c.positions_per_epoch==112681,"dedicated complete checkpoint envelope differs")?;
    c.context.validate()?;
    bound::require(bound::exact(&serde_json::to_value(&c.context).map_err(|e|e.to_string())?,
        &serde_json::to_value(context).map_err(|e|e.to_string())?),"checkpoint context raw identities differ")?;
    let w=c.state.restore();validate_end(&w,recipe)?;Ok(w)
}
pub fn verify_completed_readback(t:&Trainer,raw:&[u8],recipe:DeclaredRecipe,context:&CheckpointContext)->Result<(),String>{
    validate_end(&t.weights,recipe)?;let restored=restored_completed(raw,recipe,context)?;
    bound::require(native::checkpoint_state_equal(&t.weights,&restored),"all 15 vectors/3 scalars/step bits differ on readback")
}
pub fn completed_nearest_bytes(t:&Trainer,recipe:DeclaredRecipe)->Result<Vec<u8>,String>{
    validate_end(&t.weights,recipe)?;
    let w=native::nearest_native03(&t.weights,recipe,native::SnapshotStage::AfterPosition{expected_step:STEPS},native::FEATURE_TAG).map_err(str::to_string)?;
    native::encode_native03_layout(&w).map_err(str::to_string)
}
pub fn verify_completed_native(t:&Trainer,recipe:DeclaredRecipe,bytes:&[u8])->Result<(),String>{
    native::verify_native03_reexport(&t.weights,recipe,native::SnapshotStage::AfterPosition{expected_step:STEPS},native::FEATURE_TAG,bytes).map_err(str::to_string)
}
// This API accepts raw bytes only for equality verification; it cannot load a
// checkpoint into a Trainer or resume a partially completed epoch.

// Root dedicated step-zero serialization diagnostic. No EpochToken or resume API.
#[derive(Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
struct InitializedIoCheckpoint {
    schema:String,feature_schema:String,encoding:String,resume_allowed:bool,
    candidate_model:bool,epochs_completed:u32,positions_processed:u64,
    context:CheckpointContext,state:StateBits,
}
const INIT_SCHEMA:&str="sekirei.white-view-paired-nonlinear-initialized-io-checkpoint.v1";
fn validate_initialized(t:&Trainer,recipe:DeclaredRecipe)->Result<(),String>{
    native::validate_checkpoint_state(&t.weights,recipe,native::SnapshotStage::Initialized,native::FEATURE_TAG)
        .map_err(str::to_string)
}
pub fn initialized_checkpoint_bytes(t:&Trainer,recipe:DeclaredRecipe,context:&CheckpointContext)->Result<Vec<u8>,String>{
    context.validate()?;validate_initialized(t,recipe)?;
    bound::json_bytes(&serde_json::to_value(InitializedIoCheckpoint{
        schema:INIT_SCHEMA.into(),feature_schema:native::FEATURE_TAG.into(),encoding:"all-f32-u32-bits-le-values".into(),
        resume_allowed:false,candidate_model:false,epochs_completed:0,positions_processed:0,
        context:context.clone(),state:StateBits::capture(&t.weights),
    }).map_err(|e|e.to_string())?)
}
pub fn verify_initialized_readback(t:&Trainer,recipe:DeclaredRecipe,context:&CheckpointContext,raw:&[u8])->Result<(),String>{
    context.validate()?;validate_initialized(t,recipe)?;
    let value=bound::unique_json(raw)?;
    let c:InitializedIoCheckpoint=serde_json::from_value(value).map_err(|e|e.to_string())?;
    bound::require(c.schema==INIT_SCHEMA&&c.feature_schema==native::FEATURE_TAG
        &&c.encoding=="all-f32-u32-bits-le-values"&&!c.resume_allowed&&!c.candidate_model
        &&c.epochs_completed==0&&c.positions_processed==0,"initialized diagnostic envelope differs")?;
    c.context.validate()?;
    bound::require(bound::exact(&serde_json::to_value(&c.context).map_err(|e|e.to_string())?,
        &serde_json::to_value(context).map_err(|e|e.to_string())?),"initialized raw context differs")?;
    let restored=c.state.restore();
    native::validate_checkpoint_state(&restored,recipe,native::SnapshotStage::Initialized,native::FEATURE_TAG).map_err(str::to_string)?;
    bound::require(native::checkpoint_state_equal(&t.weights,&restored),"all initialized state bits differ")
}
pub fn initialized_native_bytes(t:&Trainer,recipe:DeclaredRecipe)->Result<Vec<u8>,String>{
    validate_initialized(t,recipe)?;
    let w=native::nearest_native03(&t.weights,recipe,native::SnapshotStage::Initialized,native::FEATURE_TAG).map_err(str::to_string)?;
    native::encode_native03_layout(&w).map_err(str::to_string)
}
pub fn verify_initialized_native(t:&Trainer,recipe:DeclaredRecipe,raw:&[u8])->Result<(),String>{
    validate_initialized(t,recipe)?;
    native::verify_native03_reexport(&t.weights,recipe,native::SnapshotStage::Initialized,native::FEATURE_TAG,raw).map_err(str::to_string)
}
#[cfg(test)]mod initialized_io_tests{
    use super::*;
    fn fixture()->(Trainer,DeclaredRecipe,CheckpointContext){
        let recipe=DeclaredRecipe{learning_rate:f32::from_bits(0x3a83126f),head_init_width:f32::from_bits(0x3b800000),
            head_bias_init:f32::from_bits(0x40800000),output_native_l1_budget:f32::from_bits(0x47000000)};
        let (trainer,_)=fresh_trainer(recipe).unwrap();
        let r=serde_json::json!({"path":"/tmp/initialized-source-fixture","bytes":1,"sha256":"ab".repeat(32)});
        let context=CheckpointContext{recipe:r.clone(),source_binding:r.clone(),manifest:r.clone(),reference03:r.clone(),
            positions:r.clone(),labels:r,source_files:serde_json::json!({"/tmp/initialized-source-fixture":{"bytes":1,"sha256":"ab".repeat(32)}})};
        (trainer,recipe,context)
    }
    #[test]fn initialized_all_state_bits_and_native_roundtrip(){
        let (trainer,recipe,context)=fixture();
        let bits=initialized_checkpoint_bytes(&trainer,recipe,&context).unwrap();
        verify_initialized_readback(&trainer,recipe,&context,&bits).unwrap();
        let native=initialized_native_bytes(&trainer,recipe).unwrap();verify_initialized_native(&trainer,recipe,&native).unwrap();
        assert!(completed_checkpoint_bytes(&trainer,recipe,&context).is_err());
    }
    #[test]fn initialized_unknown_partial_and_changed_bits_rejected(){
        let (trainer,recipe,context)=fixture();
        let raw=initialized_checkpoint_bytes(&trainer,recipe,&context).unwrap();
        let original=bound::unique_json(&raw).unwrap();
        for key in ["step","out_bias","obias_m","obias_v"]{
            let mut v=original.clone();v["state"][key]=serde_json::json!(1);
            assert!(verify_initialized_readback(&trainer,recipe,&context,&bound::json_bytes(&v).unwrap()).is_err());
        }
        let mut v=original;v["unexpected"]=serde_json::json!(true);
        assert!(verify_initialized_readback(&trainer,recipe,&context,&bound::json_bytes(&v).unwrap()).is_err());
    }
}

// Zero bytes are legal only in raw SOURCE identity maps. Semantic document refs
// above retain positive fullref validation, and JSON/O/native parsing is unchanged.
fn validate_raw_source_identity(path:&str,info:&Value)->Result<(),String>{
    let i=bound::object(info,&["bytes","sha256"])?;bound::absolute(path)?;
    let bytes=i["bytes"].as_u64().ok_or("strict integer source bytes required")?;
    bound::bounded_file_bytes(bytes)?;
    bound::sha_string(i["sha256"].as_str().ok_or("source SHA string required")?)?;Ok(())
}
#[cfg(test)]mod initialized_zero_source_tests{use super::*;
    #[test]fn initialized_source_zero_bytes_are_strict_raw_identity_only(){
        let good=serde_json::json!({"bytes":0,"sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"});
        assert!(validate_raw_source_identity("/tmp/empty-source",&good).is_ok());
        assert!(bound::fullref(&serde_json::json!({"path":"/tmp/meaningful-document","bytes":0,"sha256":good["sha256"]})).is_err());
        for bad in [serde_json::json!({"bytes":0.0,"sha256":good["sha256"]}),serde_json::json!({"bytes":false,"sha256":good["sha256"]}),serde_json::json!({"bytes":0,"sha256":"bad"}),serde_json::json!({"bytes":0,"sha256":good["sha256"],"unknown":0})]{assert!(validate_raw_source_identity("/tmp/empty-source",&bad).is_err());}
    }
}
