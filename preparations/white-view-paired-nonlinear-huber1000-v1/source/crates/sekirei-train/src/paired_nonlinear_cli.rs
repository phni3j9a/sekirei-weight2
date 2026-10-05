//! SOURCE ONLY dedicated CLI; old Args/parser/epoch/save paths are untouched.
//! Reader/native03/optimizer serialization is an explicit unimplemented trait.
//! Never enable actual_entry merely by deleting runtime_guard: finish reader,
//! source/manifest/replay/legal guards and dedicated save/proof first.
use std::collections::HashMap;
use std::path::{Path,PathBuf,Component};
use serde_json::Value;
use crate::trainer::{Trainer,TrainWeights,shuffled_order};
use crate::trainer::paired_nonlinear_adapter::{self as paired,DeclaredRecipe};
use crate::trainer::paired_nonlinear_positions::{Context,EpochToken};

pub const PROTOTYPE_ONLY:bool=false;
pub const MODE:&str=crate::weekly_nonlinear_profile::MODE;
pub const FEATURE_SCHEMA:&str="flat_white_view_aux_tied_v1";
pub fn runtime_guard()->Result<(),String> {
    crate::paired_nonlinear_float::runtime_ready().map(|_|())
}

#[derive(Clone,Debug)]
pub struct FullRef { pub path:PathBuf, pub bytes:u64, pub sha256:String }
fn sha(s:&str)->Result<String,String> {
    if s.len()!=64 || !s.bytes().all(|b|b.is_ascii_digit()||(b'a'..=b'f').contains(&b)) {
        return Err("strict lower hex SHA256 required".into());
    }
    Ok(s.to_string())
}
fn absolute(s:&str)->Result<PathBuf,String> {
    let p=PathBuf::from(s);
    if !p.is_absolute() || p.to_str()!=Some(s) || s.contains('\0')
        || s.split('/').skip(1).any(|part|part.is_empty()||part=="."||part=="..")
        || p.components().any(|v|matches!(v,Component::ParentDir|Component::CurDir)) {
        return Err("canonical absolute declaration path required".into());
    }
    Ok(p)
}
fn uint(s:&str)->Result<u64,String> {
    if s.is_empty()||!s.bytes().all(|b|b.is_ascii_digit())|| (s.len()>1&&s.starts_with('0')) {
        return Err("canonical unsigned decimal required".into());
    }
    s.parse::<u64>().map_err(|e|e.to_string())
}
#[derive(Debug)]
pub struct Args {
    pub recipe:FullRef,pub source_binding:FullRef,pub manifest:FullRef,
    pub reference03:FullRef,pub positions:FullRef,pub labels:FullRef,
    pub output:PathBuf,
}
pub fn parse(argv:&[String])->Result<Args,String> {
    let names=["recipe","source-binding","manifest","reference03","positions","labels"];
    if argv.len()!=38 {return Err("exact 19 flag/value pairs required; no defaults/resume/legacy flags".into());}
    let mut values=HashMap::new();
    for pair in argv.chunks_exact(2) {
        if values.insert(pair[0].as_str(),pair[1].as_str()).is_some() {return Err("duplicate CLI option".into());}
    }
    let mut expected=std::collections::HashSet::new();
    for name in names {for suffix in ["","-bytes","-sha256"] {expected.insert(format!("--{name}{suffix}"));}}
    expected.insert("--output".to_string());
    if values.len()!=expected.len()||values.keys().any(|k|!expected.contains(*k)) {return Err("unknown/missing dedicated CLI option".into());}
    let get=|name:&str|->Result<FullRef,String>{
        Ok(FullRef {path:absolute(values[format!("--{name}").as_str()])?,
            bytes:uint(values[format!("--{name}-bytes").as_str()])?,
            sha256:sha(values[format!("--{name}-sha256").as_str()])?})
    };
    let result=Args {recipe:get("recipe")?,source_binding:get("source-binding")?,manifest:get("manifest")?,
        reference03:get("reference03")?,positions:get("positions")?,labels:get("labels")?,
        output:absolute(values["--output"])?};
    let mut seen=std::collections::HashSet::new();
    for bound in [&result.recipe,&result.source_binding,&result.manifest,&result.reference03,&result.positions,&result.labels] {
        if !seen.insert(bound.path.clone()) || result.output.starts_with(&bound.path)
            || bound.path.starts_with(&result.output) {return Err("duplicate input or overlapping output declaration".into());}
    }
    Ok(result)
}
fn obj<'a>(v:&'a Value,keys:&[&str])->Result<&'a serde_json::Map<String,Value>,String> {
    let o=v.as_object().ok_or("object required")?;
    if o.len()!=keys.len()||keys.iter().any(|k|!o.contains_key(*k)) {return Err("typed document exact keys differ".into());}
    Ok(o)
}
fn fullref(v:&Value)->Result<FullRef,String> {
    let o=obj(v,&["path","bytes","sha256"])?;
    Ok(FullRef{path:absolute(o["path"].as_str().ok_or("path string required")?)?,
        bytes:o["bytes"].as_u64().ok_or("strict nonnegative integer bytes required")?,
        sha256:sha(o["sha256"].as_str().ok_or("sha string required")?)?})
}
fn same_ref(a:&FullRef,b:&FullRef)->bool {a.path==b.path&&a.bytes==b.bytes&&a.sha256==b.sha256}
fn bits(v:&Value)->Result<f32,String> {
    let s=v.as_str().ok_or("explicit f32-bit hex string required")?;
    if s.len()!=8||!s.bytes().all(|b|b.is_ascii_digit()||(b'a'..=b'f').contains(&b)) {return Err("canonical f32 bit string required".into());}
    let x=f32::from_bits(u32::from_str_radix(s,16).map_err(|e|e.to_string())?);
    if !x.is_finite() {return Err("nonfinite selected scalar".into());} Ok(x)
}
pub struct SelectedDeclaration {
    pub recipe:DeclaredRecipe,
    pub shuffle_seed:Option<u64>,pub teacher_identity:String,
}
/// Memory-only strict schema. raw bytes/hash/duplicate JSON keys are the
/// future Reader's separate mandatory check; serde Value alone cannot prove it.
pub fn selected_recipe(v:&Value,args:&Args)->Result<SelectedDeclaration,String> {
    let o=obj(v,&["schema","mode","feature_schema","seed","epochs","train_count","objective","huber_delta_cp_f32_bits","optimizer",
        "learning_rate_f32_bits","head_init_width_f32_bits","head_bias_init_f32_bits","output_native_l1_budget_f32_bits",
        "shuffle_seed","teacher_identity","manifest","reference03","positions","labels","source_binding",
        "resume_allowed","ft_saved_q_max","ft_bias_q"])?;
    if o["schema"].as_str()!=Some("sekirei.white-view-paired-nonlinear-huber1000-recipe.v1")
        ||o["mode"].as_str()!=Some(MODE)||o["feature_schema"].as_str()!=Some(FEATURE_SCHEMA)
        ||o["seed"].as_u64()!=Some(42)||o["epochs"].as_u64()!=Some(3)||o["train_count"].as_u64()!=Some(112681)
        ||MODE!="white-view-diverse-games-seed42-huber1000-e3-v1"||o["objective"].as_str()!=Some("absolute-cp-twice-huber")
        ||o["huber_delta_cp_f32_bits"].as_str()!=Some("447a0000")||o["optimizer"].as_str()!=Some("fresh-adam-tied-masters")
        ||o["resume_allowed"].as_bool()!=Some(false)||o["ft_saved_q_max"].as_u64()!=Some(797)
        ||o["ft_bias_q"].as_u64()!=Some(64) {return Err("fixed recipe invariants differ".into());}
    for (key,bound) in [("manifest",&args.manifest),("reference03",&args.reference03),
        ("positions",&args.positions),("labels",&args.labels),("source_binding",&args.source_binding)] {
        if !same_ref(&fullref(&o[key])?,bound) {return Err("recipe/CLI external fullref mismatch".into());}
    }
    let lr=bits(&o["learning_rate_f32_bits"])?;
    let width=bits(&o["head_init_width_f32_bits"])?;let bias=bits(&o["head_bias_init_f32_bits"])?;
    let budget=bits(&o["output_native_l1_budget_f32_bits"])?;
    if [lr.to_bits(),width.to_bits(),bias.to_bits(),budget.to_bits()] != [0x3a83126f,0x3b800000,0x40800000,0x47000000]
        || !o["shuffle_seed"].is_null() {return Err("selected fixed plan bits/no-shuffle differ".into());}
    let recipe=DeclaredRecipe {learning_rate:lr,head_init_width:width,
        head_bias_init:bias,output_native_l1_budget:budget};
    paired::validate_recipe(recipe).map_err(str::to_string)?;
    let shuffle_seed=if o["shuffle_seed"].is_null(){None}else{Some(o["shuffle_seed"].as_u64().ok_or("explicit shuffle integer or null required")?)};
    let teacher=o["teacher_identity"].as_str().ok_or("teacher identity string required")?;
    if !teacher.starts_with("external:")||teacher.len()<=9 {return Err("original O external teacher identity required".into());}
    Ok(SelectedDeclaration{recipe,shuffle_seed,teacher_identity:teacher.to_string()})
}

/// Future actual reader must verify raw SHA before parse (duplicates rejected),
/// before/after full input/source identity, original O manifest/replay/legal/
/// exclusions, cache four storage fields and SFEN complete join, new03 source/
/// core ABI and reference bytes before returning. No implementation exists.
pub struct VerifiedTraining {
    pub trainer:Trainer,pub reference:Option<TrainWeights>,
    pub samples:Vec<crate::positions::PositionSample>,pub cache:HashMap<String,i32>,
    pub declaration:SelectedDeclaration,
}
pub trait ActualReaderAndDedicatedExporter {
    fn verify_and_load(&mut self,args:&Args)->Result<VerifiedTraining,String>;
    /// Dedicated nearest03 and complete master optimizer schema ONLY. Never
    /// calls old to_nnue_weights/truncating serializer/save_resume_checkpoint.
    /// Commit after ALL THREE tokens; partial epoch state must never be saved.
    fn commit_completed_native03(&mut self,args:&Args,train:&VerifiedTraining,
                                tokens:&[EpochToken])->Result<(),String>;
}
/// The Result chain is concrete. Actual activation remains closed; the separate
/// ActualReader entry uses this fixed-recipe chain after typed body validation.
pub fn run_dedicated(args:&Args,io:&mut dyn ActualReaderAndDedicatedExporter)->Result<(),String> {
    runtime_guard()?; // actual Reader is unreachable until a separate enabled integration
    let mut train=io.verify_and_load(args)?;
    if train.samples.len()!=112681 || train.trainer.paired_optimizer_step()!=0 {return Err("fresh original TRAIN 112681 required".into());}
    let decl=&train.declaration;
    let first=decl.recipe; // one externally declared LR; no default schedule/retuning
    // State/init is supplied already validated by the reader; never use old
    // Trainer::new random weights as the declared material reference03.
    let reference=train.reference.take().ok_or("verified reference03 missing")?;
    let mut context=Context::from_verified_declaration(reference,first)?;
    let mut tokens=Vec::new();
    for epoch in 1..=3 {
        let fp=crate::paired_nonlinear_float::snapshot()?;
        eprintln!("PAIRED_EPOCH_BEGIN {}",serde_json::json!({"epoch":epoch,"start_step":train.trainer.paired_optimizer_step(),"float":fp}));
        train.trainer.lr=decl.recipe.learning_rate;
        let shuffled;
        let samples=if let Some(seed)=decl.shuffle_seed {
            let order=shuffled_order(train.samples.len(),seed ^ epoch as u64);
            shuffled=order.iter().map(|&i|train.samples[i].clone()).collect::<Vec<_>>();
            shuffled.as_slice()
        }else {train.samples.as_slice()};
        let token=train.trainer.train_paired_nonlinear_epoch(samples,&train.cache,epoch,&mut context)?;
        let fp=crate::paired_nonlinear_float::snapshot()?;
        eprintln!("PAIRED_EPOCH_COMPLETE {}",serde_json::json!({"epoch":token.epoch(),"positions":token.positions(),"end_step":token.end_step(),"float":fp}));
        tokens.push(token);
    }
    context.ensure_live()?;
    // reference is still needed by dedicated export; transfer it back via a
    // checked Context method (not present in original optimizer/checkpoint).
    train.reference=Some(context.into_completed_reference()?);
    io.commit_completed_native03(args,&train,&tokens)?;
    Ok(())
}

pub fn actual_entry(_argv:&[String])->Result<(),String> {
    runtime_guard()?; // MUST precede parse/read/load/training/export.
    Err("Use the separate ActualReader entry after Root activation; no stub success".into())
}

// Candidate-only strict declaration fixtures. No I/O or real provenance claims.
#[cfg(test)]
mod paired_nonlinear_huber1000_recipe_tests {
    use super::*;
    fn fixture()->(Value,Args) {
        let full=|name:&str|FullRef{path:PathBuf::from(format!("/tmp/huber1000-public-fixture/{name}")),bytes:1,sha256:"ab".repeat(32)};
        let args=Args{recipe:full("recipe"),source_binding:full("binding"),manifest:full("manifest"),
            reference03:full("reference03"),positions:full("positions"),labels:full("labels"),
            output:PathBuf::from("/tmp/huber1000-public-fixture/output")};
        let value=serde_json::json!({"schema":"sekirei.white-view-paired-nonlinear-huber1000-recipe.v1",
            "mode":MODE,"feature_schema":FEATURE_SCHEMA,"seed":42,"epochs":3,"train_count":112681,
            "objective":"absolute-cp-twice-huber","huber_delta_cp_f32_bits":"447a0000",
            "optimizer":"fresh-adam-tied-masters","learning_rate_f32_bits":"3a83126f",
            "head_init_width_f32_bits":"3b800000","head_bias_init_f32_bits":"40800000",
            "output_native_l1_budget_f32_bits":"47000000","shuffle_seed":null,
            "teacher_identity":"external:public-fixture-only","resume_allowed":false,"ft_saved_q_max":797,"ft_bias_q":64,
            "manifest":{"path":args.manifest.path,"bytes":1,"sha256":"ab".repeat(32)},
            "reference03":{"path":args.reference03.path,"bytes":1,"sha256":"ab".repeat(32)},
            "positions":{"path":args.positions.path,"bytes":1,"sha256":"ab".repeat(32)},
            "labels":{"path":args.labels.path,"bytes":1,"sha256":"ab".repeat(32)},
            "source_binding":{"path":args.source_binding.path,"bytes":1,"sha256":"ab".repeat(32)}});
        (value,args)
    }
    #[test]
    fn exact_new_variant_is_required() {
        let (value,args)=fixture();assert!(selected_recipe(&value,&args).is_ok());
        for (key,bad) in [("schema",serde_json::json!("sekirei.white-view-paired-nonlinear-recipe.v1")),
            ("objective",serde_json::json!("absolute-cp-mse")),("huber_delta_cp_f32_bits",serde_json::json!("447a0001")),
            ("huber_delta_cp_f32_bits",serde_json::json!(1000)),("huber_delta_cp_f32_bits",serde_json::json!("447A0000")),
            ("mode",serde_json::json!("white-view-diverse-games-seed42-e3-v1"))] {
            let mut changed=value.clone();changed[key]=bad;assert!(selected_recipe(&changed,&args).is_err());
        }
    }
    #[test]
    fn unknown_missing_or_noncanonical_variant_keys_fail() {
        let (value,args)=fixture();
        let mut changed=value.clone();changed.as_object_mut().unwrap().remove("huber_delta_cp_f32_bits");
        assert!(selected_recipe(&changed,&args).is_err());
        let mut changed=value.clone();changed["delta"]=serde_json::json!(1000);
        assert!(selected_recipe(&changed,&args).is_err());
        let mut changed=value;changed["epochs"]=serde_json::json!(true);
        assert!(selected_recipe(&changed,&args).is_err());
    }
}
