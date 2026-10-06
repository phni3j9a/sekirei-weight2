//! SOURCE ONLY own first candidate raw master / nearest03 diagnostic.
//! This module is a trainer child, never a resume or training entry.
use super::{TrainWeights,active_features,L1,L2};
use super::paired_nonlinear_checkpoint_io as checkpoint;
use super::paired_nonlinear_checkpoint_native as native;
use crate::paired_nonlinear_bound_io as bound;
use crate::paired_nonlinear_cli as cli;
use crate::paired_nonlinear_float as fp;
use sekirei_core::{board::Board,sfen::board_to_sfen,nnue::{NnueWeights,read_weights}};
use serde_json::{Value,json};
use std::{collections::{BTreeMap,HashSet},path::PathBuf};

const PROTOTYPE_ONLY:bool=true;
const MODE:&str="white-view-diverse-games-seed42-e3-v1";
const POLICY:&str="ascending floor(i*(N-1)/255), i=0..255, original split row order";
const MASTER:&str="7c2f9793bc1e3660f0202631cce2c3ae1c85b6cf7293e9285e16e60a60772f88";
const NATIVE:&str="44d1b412da22688542a406599dd40b40bc3243285aaaeae67b21d003db9a0429";
fn runtime_guard()->Result<(),String>{
    bound::require(!PROTOTYPE_ONLY,"SOURCE ONLY raw master diagnostic disabled before I/O")?;
    bound::require(cfg!(feature="nnue_white_view_aux_tied")&&(L1,L2)==(256,32),"fixed White architecture required")?;
    fp::runtime_ready()?;Ok(())
}

// FINITE OBSERVATION: no fused multiply-add, reassociation or replacement loss.
// Each check returns the original computed value when finite. On failure the
// existing outer Context::fail poisons the position/epoch before any save.
#[inline]
fn checked_finite(value:f32, stage:&'static str)->Result<f32,String> {
    if value.is_finite() {Ok(value)} else {Err(format!("nonfinite paired intermediate: {stage}"))}
}
#[inline]
fn checked_finite_f64(value:f64, stage:&'static str)->Result<f64,String> {
    if value.is_finite() {Ok(value)} else {Err(format!("nonfinite paired observation: {stage}"))}
}
#[inline]
fn checked_add(left:f32, right:f32, stage:&'static str)->Result<f32,String> {
    let sum=left+right;
    checked_finite(sum,stage)
}
#[inline]
fn checked_product(left:f32, right:f32, stage:&'static str)->Result<f32,String> {
    let product=left*right;
    checked_finite(product,stage)
}
#[inline]
fn checked_product_sum(acc:f32, left:f32, right:f32, stage:&'static str)->Result<f32,String> {
    // Validate the raw multiplication before addition can cancel or mask it.
    let product=checked_product(left,right,stage)?;
    checked_add(acc,product,stage)
}

pub fn raw_forward(w:&TrainWeights,board:&Board)->Result<f32,String>{
    let stm=board.side_to_move;
        // FT accumulation
        let mut acc_us = w.ft_bias.clone();
        let mut acc_them = acc_us.clone();

        let active_us = active_features(board, stm);
        let active_them = active_features(board, stm.flip());

        for feat in &active_us {
            let base = feat * L1;
            for j in 0..L1 {
                acc_us[j] = checked_add(acc_us[j],w.ft[base+j],"forward.ft.us.add")?;
            }
        }
        for feat in &active_them {
            let base = feat * L1;
            for j in 0..L1 {
                acc_them[j] = checked_add(acc_them[j],w.ft[base+j],"forward.ft.them.add")?;
            }
        }

        // FT ClippedReLU [0, 127]
        let relu_us: Vec<f32> = acc_us.iter().map(|&x| x.clamp(0.0, 127.0)).collect();
        let relu_them: Vec<f32> = acc_them.iter().map(|&x| x.clamp(0.0, 127.0)).collect();
        // L2 accumulation
        let mut l2_acc = w.l2_bias.clone(); // Vec<f32> len=L2
        for j in 0..L1 {
            let a = relu_us[j];
            let b = relu_them[j];
            let base_us = j * L2;
            let base_them = (L1 + j) * L2;
            for o in 0..L2 {
                l2_acc[o] = checked_product_sum(l2_acc[o],a,w.l2[base_us+o],"forward.l2.us.mul-add")?;
                l2_acc[o] = checked_product_sum(l2_acc[o],b,w.l2[base_them+o],"forward.l2.them.mul-add")?;
            }
        }

        // L2 ClippedReLU [0, 127]
        let relu_l2: Vec<f32> = l2_acc.iter().map(|&x| x.clamp(0.0, 127.0)).collect();
        // Output
        let mut output = w.out_bias;
        for o in 0..L2 {
            output = checked_product_sum(output,relu_l2[o],w.out[o],"forward.output.mul-add")?;
        }
        let score = checked_finite(output/64.0,"forward.score")?;
    Ok(score)
}


fn check_prefixes(board:&Board,w:&NnueWeights)->Result<(),String>{
    for perspective in [board.side_to_move,board.side_to_move.flip()]{
        let mut sum=[0_i32;L1];
        for j in 0..L1 {sum[j]=w.ft_bias[j] as i32;}
        for feature in active_features(board,perspective){
            for j in 0..L1 {sum[j]+=w.ft[feature][j] as i32;
                bound::require((-32768..=32767).contains(&sum[j]),"native FT prefix saturation")?;}
        }
    }Ok(())
}
fn string<'a>(v:&'a Value)->Result<&'a str,String>{v.as_str().ok_or("string required".into())}
fn input<'a>(s:&'a BTreeMap<PathBuf,bound::Snapshot>,refs:&Value,name:&str)->Result<&'a [u8],String>{
    let r=bound::fullref(&refs[name])?;let snap=s.get(&r.path).ok_or("missing bound input")?;
    bound::require(bound::same_ref(&snap.reference,&r),"raw input rebound")?;Ok(&snap.bytes)
}
pub fn run(argv:&[String])->Result<(),String>{
    runtime_guard()?; // before argv path handling, request or checkpoint reads
    bound::require(argv.len()==3,"exact request path/bytes/SHA required")?;
    let bytes=argv[1].parse::<u64>().map_err(|e|e.to_string())?;
    bound::require(bytes.to_string()==argv[1],"canonical size argument")?;
    let request_ref=bound::fullref(&json!({"path":argv[0],"bytes":bytes,"sha256":argv[2]}))?;
    let mut snapshots=BTreeMap::new();bound::insert_bound(&mut snapshots,&request_ref)?;
    let request=bound::unique_json(&snapshots[&request_ref.path].bytes)?;
    let r=bound::object(&request,&["schema","mode","phase","policy","inputs","context","samples"])?;
    bound::require(r["schema"]==json!("sekirei.raw-master-quantization-request.v1")&&r["mode"]==json!(MODE)
        &&r["phase"]==json!("completed-epoch3-step338043")&&r["policy"]==json!(POLICY),"own diagnostic schema/mode/phase/policy required")?;
    let refs=&r["inputs"];
    bound::object(refs,&["completion","master","native03","export_stage","recipe","source_binding","manifest",
        "profile","selected_plan","training_build","reference03","train.positions.jsonl","holdout.positions.jsonl"])?;
    for value in refs.as_object().ok_or("input refs object")?.values(){bound::insert_bound(&mut snapshots,&bound::fullref(value)?)?;}
    bound::require(refs["master"]["sha256"]==json!(MASTER)&&refs["native03"]["sha256"]==json!(NATIVE)
        &&refs["profile"]["sha256"]==json!(crate::weekly_nonlinear_profile::PROFILE_SHA)
        &&refs["manifest"]["sha256"]==json!(crate::weekly_nonlinear_profile::MANIFEST_SHA),"original input identities differ")?;
    let completion=bound::unique_json(input(&snapshots,refs,"completion")?)?;
    bound::require(completion["status"]==json!("complete")&&completion["mode"]==json!(MODE)
        &&completion["poisoned"]==json!(false)&&bound::exact(&completion["outputs"]["checkpoint"],&refs["master"])
        &&bound::exact(&completion["outputs"]["native03"],&refs["native03"])
        &&bound::exact(&completion["outputs"]["export_stage"],&refs["export_stage"]),"completed own outputs required")?;
    for name in ["recipe","source_binding","reference03","training_build"] {
        bound::require(bound::exact(&completion[name],&refs[name]),"completion/input fullref mismatch")?;
    }
    bound::require(bound::exact(&completion["plan"],&refs["selected_plan"]),"completion plan mismatch")?;
    let stage=bound::unique_json(input(&snapshots,refs,"export_stage")?)?;
    bound::require(stage["status"]==json!("three-epochs-export-readback-complete")&&stage["global_step"]==json!(338043)
        &&stage["epochs_completed"]==json!(3)&&stage["positions_per_epoch"]==json!(112681)
        &&stage["resume_allowed"]==json!(false)&&stage["checkpoint_all_state_bits_equal"]==json!(true)
        &&stage["nearest_all_bytes_equal"]==json!(true)&&bound::exact(&stage["context"],&r["context"]),"external export context/end phase differs")?;
    let context:checkpoint::CheckpointContext=serde_json::from_value(r["context"].clone()).map_err(|e|e.to_string())?;
    for (path,info) in context.source_files.as_object().ok_or("source files object")? {
        let source_ref=checkpoint::diagnostic_source_ref(path,info)?;
        bound::insert_bound(&mut snapshots,&source_ref)?;
    }
    let args=cli::Args{recipe:bound::fullref(&context.recipe)?,source_binding:bound::fullref(&context.source_binding)?,
        manifest:bound::fullref(&context.manifest)?,reference03:bound::fullref(&context.reference03)?,
        positions:bound::fullref(&context.positions)?,labels:bound::fullref(&context.labels)?,output:request_ref.path.clone()};
    for (name,value) in [("recipe",&context.recipe),("source_binding",&context.source_binding),("manifest",&context.manifest),
        ("reference03",&context.reference03),("train.positions.jsonl",&context.positions)] {
        bound::require(bound::exact(value,&refs[name]),"external context/input fullref differs")?;
    }
    let recipe=bound::unique_json(input(&snapshots,refs,"recipe")?)?;
    let declaration=cli::selected_recipe(&recipe,&args)?;
    let w=checkpoint::diagnostic_restore_completed(input(&snapshots,refs,"master")?,declaration.recipe,&context)?;
    let before=checkpoint::diagnostic_state_bytes(&w)?;
    let native_raw=input(&snapshots,refs,"native03")?;
    native::verify_native03_reexport(&w,declaration.recipe,native::SnapshotStage::AfterPosition{expected_step:338043},
        native::FEATURE_TAG,native_raw).map_err(str::to_string)?;
    let native_weights=read_weights(&bound::fullref(&refs["native03"])?.path).map_err(|e|e.to_string())?;
    bound::require(native::encode_native03_layout(&native_weights).map_err(str::to_string)?==native_raw,
        "fixed White compiled core native fullbyte readback differs")?;
    let dequant=TrainWeights::from_nnue_weights(&native_weights);
    let manifest=bound::unique_json(input(&snapshots,refs,"manifest")?)?;
    for split in ["train","holdout"]{
        let name=format!("{split}.positions.jsonl");let declared=bound::fullref(&refs[&name])?;
        let manifest_path=bound::fullref(&refs["manifest"])?.path;
        bound::require(declared.path==manifest_path.parent().ok_or("manifest root")?.join(&name)
            &&bound::exact(&manifest["files"][&name],&bound::info_value(&declared)),
            "original manifest split fullref/path membership differs")?;
    }
    let mut games=BTreeMap::new();
    for game in manifest["games"].as_array().ok_or("manifest games")? {
        bound::require(games.insert(string(&game["game_id"])?.to_owned(),string(&game["split"])?.to_owned()).is_none(),"duplicate game")?;
    }
    let samples=r["samples"].as_array().ok_or("sample array")?;
    bound::require(samples.len()==512,"exact train256/frozenholdout256 required")?;
    let mut split_lines=BTreeMap::new();
    for (split,count) in [("train",112681_usize),("holdout",5895_usize)]{
        let name=format!("{split}.positions.jsonl");let raw=input(&snapshots,refs,&name)?;
        bound::require(raw.ends_with(b"\n")&&!raw.contains(&b'\r'),"canonical LF split")?;
        let lines:Vec<_>=raw[..raw.len()-1].split(|&b|b==b'\n').collect();
        bound::require(lines.len()==count&&lines.iter().all(|v|!v.is_empty()),"full physical split count")?;
        split_lines.insert(split,lines);
    }
    let float_before=fp::runtime_ready()?;
    let mut outputs=Vec::new();let mut boards=HashSet::new();
    for (ordinal,sample) in samples.iter().enumerate(){
        let s=bound::object(sample,&["split","row_index","position"])?;
        let (split,count)=if ordinal<256{("train",112681_usize)}else{("holdout",5895_usize)};
        let expected=(ordinal%256)*(count-1)/255;
        bound::require(s["split"]==json!(split)&&s["row_index"].as_u64()==Some(expected as u64),"deterministic split row index required")?;
        let lines=&split_lines[split];
        bound::require(bound::exact(&bound::unique_json(lines[expected])?,&s["position"]),"sample differs from fixed original row")?;
        let row=bound::object(&s["position"],&["schema_version","sfen","source","tags"])?;
        let source=bound::object(&row["source"],&["kind","path","ply"])?;
        let tags=bound::object(&row["tags"],&["phase","side_to_move"])?;
        bound::require(row["schema_version"]==json!(1)&&source["kind"]==json!("gensfen-pack")
            &&games.get(string(&source["path"])?)==Some(&split.to_string())&&tags["phase"]==json!("middlegame"),"fixed source/split/phase")?;
        let sfen=string(&row["sfen"])?;let board=Board::from_sfen_rules_only(sfen).map_err(|e|e.to_string())?;
        bound::require(board_to_sfen(&board)==sfen,"canonical legal SFEN required")?;
        let fields:Vec<_>=sfen.split(' ').collect();
        let ply=source["ply"].as_u64().ok_or("strict source ply")?;
        bound::require(ply>=16&&fields.len()==4&&fields[3]==ply.to_string()
            &&tags["side_to_move"]==json!(if fields[1]=="b"{"black"}else{"white"}),"source ply/STM tags differ")?;
        bound::require(boards.insert(fields[..3].join(" ")),"duplicate sampled semantic board")?;
        fp::runtime_ready()?;check_prefixes(&board,&native_weights)?;
        let raw_cp=raw_forward(&w,&board)?;let dequant_cp=raw_forward(&dequant,&board)?;
        let core_cp=board.evaluate_with_weights(&native_weights);
        bound::require(raw_cp.abs()<899000.0&&dequant_cp.abs()<899000.0&&i64::from(core_cp).abs()<899000,
            "normal/mate boundary diagnostic rejection")?;
        bound::require((dequant_cp as f64-core_cp as f64).abs()<1.001,"native float/integer bridge mismatch")?;
        outputs.push(json!({"split":split,"row_index":expected,"raw_master_cp_f32_bits":format!("{:08x}",raw_cp.to_bits()),
            "native_dequant_cp_f32_bits":format!("{:08x}",dequant_cp.to_bits()),"native_core_cp":core_cp,
            "all_state_bits_unchanged":true}));
    }
    let float_after=fp::runtime_ready()?;bound::verify_all(&snapshots)?;
    bound::require(checkpoint::diagnostic_state_bytes(&w)?==before,"final full state bits changed")?;
    // Delay all row output until every physical source/raw/state check passes.
    for row in outputs{println!("{}",serde_json::to_string(&row).map_err(|e|e.to_string())?);}
    eprintln!("float_before={}",serde_json::to_string(&float_before).map_err(|e|e.to_string())?);
    eprintln!("float_after={}",serde_json::to_string(&float_after).map_err(|e|e.to_string())?);
    eprintln!("complete raw-master/native sampled count512; step338043 unchanged; optimizer_updates0; no teacher/search/export write");
    Ok(())
}

#[cfg(test)] mod tests {
    use super::*;
    #[test]fn entry_closed_before_io(){assert!(runtime_guard().is_err());}
    #[test]fn finite_checks_before_clamp(){
        assert!(checked_add(f32::MAX,f32::MAX,"sum").is_err());
        assert!(checked_product_sum(0.0,f32::MAX,2.0,"multiply").is_err());
        assert_eq!(checked_add(-0.0,-0.0,"zero").unwrap().to_bits(),0x80000000);
    }
    #[test]fn cp_divisor_interleaved_scalar_order_and_all_state_bits_unchanged(){
        // Synthetic memory-only numerical kernel control, not a valid candidate.
        // Constant FT activations make a deliberately nonassociative L2 sum:
        // bias1e20 + us0*0 + them0*(-1e20) + us1*3 == 3.
        // Collecting all us terms before them would lose the 3 and return zero.
        let reference=native::canonical_reference_native03().unwrap();
        let mut w=TrainWeights::from_nnue_weights(&reference);
        w.ft.fill(0.0);w.ft_bias.fill(1.0);w.l2.fill(0.0);w.l2_bias.fill(0.0);
        w.out.fill(0.0);w.out_bias=0.0;
        w.l2_bias[0]=1.0e20;w.l2[L1*L2]=-1.0e20;w.l2[L2]=3.0;w.out[0]=64.0;
        let board=Board::from_sfen_rules_only("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1").unwrap();
        let before=checkpoint::diagnostic_state_bytes(&w).unwrap();
        assert_eq!(raw_forward(&w,&board).unwrap().to_bits(),3.0_f32.to_bits());
        assert_eq!(checkpoint::diagnostic_state_bytes(&w).unwrap(),before);
        w.l2_bias.fill(0.0);w.l2.fill(0.0);w.out.fill(0.0);w.out_bias=64.0;
        assert_eq!(raw_forward(&w,&board).unwrap().to_bits(),1.0_f32.to_bits());
    }
    #[test]fn zero_raw_source_identity_kept_separate_from_positive_document_fullref(){
        let info=json!({"bytes":0,"sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"});
        assert_eq!(checkpoint::diagnostic_source_ref("/tmp/empty-source",&info).unwrap().bytes,0);
        assert!(bound::fullref(&json!({"path":"/tmp/meaningful-document","bytes":0,"sha256":info["sha256"]})).is_err());
        for value in [json!({"bytes":false,"sha256":info["sha256"]}),json!({"bytes":0.0,"sha256":info["sha256"]}),
            json!({"bytes":0,"sha256":"bad"}),json!({"bytes":0,"sha256":info["sha256"],"unknown":0})]{
            assert!(checkpoint::diagnostic_source_ref("/tmp/empty-source",&value).is_err());
        }
    }
}
