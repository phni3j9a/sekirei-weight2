// SOURCE ONLY nonlinear pair probe; dedicated pinned native03 core only.
mod paired_nonlinear_native_contract;
use paired_nonlinear_native_contract as contract;
const PROTOTYPE_ONLY:bool=true;
fn runtime_guard()->Result<(),String>{if PROTOTYPE_ONLY{Err("SOURCE ONLY probe disabled before I/O".into())}else{Ok(())}}
// No teacher, search, trainer mutation, global weight loading, or sidecar inference.
use sekirei_core::{
    board::Board,
    color::Color,
    eval::material_score,
    nnue::{INPUT, L1, L2, NnueWeights, feature_index, hand_feature_index, read_weights},
    piece::PieceKind,
    square::Square,
};
use std::{
    env,
    io::{self, BufRead},
    path::Path,
};

#[allow(deprecated)]
fn configure_floats(policy: &str) -> Result<(), String> {
    if policy!="x86-ftz-daz"{return Err("selected nonlinear proof requires x86-ftz-daz".into());}
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::{_mm_getcsr, _mm_setcsr};
        let original = _mm_getcsr();
        if original & 0x6000 != 0 {
            return Err("MXCSR is not round-to-nearest".into());
        }
        if original&0x1f80!=0x1f80{return Err("MXCSR exception masks differ".into());}
        let target=original|0x8040;
        if target&!0x3f!=0x9fc0{return Err("unsupported MXCSR control bits".into());}
        _mm_setcsr(target);
        if _mm_getcsr() & !0x3f != 0x9fc0 {
            return Err("MXCSR policy did not apply".into());
        }
        eprintln!(
            "float_subnormal_policy={policy}; mxcsr_control=0x{:x}; mxcsr_status=0x{:x}",
            _mm_getcsr()&!0x3f,_mm_getcsr()&0x3f
        );
        Ok(())
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Err("this pinned-runtime diagnostic requires x86_64".into())
    }
}

#[derive(Default,Debug)]
struct FloatChecks{l2_products:usize,l2_additions:usize,output_products:usize,output_additions:usize}
fn finite(x:f32,label:&str)->Result<f32,String>{if x.is_finite(){Ok(x)}else{Err(format!("nonfinite before clamp: {label}"))}}
#[allow(deprecated)]
fn verify_float_control()->Result<(),String>{
    #[cfg(target_arch="x86_64")]
    unsafe{if std::arch::x86_64::_mm_getcsr()&!0x3f!=0x9fc0{return Err("MXCSR changed during proof".into());}Ok(())}
    #[cfg(not(target_arch="x86_64"))]
    {Err("fixed x86_64 required".into())}
}
fn checked_float(board: &Board, weights: &NnueWeights) -> Result<(f32,FloatChecks), String> {
    let mut physical=0usize;
    for i in 0..81{if board.piece_at(Square::from_index(i)).is_some(){physical+=1;}}
    for color in [Color::Black,Color::White]{for k in 0..7{physical+=board.hand(color).get(PieceKind::from_u8(k).unwrap())as usize;}}
    if physical>40{return Err("physical piece count exceeds40; prefix proof not applicable".into());}
    let mut checks=FloatChecks::default();
    // Restore quantized FT in exact 1/64 units. Integer sums here are exactly
    // representable as f32 for legal <=40-piece positions. Check every prefix
    // in core refresh order, so i16 saturation cannot be silently hidden.
    let mut sums = [[0i32; L1]; 2];
    for p in 0..2 {
        for j in 0..L1 {
            sums[p][j] = weights.ft_bias[j] as i32;
        }
    }
    let mut add = |p: usize, feature: usize| -> Result<(), String> {
        for j in 0..L1 {
            sums[p][j] += weights.ft[feature][j] as i32;
            if !(-32768..=32767).contains(&sums[p][j]) {
                return Err(format!(
                    "FT i16 prefix saturation: perspective={p}, neuron={j}"
                ));
            }
        }
        Ok(())
    };
    for i in 0..81 {
        let sq = Square::from_index(i);
        if let Some(piece) = board.piece_at(sq) {
            for p in [Color::Black, Color::White] {
                add(p.index(), feature_index(sq, piece.kind, piece.color, p))?;
            }
        }
    }
    for color in [Color::Black, Color::White] {
        for kind_index in 0..7 {
            let kind = PieceKind::from_u8(kind_index).unwrap();
            for n in 1..=board.hand(color).get(kind) {
                for p in [Color::Black, Color::White] {
                    add(p.index(), hand_feature_index(kind, n, color, p))?;
                }
            }
        }
    }
    let us = board.side_to_move.index();
    let them = 1 - us;
    let mut l2 = weights.l2_bias;
    for &bias in &l2{finite(bias,"L2 bias")?;}
    for j in 0..L1 {
        let a = (sums[us][j] as f32 / 64.0).clamp(0.0, 127.0);
        let b = (sums[them][j] as f32 / 64.0).clamp(0.0, 127.0);
        for o in 0..L2 {
            let product=finite(a*weights.l2[j][o],"L2 us product")?;checks.l2_products+=1;
            l2[o]=finite(l2[o]+product,"L2 us sum")?;checks.l2_additions+=1;
            let product=finite(b*weights.l2[L1+j][o],"L2 them product")?;checks.l2_products+=1;
            l2[o]=finite(l2[o]+product,"L2 them sum")?;checks.l2_additions+=1;
        }
    }
    let mut out = finite(weights.out_bias,"output bias")?;
    for o in 0..L2 {
        let before_clamp=finite(l2[o],"L2 preactivation")?;
        let activation=before_clamp.clamp(0.0,127.0);
        let product=finite(activation*weights.out[o],"output product")?;checks.output_products+=1;
        out=finite(out+product,"output sum")?;checks.output_additions+=1;
    }
    let cp = out / 64.0;
    if !cp.is_finite() {
        return Err("nonfinite float prediction".to_string());
    }
    if cp.abs()>=899_000.0{return Err("float enters normal/mate boundary".into());}
    verify_float_control()?;
    Ok((cp,checks))
}

fn run() -> Result<(), String> {
    runtime_guard()?; // absolutely before args, native paths or stdin I/O
    if (INPUT, L1, L2) != (2420, 256, 32) {
        return Err("wrong compiled architecture".into());
    }
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        return Err(
            "usage: paired_nonlinear_core_pair_probe candidate.nearest03.bin reference03.bin x86-ftz-daz < sfens.txt".into(),
        );
    }
    configure_floats(&args[3])?;
    let load=|path:&str|->Result<NnueWeights,String>{
        let raw=std::fs::read(path).map_err(|e|e.to_string())?;
        if raw.len()!=contract::NATIVE_BYTES||&raw[..8]!=contract::MAGIC{return Err("dedicated native03 size/magic required".into());}
        let loaded=read_weights(Path::new(path)).map_err(|e|e.to_string())?;
        if contract::native03_layout_bytes(&loaded).map_err(str::to_string)?!=raw{return Err("actual core-loaded native fullbytes reconstruction differs".into());}
        Ok(loaded)
    };
    let old=load(&args[1])?;let new=load(&args[2])?;
    let safety=contract::validate_pair(&old,&new).map_err(str::to_string)?;
    eprintln!("nonlinear_native_structure=verified; mode={}; epochs={}; q_l1_upper={:?}; absolute_cp_upper={:?}",contract::MODEL_MODE,contract::EPOCHS,safety.q_l1_upper,safety.absolute_cp_upper);
    let mut count = 0usize;
    for (index, line) in io::stdin().lock().lines().enumerate() {
        let sfen = line.map_err(|e| e.to_string())?;
        if sfen.trim().is_empty() {
            return Err(format!("empty input at index {index}"));
        }
        let board =
            Board::from_sfen_rules_only(&sfen).map_err(|e| format!("index {index}: {e}"))?;
        let native_cp = board.evaluate_with_weights(&old);
        let nearest_cp = board.evaluate_with_weights(&new);
        let (native_float,native_checks)=checked_float(&board,&old).map_err(|e|format!("candidate index {index}: {e}"))?;
        let (nearest_float,reference_checks)=checked_float(&board,&new).map_err(|e|format!("reference index {index}: {e}"))?;
        if i64::from(native_cp).abs()>=899_000||i64::from(nearest_cp).abs()>=899_000{return Err(format!("integer normal/mate boundary at index {index}"));}
        if native_float.abs()as f64>safety.absolute_cp_upper{return Err(format!("selected native output bound at index {index}"));}
        if (native_float as f64 - native_cp as f64).abs() >= 1.001
            || (nearest_float as f64 - nearest_cp as f64).abs() >= 1.001
        {
            return Err(format!("core bridge >=1.001cp at index {index}"));
        }
        // Debug float formatting preserves -0.0 as a JSON float; Display would
        // emit -0, which integer-first JSON parsers lose when restoring f32 bits.
        println!(concat!("{{\"index\":{},\"native_core_cp\":{},\"nearest_core_cp\":{},",
            "\"native_quantized_float_cp\":{:?},\"nearest_quantized_float_cp\":{:?},\"material_cp\":{},",
            "\"ft_prefixes_checked\":true,\"all_preclamp_intermediates_finite\":true,",
            "\"candidate_l2_products\":{},\"candidate_l2_additions\":{},\"candidate_output_products\":{},\"candidate_output_additions\":{},",
            "\"reference_l2_products\":{},\"reference_l2_additions\":{},\"reference_output_products\":{},\"reference_output_additions\":{}}}"),
            index,native_cp,nearest_cp,native_float,nearest_float,material_score(&board),
            native_checks.l2_products,native_checks.l2_additions,native_checks.output_products,native_checks.output_additions,
            reference_checks.l2_products,reference_checks.l2_additions,reference_checks.output_products,reference_checks.output_additions);
        count += 1;
    }
    if count == 0 {
        return Err("empty position set".into());
    }
    eprintln!("complete count={count}; nonlinear absolute STM; both stored-native float bridges, FT prefixes, every L2/output product/add finite before clamp");
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("failed: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]mod tests{use super::*;
 #[test]fn finite_failure_is_seen_before_clamp(){assert!(finite(f32::INFINITY,"sum").is_err());assert!(finite(f32::MAX*2.0,"product").is_err());assert!(finite(f32::NAN,"bias").is_err());assert_eq!(finite(-0.0,"zero").unwrap().to_bits(),0x8000_0000);}
 #[test]fn prototype_entry_is_closed(){assert!(runtime_guard().is_err());}
}
