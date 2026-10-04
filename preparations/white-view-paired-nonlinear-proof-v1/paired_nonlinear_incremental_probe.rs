//! Technical fixture/random-walk checks against the fixed core; no search or learning.
//! SOURCE ONLY dedicated nonlinear native03; old bounded-material source unchanged.
mod paired_nonlinear_native_contract;
use paired_nonlinear_native_contract as contract;
const PROTOTYPE_ONLY:bool=true;
#[derive(Default)]
struct Observed{positions:usize,l2_products:usize,l2_additions:usize,output_products:usize,output_additions:usize,max_abs_core:i64,max_abs_float:f32,max_bridge:f64}
fn ordinary(cp:f32)->bool{cp.is_finite()&&cp.abs()<899_000.0}
fn runtime_guard()->Result<(),&'static str>{if PROTOTYPE_ONLY{Err("SOURCE ONLY incremental probe disabled before I/O")}else{Ok(())}}
use sekirei_core::{board::Board, eval, movegen, nnue, sfen};
use std::{env, path::Path};

fn snapshot(board: &Board) -> (String, u64, usize, u32, [[i16; nnue::L1]; 2]) {
    (
        sfen::board_to_sfen(board),
        board.hash(),
        board.side_to_move.index(),
        board.ply,
        board.acc.values,
    )
}

fn finite_forward(board:&Board,weights:&nnue::NnueWeights,observed:&mut Observed)->f32{
    let us = board.side_to_move.index();
    let them = 1 - us;
    let mut l2 = weights.l2_bias;
    assert!(l2.iter().all(|x| x.is_finite()));
    for j in 0..nnue::L1 {
        let a = board.acc.values[us][j].clamp(0, 8128) as f32 / 64.0;
        let b = board.acc.values[them][j].clamp(0, 8128) as f32 / 64.0;
        for o in 0..nnue::L2 {
            let p = a * weights.l2[j][o];
            let q = b * weights.l2[nnue::L1 + j][o];
            assert!(p.is_finite()&&q.is_finite(),"nonfinite L2 product before clamp");observed.l2_products+=2;
            l2[o] += p;
            assert!(l2[o].is_finite(),"nonfinite L2 sum before clamp");observed.l2_additions+=1;
            l2[o] += q;
            assert!(l2[o].is_finite(),"nonfinite L2 sum before clamp");observed.l2_additions+=1;
        }
    }
    let mut output = weights.out_bias;
    assert!(output.is_finite());
    for o in 0..nnue::L2 {
        assert!(l2[o].is_finite(),"nonfinite L2 preactivation before clamp");
        let p=l2[o].clamp(0.0,127.0)*weights.out[o];
        assert!(p.is_finite(),"nonfinite output product");observed.output_products+=1;
        output += p;
        assert!(output.is_finite(),"nonfinite output sum");observed.output_additions+=1;
    }
    let cp = output / 64.0;
    assert!(ordinary(cp),"stored native float enters normal/mate boundary");
    cp
}

#[allow(deprecated)]
fn check(board:&Board,weights:&nnue::NnueWeights,observed:&mut Observed)->i64{
    let mut fresh = board.clone();
    fresh.refresh_acc();
    assert_eq!(
        board.acc.values, fresh.acc.values,
        "incremental accumulator differs from refresh"
    );
    let rebuilt = board.evaluate_with_weights(weights);
    let float=finite_forward(board,weights,observed);
    assert!(i64::from(rebuilt).abs()<899_000,"core integer enters normal/mate boundary");
    assert_eq!(float as i32,rebuilt,"finite diagnostic differs from fixed core");
    let bridge=(float as f64-f64::from(rebuilt)).abs();assert!(bridge<1.001,"stored native float/core bridge differs");
    observed.positions+=1;observed.max_abs_core=observed.max_abs_core.max(i64::from(rebuilt).abs());
    observed.max_abs_float=observed.max_abs_float.max(float.abs());observed.max_bridge=observed.max_bridge.max(bridge);
    #[cfg(target_arch="x86_64")]
    unsafe{assert_eq!(std::arch::x86_64::_mm_getcsr()&!0x3f,0x9fc0,"float control changed during incremental work");}
    let incremental = eval::evaluate(board);
    assert_eq!(
        incremental, rebuilt,
        "incremental NNUE differs from refresh"
    );
    let difference = (i64::from(rebuilt) - i64::from(eval::material_score(board))).abs();
    difference
}

#[allow(deprecated)]
fn main(){
    runtime_guard().expect("disabled SOURCE entry"); // before hardware/args/files

    #[cfg(target_arch = "x86_64")]
    unsafe {
        let old=std::arch::x86_64::_mm_getcsr();
        assert_eq!(old&0x6000,0,"RoundNearest required");assert_eq!(old&0x1f80,0x1f80,"exception masks differ");
        let target=old|0x8040;assert_eq!(target&!0x3f,0x9fc0,"unsupported float control");
        std::arch::x86_64::_mm_setcsr(target);assert_eq!(std::arch::x86_64::_mm_getcsr()&!0x3f,0x9fc0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    panic!("this pinned probe requires x86_64 FTZ/DAZ policy");
    let args: Vec<_> = env::args().collect();
    assert_eq!(args.len(),4,"usage: nonlinear_incremental CANDIDATE.nearest03 REFERENCE03 FIXTURE_TSV");
    let candidate_raw=std::fs::read(&args[1]).expect("candidate raw native03");
    let reference_raw=std::fs::read(&args[2]).expect("reference raw native03");
    for raw in [&candidate_raw,&reference_raw]{assert_eq!(raw.len(),contract::NATIVE_BYTES);assert_eq!(&raw[..8],contract::MAGIC);}
    let reference=nnue::read_weights(Path::new(&args[2])).expect("read reference03");
    assert_eq!(contract::native03_layout_bytes(&reference).unwrap(),reference_raw,"compiled reference load differs from raw bytes");
    nnue::load_weights(Path::new(&args[1])).expect("load weight");
    assert!(nnue::weights_active());
    let weights=nnue::weights();
    assert_eq!(contract::native03_layout_bytes(weights).unwrap(),candidate_raw,"compiled global candidate load differs from raw bytes");
    let safety=contract::validate_pair(weights,&reference).expect("selected nonlinear native parameters");
    let mut observed=Observed::default();
    eval::set_nnue_output_mode(eval::NnueOutputMode::Absolute);
    let (mut positions, mut fixtures, mut max_difference) = (0usize, 0usize, 0i64);
    for line in std::fs::read_to_string(&args[3])
        .expect("public fixture TSV")
        .lines()
    {
        let (score, sfen) = line
            .split_once('\t')
            .expect("expected material cp TAB SFEN");
        let board = Board::from_sfen(sfen).expect("fixture SFEN");
        assert_eq!(
            eval::material_score(&board),
            score.parse::<i32>().expect("fixture cp")
        );
        max_difference = max_difference.max(check(&board,weights,&mut observed));
        positions += 1;
        fixtures += 1;
    }
    assert_eq!(fixtures, 15, "expected the fixed public technical fixtures");
    let (mut captures, mut promotions, mut drops, mut undos) = (0usize, 0usize, 0usize, 0usize);
    let mut rng = 42u64;
    let mut null_undos = 0usize;
    for walk in 0..16 {
        let search_path = walk % 2 == 1;
        let mut board = Board::startpos();
        let mut history = Vec::new();
        max_difference = max_difference.max(check(&board,weights,&mut observed));
        positions += 1;
        for step in 0..256 {
            if step % 17 == 0 {
                let before = snapshot(&board);
                let token = board.do_null_move();
                max_difference = max_difference.max(check(&board,weights,&mut observed));
                board.undo_null_move(token);
                assert_eq!(
                    snapshot(&board),
                    before,
                    "null undo failed to restore parent"
                );
                max_difference = max_difference.max(check(&board,weights,&mut observed));
                positions += 2;
                null_undos += 1;
            }
            let moves = movegen::generate_legal_moves(&mut board);
            if moves.is_empty() {
                break;
            }
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let m = moves[(rng as usize) % moves.len()];
            captures += usize::from(board.piece_at(m.to).is_some());
            promotions += usize::from(m.promote);
            drops += usize::from(m.is_drop());
            let before = snapshot(&board);
            let token = if search_path {
                board.do_move_for_search(m)
            } else {
                board.do_move(m)
            };
            history.push((token, before));
            max_difference = max_difference.max(check(&board,weights,&mut observed));
            positions += 1;
        }
        while let Some((token, before)) = history.pop() {
            if search_path {
                board.undo_move_for_search(token);
            } else {
                board.undo_move(token);
            }
            assert_eq!(
                snapshot(&board),
                before,
                "move undo failed to restore parent"
            );
            max_difference = max_difference.max(check(&board,weights,&mut observed));
            positions += 1;
            undos += 1;
        }
    }
    assert!(captures > 0 && promotions > 0 && drops > 0 && undos > 0);
    #[cfg(target_arch = "x86_64")]
    unsafe {
        assert_eq!(std::arch::x86_64::_mm_getcsr()&!0x3f,0x9fc0);
    }
    assert_eq!(positions,8185,"fixed observation scope differs");assert_eq!(observed.positions,positions);
    assert_eq!(observed.l2_products,positions*16384);assert_eq!(observed.l2_additions,positions*16384);
    assert_eq!(observed.output_products,positions*32);assert_eq!(observed.output_additions,positions*32);
    assert!((observed.max_abs_float as f64)<=safety.absolute_cp_upper,"selected native output bound differs");
    println!(concat!("{{\"positions_checked\":{},\"fixtures\":{},\"walks\":16,\"search_walks\":8,",
        "\"captures\":{},\"promotions\":{},\"drops\":{},\"undos\":{},\"null_undos\":{},",
        "\"max_material_difference_cp\":{},\"incremental_refresh_error_cp\":0,\"accumulator_refresh_equal\":true,\"parent_restoration_equal\":true,",
        "\"observed_float_intermediates_finite\":true,\"all_preclamp_intermediates_finite\":true,\"mxcsr_control\":\"9fc0\",\"material_bound_enabled\":false,",
        "\"normal_score_domain_verified\":true,\"native_structure_verified\":true,\"native_loaded_fullbytes_equal\":true,",
        "\"max_abs_core_cp\":{},\"max_abs_quantized_float_cp\":{:?},\"maximum_stored_float_core_bridge_cp\":{:?},",
        "\"l2_products_checked\":{},\"l2_additions_checked\":{},\"output_products_checked\":{},\"output_additions_checked\":{},",
        "\"native_q_l1_upper\":{:?},\"native_absolute_cp_upper\":{:?}}}"),
        positions,fixtures,captures,promotions,drops,undos,null_undos,max_difference,
        observed.max_abs_core,observed.max_abs_float,observed.max_bridge,observed.l2_products,observed.l2_additions,
        observed.output_products,observed.output_additions,safety.q_l1_upper,safety.absolute_cp_upper);
}
#[cfg(test)]mod tests{use super::*;
 #[test]fn domain_and_entry_are_strict(){assert!(runtime_guard().is_err());assert!(ordinary(898999.0));for v in [899000.0,-899000.0,f32::INFINITY,f32::NAN]{assert!(!ordinary(v));}}
}
