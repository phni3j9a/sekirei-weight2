// Independent static probe; build only against pinned default sekirei-core.
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
    if policy != "x86-ftz-daz" && policy != "gradual" {
        return Err("float policy must be x86-ftz-daz or gradual".into());
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::{_mm_getcsr, _mm_setcsr};
        let original = _mm_getcsr();
        if original & 0x6000 != 0 {
            return Err("MXCSR is not round-to-nearest".into());
        }
        let target = if policy == "x86-ftz-daz" {
            original | 0x8040
        } else {
            original & !0x8040
        };
        _mm_setcsr(target);
        if _mm_getcsr() & 0x8040 != target & 0x8040 {
            return Err("MXCSR policy did not apply".into());
        }
        eprintln!(
            "float_subnormal_policy={policy}; mxcsr=0x{:x}",
            _mm_getcsr()
        );
        Ok(())
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Err("this pinned-runtime diagnostic requires x86_64".into())
    }
}

fn checked_float(board: &Board, weights: &NnueWeights) -> Result<f32, String> {
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
    for j in 0..L1 {
        let a = (sums[us][j] as f32 / 64.0).clamp(0.0, 127.0);
        let b = (sums[them][j] as f32 / 64.0).clamp(0.0, 127.0);
        for o in 0..L2 {
            l2[o] += a * weights.l2[j][o];
            l2[o] += b * weights.l2[L1 + j][o];
        }
    }
    let mut out = weights.out_bias;
    for o in 0..L2 {
        out += l2[o].clamp(0.0, 127.0) * weights.out[o];
    }
    let cp = out / 64.0;
    if !cp.is_finite() {
        return Err("nonfinite float prediction".to_string());
    }
    Ok(cp)
}

fn run() -> Result<(), String> {
    if (INPUT, L1, L2) != (2420, 256, 32) {
        return Err("wrong compiled architecture".into());
    }
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        return Err(
            "usage: core_pair_probe native.bin nearest.bin x86-ftz-daz|gradual < sfens.txt".into(),
        );
    }
    configure_floats(&args[3])?;
    let old = read_weights(Path::new(&args[1])).map_err(|e| e.to_string())?;
    let new = read_weights(Path::new(&args[2])).map_err(|e| e.to_string())?;
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
        let native_float =
            checked_float(&board, &old).map_err(|e| format!("native index {index}: {e}"))?;
        let nearest_float =
            checked_float(&board, &new).map_err(|e| format!("nearest index {index}: {e}"))?;
        if (native_float as f64 - native_cp as f64).abs() >= 1.001
            || (nearest_float as f64 - nearest_cp as f64).abs() >= 1.001
        {
            return Err(format!("core bridge >=1.001cp at index {index}"));
        }
        println!(
            "{{\"index\":{index},\"native_core_cp\":{native_cp},\"nearest_core_cp\":{nearest_cp},\"native_quantized_float_cp\":{native_float},\"nearest_quantized_float_cp\":{nearest_float},\"material_cp\":{}}}",
            material_score(&board)
        );
        count += 1;
    }
    if count == 0 {
        return Err("empty position set".into());
    }
    eprintln!("complete count={count}; absolute STM; checked both float bridges and FT prefixes");
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("failed: {error}");
        std::process::exit(1);
    }
}
