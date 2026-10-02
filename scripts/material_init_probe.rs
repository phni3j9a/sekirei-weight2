//! Check material initialization against the pinned core, including incremental moves.
//! Compile with the pinned sekirei_core rlib; this never changes the search binary.
use sekirei_core::{board::Board, eval, movegen, nnue};
use std::{env, path::Path};

fn check(board: &Board, weights: &nnue::NnueWeights) {
    let expected = eval::material_score(board);
    assert_eq!(board.evaluate_with_weights(weights), expected, "rebuilt NNUE differs from material");
    assert_eq!(eval::evaluate(board), expected, "incremental NNUE differs from material");
}

fn main() {
    let path = env::args().nth(1).expect("usage: material-init-probe WEIGHTS");
    let weights = nnue::read_weights(Path::new(&path)).expect("valid weight");
    nnue::load_weights(Path::new(&path)).expect("load weight");
    assert!(nnue::weights_active());
    eval::set_nnue_output_mode(eval::NnueOutputMode::Absolute);
    let mut positions = 0usize;
    let mut fixtures = 0usize;
    if let Some(fixture_path) = env::args().nth(2) {
        for line in std::fs::read_to_string(fixture_path).expect("fixture TSV").lines() {
            let (score, sfen) = line.split_once('\t').expect("expected cp TAB SFEN");
            let board = Board::from_sfen(sfen).expect("fixture SFEN");
            assert_eq!(eval::material_score(&board), score.parse::<i32>().expect("fixture cp"));
            check(&board, &weights);
            positions += 1;
            fixtures += 1;
        }
    }
    let (mut captures, mut promotions, mut drops, mut undos) = (0usize, 0usize, 0usize, 0usize);
    let mut rng = 42u64;
    for _ in 0..16 {
        let mut board = Board::startpos();
        let mut history = Vec::new();
        check(&board, &weights);
        positions += 1;
        for _ in 0..256 {
            let moves = movegen::generate_legal_moves(&mut board);
            if moves.is_empty() { break; }
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let m = moves[(rng as usize) % moves.len()];
            captures += usize::from(board.piece_at(m.to).is_some());
            promotions += usize::from(m.promote);
            drops += usize::from(m.is_drop());
            history.push(board.do_move(m));
            check(&board, &weights);
            positions += 1;
        }
        while let Some(token) = history.pop() {
            board.undo_move(token);
            check(&board, &weights);
            positions += 1;
            undos += 1;
        }
    }
    assert!(captures > 0 && promotions > 0 && drops > 0 && undos > 0);
    println!("{{\"positions_checked\":{positions},\"fixtures\":{fixtures},\"captures\":{captures},\"promotions\":{promotions},\"drops\":{drops},\"undos\":{undos},\"max_error_cp\":0}}");
}
