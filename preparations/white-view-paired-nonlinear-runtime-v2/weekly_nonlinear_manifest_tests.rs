//! Synthetic metadata only. No private dataset, model or engine is accessed.
use super::*;

fn info()->Value{serde_json::json!({"bytes":10,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"})}
fn game_id(wanted:&str)->String{
    for i in 0..1000{
        let gid=sha::hex(format!("public-game-{i}").as_bytes()).unwrap();
        let h=sha::hex(format!("fixture:{gid}").as_bytes()).unwrap();
        let split=if u64::from_str_radix(&h[..16],16).unwrap()%10==0{"holdout"}else{"train"};
        if split==wanted{return gid;}
    }panic!("fixture split unavailable")
}
fn fixtures()->(Value,Value){
    let first=weekly::TOTAL_SELECTED_GAMES-12*weekly::GAMES_PER_PACK;
    assert!(first>0&&first<=weekly::GAMES_PER_PACK);
    let packs:Vec<_>=(1..=13).map(|i|serde_json::json!({"sha256":format!("{i:064x}"),"bytes":1000000,
        "games":if i==1&&first<weekly::GAMES_PER_PACK{first}else{100000},
        "selected_games":if i==1{first}else{weekly::GAMES_PER_PACK}})).collect();
    let oldheld=serde_json::json!({"game_id":game_id("holdout"),"split":"holdout","pack_sha256":packs[0]["sha256"],"game_index":0});
    let files=serde_json::json!({"train.positions.jsonl":info(),"train.labels.jsonl":info(),
        "holdout.positions.jsonl":info(),"holdout.labels.jsonl":info()});
    let exclusions=serde_json::json!({"games":1000,"unique_positions":1,"corpus_canonical_sha256":info()["sha256"],
        "source_manifest_sha256":info()["sha256"],"policy":"exclude entire acquired pool; no final split files opened"});
    let dependencies=serde_json::json!({"cshogi":"1.0.4","numpy":"1.26.4"});
    let frozen=serde_json::json!({"schema_version":1,"teacher_identity":ORIGINAL_TEACHER,
        "source_corpus_manifest_sha256":weekly::CORPUS_SHA,"dependencies":dependencies,
        "games_per_pack":400,"sampling":SAMPLING,"seed":"fixture","split":SPLIT,"games":[oldheld],
        "independent_exclusions":exclusions,"positions":{"train":N,"holdout":HOLDOUT},"files":files,
        "label_depth":LABEL_DEPTH,"limitations":["synthetic metadata"],
        "derivation":{"kind":"expanded-train-frozen-holdout-v1","input_sha256":{"/tmp/source-fixture/input":info()["sha256"]},
            "expanded_dataset":"/tmp/source-fixture/expanded","frozen_holdout_dataset":"/tmp/source-fixture/frozen",
            "script_sha256":info()["sha256"],"excluded_training_rows":{},"expanded_train_count":N,
            "frozen_holdout_game_count":1,"reserved_new_holdout_game_count":0,
            "holdout_policy":"original bytes frozen; added holdout games remain unused; no final evaluation split created",
            "input_unchanged":true,"wall_seconds":0.0}});
    let sampling="whole-pack SHA256(seed:pack-sha:index) rank; explicit per-pack game counts; ply>=16/every4/max32";
    let ordering="hash-ordered packs round-robin by selected game rank; truncate final game rows";
    let profile=serde_json::json!({"schema_version":1,"kind":"whole-pack-hash-ranked-frozen-holdout-profile-v2",
        "sampling":sampling,"ordering":ordering,"selection_seed":weekly::SELECTION_SEED,
        "games_per_pack":weekly::GAMES_PER_PACK,"train_budget":N,"split_seed":"fixture","split":SPLIT,
        "corpus_manifest_sha256":weekly::CORPUS_SHA,"packs":packs,
        "frozen_dataset_manifest_sha256":ORIGINAL_MANIFEST,
        "frozen_holdout_files":{"holdout.positions.jsonl":info(),"holdout.labels.jsonl":info()},
        "independent_exclusions":exclusions,"dependencies":dependencies,"producer_sources":{"scripts/diverse_pack_dataset.py":info()},
        "row_filter":{"min_game_ply":16,"ply_stride":4,"per_game_cap":32,"max_abs_cp_exclusive":30000,"max_decoded_game_length":2048}});
    let indexes:Vec<_>=packs.iter().map(|pack|{
        let h=pack["sha256"].as_str().unwrap();
        let games=pack["games"].as_u64().unwrap();let selected=pack["selected_games"].as_u64().unwrap();
        let start=if games<1000{0}else{1000};
        let mut spans:Vec<_>=(start..start+selected).map(|i|serde_json::json!({"game_index":i,
            "start":i*100,"end":i*100+100,"positions":24,
            "rank":sha::hex(format!("{}:{h}:{i}",weekly::SELECTION_SEED).as_bytes()).unwrap()})).collect();
        spans.sort_by(|a,b|a["rank"].as_str().unwrap().cmp(b["rank"].as_str().unwrap()));
        serde_json::json!({"pack_sha256":h,"census":{"games":games,"positions":games*24,
            "prefix_400_positions":games.min(400)*24,"after_prefix_positions":games.saturating_sub(400)*24,
            "selected_prefix_400_games":if start==0{selected.min(400)}else{0}},
            "selected_spans":spans,"temporary_selected_bytes_sha256":info()["sha256"]})
    }).collect();
    let pack=packs[1]["sha256"].as_str().unwrap();
    let selected_game=serde_json::json!({"game_id":game_id("train"),"split":"train","pack_sha256":pack,"game_index":1000,
        "selection_rank":sha::hex(format!("{}:{pack}:1000",weekly::SELECTION_SEED).as_bytes()).unwrap()});
    let manifest=serde_json::json!({"schema_version":1,"teacher_identity":ORIGINAL_TEACHER,
        "source_corpus_manifest_sha256":weekly::CORPUS_SHA,"dependencies":dependencies,
        "sampling":sampling,"selection_seed":weekly::SELECTION_SEED,"games_per_pack":weekly::GAMES_PER_PACK,"train_budget":N,
        "seed":"fixture","split":SPLIT,"games":[selected_game,oldheld],"counts":{"selected_training_games":1},
        "positions":{"train":N,"holdout":HOLDOUT},"independent_exclusions":exclusions,"label_depth":LABEL_DEPTH,"files":files,
        "derivation":{"kind":"whole-pack-hash-ranked-frozen-holdout-v2","profile":profile,"profile_sha256":weekly::PROFILE_SHA,
            "producer_sources":{"scripts/diverse_pack_dataset.py":info()},
            "input_sha256":{"/tmp/source-fixture/scripts/diverse_pack_dataset.py":info()["sha256"]},"script_sha256":info()["sha256"],
            "ordering":ordering,"indexes":indexes,"frozen_dataset_manifest_sha256":ORIGINAL_MANIFEST,
            "train_histogram":{},"old_train_histogram":{},"old_train_board_overlap":0,"input_unchanged":true,"wall_seconds":0.0},
        "limitations":["synthetic metadata"]});
    (manifest,frozen)
}

#[test]fn weekly_reader_accepts_ranked_game_beyond_prefix_cap(){
    let (manifest,frozen)=fixtures();
    let view=validate_weekly_manifest(&manifest,&frozen).unwrap();
    assert!(view.train_games.contains(&game_id("train")));
    assert_eq!(manifest["games"][0]["game_index"],1000);
}
#[test]fn weekly_reader_rejects_rebound_rank_missing_holdout_and_wrong_counts(){
    let (manifest,frozen)=fixtures();
    let mut bad=manifest.clone();bad["games"][0]["selection_rank"]=info()["sha256"].clone();
    assert!(validate_weekly_manifest(&bad,&frozen).is_err());
    let mut bad=manifest.clone();bad["games"].as_array_mut().unwrap().pop();
    assert!(validate_weekly_manifest(&bad,&frozen).is_err());
    let mut bad=manifest.clone();bad["positions"]["train"]=serde_json::json!(112680);
    assert!(validate_weekly_manifest(&bad,&frozen).is_err());
    let mut bad=manifest.clone();bad["files"]["holdout.labels.jsonl"]["bytes"]=serde_json::json!(11);
    assert!(validate_weekly_manifest(&bad,&frozen).is_err());
}
#[test]fn weekly_reader_rejects_boolean_counts_unknown_fields_or_profile_drift(){
    let (manifest,frozen)=fixtures();
    for (path,value) in [("/games_per_pack",serde_json::json!(true)),("/derivation/input_unchanged",serde_json::json!(false)),
        ("/derivation/profile/row_filter/ply_stride",serde_json::json!(5)),("/derivation/indexes/0/selected_spans/0/end",serde_json::json!(1000001))]{
        let mut bad=manifest.clone();*bad.pointer_mut(path).unwrap()=value;
        assert!(validate_weekly_manifest(&bad,&frozen).is_err(),"{path}");
    }
    let mut bad=manifest;bad["unrecognized"]=serde_json::json!(true);
    assert!(validate_weekly_manifest(&bad,&frozen).is_err());
}
