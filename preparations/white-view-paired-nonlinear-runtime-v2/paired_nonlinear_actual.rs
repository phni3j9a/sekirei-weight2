//! SOURCE ONLY concrete ActualReaderAndDedicatedExporter body.
//! Runtime guards precede all I/O. Typed parent/source predecessor bodies are
//! implemented separately; only Root's compiled, activated integration may
//! replace the closed runtime guard after actual evidence has been generated.
use std::{collections::{BTreeMap,BTreeSet,HashMap,HashSet},path::{Path,PathBuf}};
use serde_json::{Value,Map};
use sekirei_core::{board::Board,sfen::board_to_sfen};
use crate::positions::PositionSample;
use crate::paired_nonlinear_cli::{self as cli,Args,FullRef,VerifiedTraining,ActualReaderAndDedicatedExporter};
use crate::paired_nonlinear_bound_io::{self as io,Snapshot,object,require};
use crate::paired_nonlinear_sha256 as sha;
use crate::paired_nonlinear_parent_binding as parent;
use crate::trainer::{paired_nonlinear_checkpoint_native as native,paired_nonlinear_checkpoint_io as checkpoint};
use crate::trainer::paired_nonlinear_positions::EpochToken;

pub const PROTOTYPE_ONLY:bool=false;
const N:usize=112681;
const HOLDOUT:u64=5895;
const ORIGINAL_TEACHER:&str="external:suisho11beta-1m-pack:376d4ef6e503d2ebe687f99e873103845b6eddc0c04d0d08e9b9c785ea061b8d";
const ORIGINAL_MANIFEST:&str="ecc419da180b86b046e1af507e6de0e218d271a5d9eaa9bff07f19ec072719a6";
use crate::weekly_nonlinear_profile as weekly;
const SPLIT:&str="SHA256(seed:complete-game-identity) modulo 10; holdout=0";
const SAMPLING:&str="first bounded games of each hash-ordered pack; ply>=16/every4/max32";
const LABEL_DEPTH:&str="0 is an external-cache sentinel, not a search depth claim";
pub fn runtime_guard()->Result<(),String>{crate::paired_nonlinear_float::runtime_ready().map(|_|())}
fn string(v:&Value)->Result<&str,String>{v.as_str().ok_or("string required".into())}
fn uint(v:&Value,min:u64)->Result<u64,String>{let n=v.as_u64().ok_or("strict unsigned JSON integer required")?;require(n>=min,"unsigned integer lower bound")?;Ok(n)}
fn nonempty(v:&Value)->Result<&str,String>{let s=string(v)?;require(!s.is_empty(),"nonempty string required")?;Ok(s)}
fn eq(v:&Value,x:Value)->Result<(),String>{require(io::exact(v,&x),"typed fixed invariant differs")}
fn fixed_sha(v:&Value)->Result<String,String>{io::sha_string(string(v)?) }
fn schema(v:&Value,s:&str)->Result<(),String>{eq(v,Value::String(s.into()))}
fn info_ref(path:&Path,info:&Value)->Result<FullRef,String>{
    let o=object(info,&["bytes","sha256"])?;
    io::fullref(&serde_json::json!({"path":path,"bytes":o["bytes"],"sha256":o["sha256"]}))
}
fn source_map(v:&Value)->Result<Vec<FullRef>,String>{
    let o=v.as_object().ok_or("immutable source/input map required")?;require(!o.is_empty(),"empty identity map")?;
    o.iter().map(|(p,info)|info_ref(&io::absolute(p)?,info)).collect()
}
fn fixed_file_info(info:&Value,count:u64)->Result<(),String>{
    let o=info.as_object().ok_or("file info object required")?;
    require(o.len()==2||o.len()==3,"unknown file metadata")?;
    require(o.contains_key("bytes")&&o.contains_key("sha256")
        &&o.keys().all(|k|matches!(k.as_str(),"bytes"|"sha256"|"count")),"file info exact allowed fields")?;
    uint(&o["bytes"],1)?;fixed_sha(&o["sha256"])?;
    if let Some(n)=o.get("count"){eq(n,serde_json::json!(count))?;}Ok(())
}
pub struct ManifestView{pub files:Map<String,Value>,pub train_games:HashSet<String>}
/// Full known expanded/frozen O manifest shape. Replay/exclusions are separately
/// verified by Root's parent; this function does not open holdout/raw games.
pub fn validate_frozen_manifest(v:&Value)->Result<ManifestView,String>{
    let o=object(v,&["schema_version","teacher_identity","source_corpus_manifest_sha256","dependencies",
        "games_per_pack","sampling","seed","split","games","independent_exclusions","positions",
        "files","label_depth","limitations","derivation"])?;
    eq(&o["schema_version"],serde_json::json!(1))?;schema(&o["teacher_identity"],ORIGINAL_TEACHER)?;
    let corpus=fixed_sha(&o["source_corpus_manifest_sha256"])?;
    require(format!("external:suisho11beta-1m-pack:{corpus}")==ORIGINAL_TEACHER,"O teacher/source pack identity")?;
    schema(&o["sampling"],SAMPLING)?;schema(&o["split"],SPLIT)?;schema(&o["label_depth"],LABEL_DEPTH)?;
    let seed=nonempty(&o["seed"])?;let cap=uint(&o["games_per_pack"],1)?;require(cap<=1000,"pack cap exceeds original bound")?;
    let dependencies=object(&o["dependencies"],&["cshogi","numpy"])?;
    for version in dependencies.values(){nonempty(version)?;}
    let positions=object(&o["positions"],&["train","holdout"])?;
    eq(&positions["train"],serde_json::json!(N))?;eq(&positions["holdout"],serde_json::json!(HOLDOUT))?;
    let f=object(&o["files"],&["train.positions.jsonl","train.labels.jsonl","holdout.positions.jsonl","holdout.labels.jsonl"])?;
    for (name,info) in f{fixed_file_info(info,if name.starts_with("train."){N as u64}else{HOLDOUT})?;}
    let exclusion=object(&o["independent_exclusions"],&["games","unique_positions","corpus_canonical_sha256","source_manifest_sha256","policy"])?;
    eq(&exclusion["games"],serde_json::json!(1000))?;uint(&exclusion["unique_positions"],1)?;
    fixed_sha(&exclusion["corpus_canonical_sha256"])?;fixed_sha(&exclusion["source_manifest_sha256"])?;
    schema(&exclusion["policy"],"exclude entire acquired pool; no final split files opened")?;
    let d=object(&o["derivation"],&["kind","input_sha256","expanded_dataset","frozen_holdout_dataset","script_sha256",
        "excluded_training_rows","expanded_train_count","frozen_holdout_game_count","reserved_new_holdout_game_count",
        "holdout_policy","input_unchanged","wall_seconds"])?;
    schema(&d["kind"],"expanded-train-frozen-holdout-v1")?;eq(&d["input_unchanged"],Value::Bool(true))?;
    let input=d["input_sha256"].as_object().ok_or("derivation inputs object required")?;require(!input.is_empty(),"missing derivation inputs")?;
    for (path,digest) in input{io::absolute(path)?;fixed_sha(digest)?;}
    io::absolute(string(&d["expanded_dataset"])?)?;io::absolute(string(&d["frozen_holdout_dataset"])?)?;fixed_sha(&d["script_sha256"])?;
    let exclusions=d["excluded_training_rows"].as_object().ok_or("excluded counts object required")?;
    for (key,n) in exclusions{require(matches!(key.as_str(),"frozen_holdout_game_rows"|"new_reserved_holdout_game_rows"|"frozen_holdout_board_rows"|"expanded_reserved_holdout_board_rows"),"unknown exclusion category")?;uint(n,1)?;}
    let excluded=exclusions.values().try_fold(0u64,|sum,v|sum.checked_add(v.as_u64().unwrap()).ok_or("exclusion count overflow"))?;
    let expanded=(N as u64).checked_add(excluded).ok_or("expanded count overflow")?;
    require(uint(&d["expanded_train_count"],N as u64)?==expanded,"expanded/train excluded count mismatch")?;
    uint(&d["frozen_holdout_game_count"],1)?;uint(&d["reserved_new_holdout_game_count"],0)?;
    schema(&d["holdout_policy"],"original bytes frozen; added holdout games remain unused; no final evaluation split created")?;
    let wall=d["wall_seconds"].as_f64().ok_or("finite wall scalar required")?;require(wall.is_finite()&&wall>=0.0,"nonfinite/negative wall scalar")?;
    let limitations=o["limitations"].as_array().ok_or("limitations array required")?;
    for text in limitations{nonempty(text)?;}
    let games=o["games"].as_array().ok_or("game inventory array required")?;require(!games.is_empty(),"empty game inventory")?;
    let mut identities=HashSet::new();let mut locations=HashSet::new();let mut train_games=HashSet::new();
    for game in games{
        let g=object(game,&["game_id","split","pack_sha256","game_index"])?;
        let gid=fixed_sha(&g["game_id"])?;let pack=fixed_sha(&g["pack_sha256"])?;let index=uint(&g["game_index"],0)?;
        let hash=sha::hex(format!("{seed}:{gid}").as_bytes())?;
        let expected=if u64::from_str_radix(&hash[..16],16).map_err(|e|e.to_string())?%10==0{"holdout"}else{"train"};
        schema(&g["split"],expected)?;
        require(index<cap&&identities.insert(gid.clone())&&locations.insert((pack,index)),"duplicate/misassigned source game")?;
        if expected=="train"{train_games.insert(gid);}
    }
    Ok(ManifestView{files:f.clone(),train_games})
}
/// The weekly dataset has a new sampling/derivation schema. It is checked
/// against the compile-bound profile and the unchanged frozen O manifest.
/// Numeric training, strict SFEN/label joins and dedicated export below are
/// byte-preserved from the compiled E3 reader.
pub fn validate_weekly_manifest(v:&Value,frozen:&Value)->Result<ManifestView,String>{
    validate_frozen_manifest(frozen)?;
    let o=object(v,&["schema_version","teacher_identity","source_corpus_manifest_sha256","dependencies",
        "sampling","selection_seed","games_per_pack","train_budget","seed","split","games","counts",
        "positions","independent_exclusions","label_depth","files","derivation","limitations"])?;
    eq(&o["schema_version"],serde_json::json!(1))?;schema(&o["teacher_identity"],ORIGINAL_TEACHER)?;
    schema(&o["source_corpus_manifest_sha256"],weekly::CORPUS_SHA)?;
    schema(&o["sampling"],"whole-pack SHA256(seed:pack-sha:index) rank; explicit per-pack game counts; ply>=16/every4/max32")?;
    schema(&o["selection_seed"],weekly::SELECTION_SEED)?;
    eq(&o["games_per_pack"],serde_json::json!(weekly::GAMES_PER_PACK))?;
    eq(&o["train_budget"],serde_json::json!(N))?;schema(&o["split"],SPLIT)?;
    schema(&o["label_depth"],LABEL_DEPTH)?;
    for key in ["teacher_identity","source_corpus_manifest_sha256","dependencies","seed","split","positions","independent_exclusions","label_depth"]{
        require(io::exact(&o[key],&frozen[key]),"weekly/frozen source, split, count or exclusion metadata differs")?;
    }
    let seed=nonempty(&o["seed"])?;
    let f=object(&o["files"],&["train.positions.jsonl","train.labels.jsonl","holdout.positions.jsonl","holdout.labels.jsonl"])?;
    for (name,info) in f{fixed_file_info(info,if name.starts_with("train."){N as u64}else{HOLDOUT})?;}
    for name in ["holdout.positions.jsonl","holdout.labels.jsonl"]{
        require(io::exact(&f[name],&frozen["files"][name]),"frozen holdout bytes/SHA changed")?;
    }
    let d=object(&o["derivation"],&["kind","profile","profile_sha256","producer_sources","input_sha256","script_sha256","ordering",
        "indexes","frozen_dataset_manifest_sha256","train_histogram","old_train_histogram",
        "old_train_board_overlap","input_unchanged","wall_seconds"])?;
    schema(&d["kind"],"whole-pack-hash-ranked-frozen-holdout-v2")?;
    schema(&d["profile_sha256"],weekly::PROFILE_SHA)?;
    schema(&d["frozen_dataset_manifest_sha256"],ORIGINAL_MANIFEST)?;
    schema(&d["ordering"],"hash-ordered packs round-robin by selected game rank; truncate final game rows")?;
    eq(&d["input_unchanged"],serde_json::json!(true))?;fixed_sha(&d["script_sha256"])?;
    let wall=d["wall_seconds"].as_f64().ok_or("finite dataset wall seconds required")?;
    require(wall.is_finite()&&wall>=0.0,"negative/nonfinite dataset wall seconds")?;
    let hashes=d["input_sha256"].as_object().ok_or("whole dataset derivation input hashes required")?;
    require(!hashes.is_empty(),"empty dataset input hashes")?;
    for (path,digest) in hashes{io::absolute(path)?;fixed_sha(digest)?;}
    require(uint(&d["old_train_board_overlap"],0)?<=N as u64,"old overlap exceeds train count")?;
    for key in ["train_histogram","old_train_histogram"]{
        let counts=d[key].as_object().ok_or("diagnostic histogram object required")?;
        for count in counts.values(){uint(count,0)?;}
    }
    let counts=o["counts"].as_object().ok_or("dataset filter counts object required")?;
    for (key,count) in counts{
        require(matches!(key.as_str(),"empty_selected_games"|"duplicate_selected_games"|"reserved_selected_games"
            |"reserved_board_rows"|"duplicate_training_board_rows"|"selected_training_games"),"unknown dataset filter count")?;
        uint(count,0)?;
    }
    let profile=object(&d["profile"],&["schema_version","kind","sampling","ordering","row_filter","selection_seed",
        "games_per_pack","train_budget","split_seed","split","corpus_manifest_sha256","packs",
        "frozen_dataset_manifest_sha256","frozen_holdout_files","independent_exclusions","dependencies","producer_sources"])?;
    eq(&profile["schema_version"],serde_json::json!(1))?;schema(&profile["kind"],"whole-pack-hash-ranked-frozen-holdout-profile-v2")?;
    for (key,source) in [("sampling",&o["sampling"]),("ordering",&d["ordering"]),("selection_seed",&o["selection_seed"]),
        ("games_per_pack",&o["games_per_pack"]),("train_budget",&o["train_budget"]),("split_seed",&o["seed"]),
        ("split",&o["split"]),("corpus_manifest_sha256",&o["source_corpus_manifest_sha256"]),
        ("frozen_dataset_manifest_sha256",&d["frozen_dataset_manifest_sha256"]),
        ("independent_exclusions",&o["independent_exclusions"]),("dependencies",&o["dependencies"])]{
        require(io::exact(&profile[key],source),"profile/manifest declaration differs")?;
    }
    require(io::exact(&profile["row_filter"],&serde_json::json!({"min_game_ply":16,"ply_stride":4,"per_game_cap":32,
        "max_abs_cp_exclusive":30000,"max_decoded_game_length":2048})),"profile row filter differs")?;
    require(io::exact(&profile["frozen_holdout_files"],&serde_json::json!({"holdout.positions.jsonl":f["holdout.positions.jsonl"],
        "holdout.labels.jsonl":f["holdout.labels.jsonl"]})),"profile frozen holdout differs")?;
    require(io::exact(&d["producer_sources"],&profile["producer_sources"]),"manifest/profile producer source closure differs")?;
    let producers=profile["producer_sources"].as_object().ok_or("producer source closure required")?;
    require(producers.contains_key("scripts/diverse_pack_dataset.py"),"selected replay producer source missing")?;
    for (name,info) in producers{
        require(!name.starts_with('/')&&name.split('/').all(|part|!part.is_empty()&&part!="."&&part!=".."),"canonical relative producer path required")?;
        fixed_file_info(info,0)?;
        let matches:Vec<_>=hashes.iter().filter(|(path,_)|path.ends_with(&format!("/{name}"))).collect();
        require(matches.len()==1&&io::exact(matches[0].1,&info["sha256"]),"producer hash absent/rebound in full derivation input closure")?;
    }
    require(io::exact(&producers["scripts/diverse_pack_dataset.py"]["sha256"],&d["script_sha256"]),"manifest producer script hash differs")?;
    let packs=profile["packs"].as_array().ok_or("profile pack inventory required")?;
    require(packs.len()as u64==weekly::PACK_COUNT,"profile pack count differs")?;
    let mut pack_sizes=BTreeMap::new();let mut previous="".to_string();
    for pack in packs{
        let pack=object(pack,&["sha256","bytes","games","selected_games"])?;let hash=fixed_sha(&pack["sha256"])?;let size=uint(&pack["bytes"],1)?;
        let games=uint(&pack["games"],1)?;let selected=uint(&pack["selected_games"],1)?;
        require(selected<=games&&selected<=weekly::GAMES_PER_PACK,"profile per-pack selection exceeds census or cap")?;
        require(hash>previous&&pack_sizes.insert(hash.clone(),(size,games,selected)).is_none(),"profile packs not unique SHA ordered")?;previous=hash;
    }
    require(pack_sizes.values().map(|x|x.2).sum::<u64>()==weekly::TOTAL_SELECTED_GAMES,"profile total selected games differs")?;
    let indexes=d["indexes"].as_array().ok_or("pack index array required")?;
    require(indexes.len()==packs.len(),"whole pack index membership differs")?;
    let mut selected=BTreeMap::new();let mut indexed=BTreeSet::new();
    for index in indexes{
        let index=object(index,&["pack_sha256","census","selected_spans","temporary_selected_bytes_sha256"])?;
        let pack=fixed_sha(&index["pack_sha256"])?;fixed_sha(&index["temporary_selected_bytes_sha256"])?;
        let &(size,declared_games,selected_games)=pack_sizes.get(&pack).ok_or("index pack absent from profile")?;
        require(indexed.insert(pack.clone()),"duplicate pack index")?;
        let census=object(&index["census"],&["games","positions","prefix_400_positions","after_prefix_positions","selected_prefix_400_games"])?;
        let game_count=uint(&census["games"],1)?;
        require(game_count==declared_games,"pack full census differs from preregistered profile")?;
        let positions=uint(&census["positions"],0)?;
        require(uint(&census["prefix_400_positions"],0)?.checked_add(uint(&census["after_prefix_positions"],0)?)==Some(positions),"index census position totals differ")?;
        let spans=index["selected_spans"].as_array().ok_or("selected game spans required")?;
        require(spans.len()as u64==selected_games,"selected count differs from per-pack profile")?;
        let mut last_rank="".to_string();let mut intervals=Vec::new();let mut prefix=0;
        for span in spans{
            let span=object(span,&["game_index","start","end","positions","rank"])?;
            let location=uint(&span["game_index"],0)?;let start=uint(&span["start"],0)?;let end=uint(&span["end"],1)?;
            require(location<game_count&&start<end&&end<=size,"selected span outside complete pack")?;
            require(uint(&span["positions"],0)?<=2048,"selected game exceeds replay length bound")?;
            let rank=fixed_sha(&span["rank"])?;
            require(rank==sha::hex(format!("{}:{pack}:{location}",weekly::SELECTION_SEED).as_bytes())?
                &&rank>last_rank,"selected game rank/order differs")?;last_rank=rank.clone();
            require(selected.insert((pack.clone(),location),rank).is_none(),"duplicate selected pack location")?;
            intervals.push((start,end));if location<400{prefix+=1;}
        }
        intervals.sort_unstable();require(intervals.windows(2).all(|w|w[0].1<=w[1].0),"selected pack spans overlap")?;
        require(uint(&census["selected_prefix_400_games"],0)?==prefix,"selected prefix census differs")?;
    }
    let frozen_games=frozen["games"].as_array().ok_or("frozen game inventory required")?;
    let held:BTreeMap<_,_>=frozen_games.iter().filter(|g|g["split"].as_str()==Some("holdout"))
        .map(|g|(g["game_id"].as_str().unwrap().to_string(),g)).collect();
    let games=o["games"].as_array().ok_or("weekly game inventory required")?;
    require(!games.is_empty(),"empty weekly game inventory")?;
    let mut identities=HashSet::new();let mut locations=HashSet::new();let mut train_games=HashSet::new();let mut seen_held=BTreeSet::new();
    for game in games{
        let ranked=game.get("selection_rank").is_some();
        let g=object(game,if ranked{&["game_id","split","pack_sha256","game_index","selection_rank"]}
            else{&["game_id","split","pack_sha256","game_index"]})?;
        let gid=fixed_sha(&g["game_id"])?;let pack=fixed_sha(&g["pack_sha256"])?;let location=uint(&g["game_index"],0)?;
        let hash=sha::hex(format!("{seed}:{gid}").as_bytes())?;
        let split=if u64::from_str_radix(&hash[..16],16).map_err(|e|e.to_string())?%10==0{"holdout"}else{"train"};
        schema(&g["split"],split)?;
        require(identities.insert(gid.clone())&&locations.insert((pack.clone(),location)),"duplicate weekly game identity/location")?;
        if ranked{
            let rank=fixed_sha(&g["selection_rank"])?;
            require(selected.get(&(pack,location))==Some(&rank),"weekly game not from preregistered selected pack span")?;
            require(!held.contains_key(&gid),"frozen holdout provenance must remain original")?;
        }else{
            let original=held.get(&gid).ok_or("unranked game is not original frozen holdout")?;
            require(io::exact(game,original),"frozen holdout game provenance changed")?;seen_held.insert(gid.clone());
        }
        if split=="train"{train_games.insert(gid);}
    }
    require(seen_held.len()==held.len(),"frozen holdout inventory incomplete")?;
    for limitation in o["limitations"].as_array().ok_or("limitations array required")?{nonempty(limitation)?;}
    Ok(ManifestView{files:f.clone(),train_games})
}
fn rows(raw:&[u8])->Result<Vec<Value>,String>{
    let text=std::str::from_utf8(raw).map_err(|e|e.to_string())?;
    require(!text.is_empty()&&text.ends_with('\n')&&!text.contains('\r'),"nonempty canonical LF JSONL required")?;
    text[..text.len()-1].split('\n').map(|line|{
        require(!line.trim().is_empty(),"blank JSONL row")?;io::unique_json(line.as_bytes())
    }).collect()
}
fn canonical_sfen(s:&str)->Result<(Board,String),String>{
    require(!s.contains('\n')&&!s.contains('\r'),"single SFEN line required")?;
    let board=Board::from_sfen(s).map_err(|e|e.to_string())?;let canonical=board_to_sfen(&board);
    require(canonical==s,"noncanonical SFEN rejected, never rewritten")?;
    let parts:Vec<&str>=canonical.split(' ').collect();require(parts.len()==4,"SFEN four canonical fields")?;
    Ok((board,parts[..3].join(" ")))
}
/// Labels and positions are parsed from already hashed immutable bytes once.
/// Position order survives; cache order can differ because the join is keyed.
pub fn strict_train_join(positions:&[u8],labels:&[u8],view:&ManifestView,teacher:&str,expected:usize)
    ->Result<(Vec<PositionSample>,HashMap<String,i32>),String>{
    require(teacher==ORIGINAL_TEACHER,"fixed O teacher identity required")?;
    let mut cache=HashMap::new();let mut semantic_labels=HashSet::new();
    for row in rows(labels)?{
        let o=object(&row,&["sfen","score_cp","label_depth","teacher_identity"])?;
        let sfen=string(&o["sfen"])?;schema(&o["teacher_identity"],teacher)?;eq(&o["label_depth"],serde_json::json!(0))?;
        let cp=o["score_cp"].as_i64().ok_or("strict signed integer cp required")?;
        require((-29999..=29999).contains(&cp),"teacher cp mate/nonordinary bound")?;
        let (_,semantic)=canonical_sfen(sfen)?;
        require(semantic_labels.insert(semantic)&&cache.insert(sfen.to_owned(),cp as i32).is_none(),"duplicate teacher SFEN/semantic position")?;
    }
    let mut samples=Vec::new();let mut sfens=HashSet::new();let mut semantics=HashSet::new();
    for row in rows(positions)?{
        let o=object(&row,&["schema_version","sfen","source","tags"])?;eq(&o["schema_version"],serde_json::json!(1))?;
        let source=object(&o["source"],&["kind","path","ply"])?;schema(&source["kind"],"gensfen-pack")?;
        let gid=string(&source["path"])?;require(view.train_games.contains(gid),"position must use original train source game")?;
        let ply=u32::try_from(uint(&source["ply"],16)?).map_err(|e|e.to_string())?;
        let tags=object(&o["tags"],&["side_to_move","phase"])?;schema(&tags["phase"],"middlegame")?;
        let sfen=string(&o["sfen"])?;let (board,semantic)=canonical_sfen(sfen)?;
        let parts:Vec<_>=sfen.split(' ').collect();let stm=if parts[1]=="b"{"black"}else if parts[1]=="w"{"white"}else{return Err("invalid STM".into())};
        schema(&tags["side_to_move"],stm)?;
        require(parts[3].parse::<u32>().map_err(|e|e.to_string())?==ply,"source ply/SFEN ply mismatch")?;
        require(sfens.insert(sfen.to_owned())&&semantics.insert(semantic)&&cache.contains_key(sfen),"duplicate/missing position label")?;
        samples.push(PositionSample{board,phase:"middlegame".into(),side_to_move:stm.into(),ply,source:gid.into()});
    }
    require(samples.len()==expected&&cache.len()==expected&&sfens.len()==cache.len(),"full cache/positions count/set differs")?;
    require(cache.keys().all(|s|sfens.contains(s)),"extra cache SFEN")?;
    Ok((samples,cache))
}
pub struct ReaderExporter{snapshots:BTreeMap<PathBuf,Snapshot>,context:Option<checkpoint::CheckpointContext>,output:Option<PathBuf>}
impl ReaderExporter{pub fn new()->Self{Self{snapshots:BTreeMap::new(),context:None,output:None}}
    fn raw(&self,r:&FullRef)->Result<&[u8],String>{let s=self.snapshots.get(&r.path).ok_or("unbound raw input")?;require(io::same_ref(&s.reference,r),"raw ref rebound")?;Ok(&s.bytes)}
}
impl ActualReaderAndDedicatedExporter for ReaderExporter{
    fn verify_and_load(&mut self,args:&Args)->Result<VerifiedTraining,String>{
        runtime_guard()?; // absolutely before raw input/source/reference file reads
        require(self.snapshots.is_empty()&&self.context.is_none()&&self.output.is_none(),"reader instance cannot reuse a previous attempt")?;
        for r in [&args.recipe,&args.source_binding,&args.manifest,&args.reference03,&args.positions,&args.labels]{io::insert_bound(&mut self.snapshots,r)?;}
        let recipe=io::unique_json(self.raw(&args.recipe)?)?;let selected=cli::selected_recipe(&recipe,args)?;
        require(selected.teacher_identity==ORIGINAL_TEACHER&&args.manifest.sha256==weekly::MANIFEST_SHA,"fixed weekly dataset identities differ")?;
        let binding=parent::declared_binding(self.raw(&args.source_binding)?,args)?;
        binding.require_current_exe(&std::env::current_exe().map_err(|e|e.to_string())?)?;
        for r in binding.raw_refs_required(){
            require(!r.path.starts_with(&args.output)&&!args.output.starts_with(&r.path),"output/input/source overlap")?;
            io::insert_bound(&mut self.snapshots,&r)?;
        }
        let preliminary=parent::RawInputs::from_reader_snapshots(&self.snapshots)?;
        for r in binding.additional_parent_input_refs(&preliminary)?{
            require(!r.path.starts_with(&args.output)&&!args.output.starts_with(&r.path),"output/preflight input overlap")?;
            io::insert_bound(&mut self.snapshots,&r)?;
        }
        let inputs=parent::RawInputs::from_reader_snapshots(&self.snapshots)?;
        binding.validate_parent_and_source(&inputs,args,None)?;
        let manifest=io::unique_json(self.raw(&args.manifest)?)?;
        let frozen_ref=binding.frozen_manifest_ref(&inputs)?;
        let frozen=io::unique_json(self.raw(&frozen_ref)?)?;
        let view=validate_weekly_manifest(&manifest,&frozen)?;
        let root=args.manifest.path.parent().ok_or("manifest dataset parent missing")?;
        require(!args.output.starts_with(root),"fresh output may not alter original dataset tree")?;
        let binding_files=binding.original_refs();
        for (name,info) in &view.files{
            let r=binding_files.get(name).ok_or("original fullref missing")?;
            require(r.path==root.join(name)&&io::exact(&io::info_value(&r),&serde_json::json!({"bytes":info["bytes"],"sha256":info["sha256"]})),"manifest/O four file fullref disagreement")?;
        }
        let (samples,cache)=strict_train_join(self.raw(&args.positions)?,self.raw(&args.labels)?,&view,&selected.teacher_identity,N)?;
        let reference_bytes=self.raw(&args.reference03)?;
        let reconstructed=native::encode_native03_layout(&native::canonical_reference_native03().map_err(str::to_string)?).map_err(str::to_string)?;
        require(reference_bytes==reconstructed,"reference03 all bytes must equal independent seed42 fullhand reconstruction")?;
        let native03=native::decode_native03_layout(reference_bytes).map_err(str::to_string)?;
        require(native::encode_native03_layout(&native03).map_err(str::to_string)?==reference_bytes,"reference03 independent decode roundtrip")?;
        // This is the dedicated compiled new03 core reader, NOT retagging into
        // old01. Root must provide its typed source/build/body proof above.
        let compiled=sekirei_core::nnue::read_weights(&args.reference03.path).map_err(|e|e.to_string())?;
        require(native::encode_native03_layout(&compiled).map_err(str::to_string)?==reference_bytes,"compiled new03 reference reader fullbyte mismatch")?;
        let (trainer,reference)=checkpoint::fresh_trainer(selected.recipe)?;
        io::verify_all(&self.snapshots)?;
        self.context=Some(checkpoint::CheckpointContext{recipe:io::ref_value(&args.recipe),source_binding:io::ref_value(&args.source_binding),
            manifest:io::ref_value(&args.manifest),reference03:io::ref_value(&args.reference03),positions:io::ref_value(&args.positions),
            labels:io::ref_value(&args.labels),source_files:io::identity_map(&self.snapshots)});
        self.output=Some(args.output.clone());
        Ok(VerifiedTraining{trainer,reference:Some(reference),samples,cache,declaration:selected})
    }
    fn commit_completed_native03(&mut self,args:&Args,train:&VerifiedTraining,tokens:&[EpochToken])->Result<(),String>{
        runtime_guard()?; // no writes before an explicitly enabled separate integration
        let context=self.context.as_ref().ok_or("export without verified reader state")?;
        require(self.output.as_ref()==Some(&args.output),"output declaration changed after reader validation")?;
        for (r,v) in [(&args.recipe,&context.recipe),(&args.source_binding,&context.source_binding),
            (&args.manifest,&context.manifest),(&args.reference03,&context.reference03),
            (&args.positions,&context.positions),(&args.labels,&context.labels)]{
            require(io::same_ref(r,&io::fullref(v)?),"export argument raw fullref changed")?;
        }
        let original_declaration=cli::selected_recipe(&io::unique_json(self.raw(&args.recipe)?)?,args)?;
        let actual=train.declaration.recipe;let original=original_declaration.recipe;
        require([actual.learning_rate.to_bits(),actual.head_init_width.to_bits(),actual.head_bias_init.to_bits(),actual.output_native_l1_budget.to_bits()]
            ==[original.learning_rate.to_bits(),original.head_init_width.to_bits(),original.head_bias_init.to_bits(),original.output_native_l1_budget.to_bits()]
            &&train.declaration.teacher_identity==original_declaration.teacher_identity
            &&train.declaration.shuffle_seed==original_declaration.shuffle_seed,"export selected recipe differs from frozen raw declaration")?;
        require(tokens.len()==3&&tokens.iter().enumerate().all(|(i,t)|t.epoch()==i as u32+1&&t.positions()==N&&t.end_step()==(i as u64+1)*N as u64)
            &&train.trainer.paired_optimizer_step()==3*N as u64&&train.reference.is_some(),"three complete original TRAIN epochs required; partial export/resume forbidden")?;
        io::verify_all(&self.snapshots)?;
        let recipe=train.declaration.recipe;
        let checkpoint_bytes=checkpoint::completed_checkpoint_bytes(&train.trainer,recipe,context)?;
        let nearest=checkpoint::completed_nearest_bytes(&train.trainer,recipe)?;
        checkpoint::verify_completed_readback(&train.trainer,&checkpoint_bytes,recipe,context)?;
        checkpoint::verify_completed_native(&train.trainer,recipe,&nearest)?;
        io::fresh_directory(&args.output)?; // never overwrites a prior partial/destination
        let cp_ref=io::write_exclusive(&args.output.join("fullstate.bits.json"),&checkpoint_bytes)?;
        let reread=io::read_bound(&cp_ref)?;
        checkpoint::verify_completed_readback(&train.trainer,&reread.bytes,recipe,context)?;
        let native_ref=io::write_exclusive(&args.output.join("weights.nearest03.bin"),&nearest)?;
        let native_snapshot=io::read_bound(&native_ref)?;
        checkpoint::verify_completed_native(&train.trainer,recipe,&native_snapshot.bytes)?;
        // Files that belong to the frozen reader state are checked again before
        // writing a numeric/export stage draft. Child cannot claim its own reap.
        io::verify_all(&self.snapshots)?;
        let float_export=crate::paired_nonlinear_float::snapshot()?;
        let draft=serde_json::json!({"schema":"sekirei.white-view-paired-nonlinear-export-stage.v1",
            "float_export":float_export,
            "status":"three-epochs-export-readback-complete","epochs_completed":3,"positions_per_epoch":N,"global_step":3*N,
            "resume_allowed":false,"feature_schema":native::FEATURE_TAG,"context":context,
            "outputs":{"checkpoint":io::ref_value(&cp_ref),"nearest_native03":io::ref_value(&native_ref)},
            "native_fnv1a":format!("{:016x}",native::native03_fnv1a(&nearest).map_err(str::to_string)?),
            "inputs_before":io::identity_map(&self.snapshots),"inputs_after":io::identity_map(&self.snapshots),
            "checkpoint_all_state_bits_equal":true,"nearest_all_bytes_equal":true,"cleanup_verified":false,
            "parent_reap_verified":false,"core_fullrows_verified":false,"incremental_verified":false,"adoption_claimed":false,"final_used":false});
        io::write_exclusive(&args.output.join("export-stage.json"),&io::json_bytes(&draft)?)?;
        io::sync_dir(&args.output)?;io::verify_all(&self.snapshots)?;Ok(())
    }
}
pub fn actual_entry(argv:&[String])->Result<(),String>{
    runtime_guard()?; // before CLI parse, output creation, reads, training or export
    let args=cli::parse(argv)?;let mut reader=ReaderExporter::new();
    cli::run_dedicated(&args,&mut reader) // frozen CLI also remains guarded
}

// Read original inputs and real codecs once, without any optimizer update.
pub fn verify_inputs_and_initialized_io(argv:&[String])->Result<(),String>{
    runtime_guard()?;
    let float_before=crate::paired_nonlinear_float::snapshot()?;
    let args=cli::parse(argv)?;
    let mut reader=ReaderExporter::new();let train=reader.verify_and_load(&args)?;
    require(train.trainer.paired_optimizer_step()==0&&train.samples.len()==N,"initialized I/O diagnostic must have zero optimizer updates")?;
    let float_after_load=crate::paired_nonlinear_float::snapshot()?;
    let context=reader.context.as_ref().ok_or("verified diagnostic context missing")?;
    let recipe=train.declaration.recipe;
    let checkpoint_bytes=checkpoint::initialized_checkpoint_bytes(&train.trainer,recipe,context)?;
    let nearest=checkpoint::initialized_native_bytes(&train.trainer,recipe)?;
    checkpoint::verify_initialized_readback(&train.trainer,recipe,context,&checkpoint_bytes)?;
    checkpoint::verify_initialized_native(&train.trainer,recipe,&nearest)?;
    io::verify_all(&reader.snapshots)?;
    io::fresh_directory(&args.output)?;
    let cp=io::write_exclusive(&args.output.join("initialized-fullstate.bits.json"),&checkpoint_bytes)?;
    let cp_read=io::read_bound(&cp)?;
    checkpoint::verify_initialized_readback(&train.trainer,recipe,context,&cp_read.bytes)?;
    let nw=io::write_exclusive(&args.output.join("initialized.nearest03.bin"),&nearest)?;
    let nw_read=io::read_bound(&nw)?;checkpoint::verify_initialized_native(&train.trainer,recipe,&nw_read.bytes)?;
    let core=sekirei_core::nnue::read_weights(&nw.path).map_err(|e|e.to_string())?;
    require(native::encode_native03_layout(&core).map_err(str::to_string)?==nearest,"actual initialized native core fullbyte readback differs")?;
    io::verify_all(&reader.snapshots)?;
    let float_after_io=crate::paired_nonlinear_float::snapshot()?;
    let draft=serde_json::json!({"schema":"sekirei.white-view-paired-nonlinear-initialized-io-stage.v1",
        "status":"verified-original-inputs-and-initialized-codecs","mode":cli::MODE,
        "train_count":N,"global_step":0,"epochs_completed":0,"candidate_model":false,"resume_allowed":false,
        "float_observations":{"before":float_before,"after_load":float_after_load,"after_io":float_after_io},
        "context":context,"outputs":{"initialized_checkpoint":io::ref_value(&cp),"initialized_native03":io::ref_value(&nw)},
        "inputs_before":io::identity_map(&reader.snapshots),"inputs_after":io::identity_map(&reader.snapshots),
        "all_state_bits_equal":true,"all_nearest_bytes_equal":true,"compiled_native_reader_fullbytes_equal":true,
        "parent_reap_verified":false,"actual_fit_started":false,"model_adopted":false,"final_used":false});
    io::write_exclusive(&args.output.join("initialized-io-stage.json"),&io::json_bytes(&draft)?)?;
    io::sync_dir(&args.output)?;io::verify_all(&reader.snapshots)?;
    eprintln!("PAIRED_INITIALIZED_IO_OK train_count={N} global_step=0");Ok(())
}
#[cfg(test)]
#[path="weekly_nonlinear_manifest_tests.rs"]
mod weekly_manifest_tests;
