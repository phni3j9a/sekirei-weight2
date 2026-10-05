//! SOURCE ONLY. Recursive duplicate rejection, typed refs, immutable raw reads.
//! Runtime Reader/exporter guards are in the caller; these are not verified tokens.
use std::{collections::BTreeMap,fmt,fs::{self,File,OpenOptions},io::{Read,Write},path::{Path,PathBuf}};
use std::os::unix::fs::{MetadataExt,OpenOptionsExt,DirBuilderExt};
use serde::{Deserialize,de::{self,Visitor,SeqAccess,MapAccess}};
use serde_json::{Value,Map,Number};
use crate::paired_nonlinear_cli::FullRef;
use crate::paired_nonlinear_sha256 as sha;
const MAX_FILE_BYTES:u64=512*1024*1024;
// Linux v6.8 include/uapi/asm-generic/fcntl.h, O_NOFOLLOW=00400000.
// Dedicated integration is Linux-only; no added sha2/libc executable/dependency.
const O_NOFOLLOW:i32=0o400000;
pub fn require(ok:bool,msg:&str)->Result<(),String>{if ok{Ok(())}else{Err(msg.into())}}
pub fn object<'a>(v:&'a Value,keys:&[&str])->Result<&'a Map<String,Value>,String>{
    let o=v.as_object().ok_or("object required")?;
    require(o.len()==keys.len()&&keys.iter().all(|k|o.contains_key(*k)),"unknown/missing JSON keys")?;Ok(o)
}
pub fn absolute(s:&str)->Result<PathBuf,String>{
    require(s.starts_with('/')&&s.len()>1&&!s.contains('\0')
        &&!s.split('/').skip(1).any(|x|x.is_empty()||x=="."||x==".."),"lexical canonical absolute path required")?;
    Ok(PathBuf::from(s))
}
pub fn sha_string(s:&str)->Result<String,String>{
    require(s.len()==64&&s.bytes().all(|b|b.is_ascii_digit()||(b'a'..=b'f').contains(&b)),"lower SHA256 required")?;Ok(s.into())
}
pub fn fullref(v:&Value)->Result<FullRef,String>{
    let o=object(v,&["path","bytes","sha256"])?;
    let bytes=o["bytes"].as_u64().ok_or("strict integer byte size required")?;
    require(bytes>0&&bytes<=MAX_FILE_BYTES,"positive bounded file size required")?;
    Ok(FullRef{path:absolute(o["path"].as_str().ok_or("path string required")?)?,bytes,
        sha256:sha_string(o["sha256"].as_str().ok_or("SHA string required")?)?})
}
pub fn same_ref(a:&FullRef,b:&FullRef)->bool{a.path==b.path&&a.bytes==b.bytes&&a.sha256==b.sha256}
pub fn ref_value(r:&FullRef)->Value{serde_json::json!({"path":r.path,"bytes":r.bytes,"sha256":r.sha256})}
pub fn info_value(r:&FullRef)->Value{serde_json::json!({"bytes":r.bytes,"sha256":r.sha256})}
pub fn exact(a:&Value,b:&Value)->bool{
    match (a,b){
        (Value::Object(x),Value::Object(y))=>x.len()==y.len()&&x.iter().all(|(k,v)|y.get(k).is_some_and(|w|exact(v,w))),
        (Value::Array(x),Value::Array(y))=>x.len()==y.len()&&x.iter().zip(y).all(|(v,w)|exact(v,w)),
        (Value::Number(x),Value::Number(y))=>x.is_f64()==y.is_f64()&&x==y,
        _=>a==b,
    }
}
struct Unique(Value);
impl<'de> Deserialize<'de> for Unique {
    fn deserialize<D:serde::Deserializer<'de>>(d:D)->Result<Self,D::Error>{
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value=Unique;
            fn expecting(&self,f:&mut fmt::Formatter)->fmt::Result{f.write_str("finite JSON with unique recursive object keys")}
            fn visit_bool<E:de::Error>(self,x:bool)->Result<Unique,E>{Ok(Unique(Value::Bool(x)))}
            fn visit_i64<E:de::Error>(self,x:i64)->Result<Unique,E>{Ok(Unique(Value::Number(x.into())))}
            fn visit_u64<E:de::Error>(self,x:u64)->Result<Unique,E>{Ok(Unique(Value::Number(x.into())))}
            fn visit_f64<E:de::Error>(self,x:f64)->Result<Unique,E>{
                Number::from_f64(x).map(|n|Unique(Value::Number(n))).ok_or_else(||E::custom("nonfinite JSON"))
            }
            fn visit_str<E:de::Error>(self,x:&str)->Result<Unique,E>{Ok(Unique(Value::String(x.into())))}
            fn visit_string<E:de::Error>(self,x:String)->Result<Unique,E>{Ok(Unique(Value::String(x)))}
            fn visit_unit<E:de::Error>(self)->Result<Unique,E>{Ok(Unique(Value::Null))}
            fn visit_none<E:de::Error>(self)->Result<Unique,E>{Ok(Unique(Value::Null))}
            fn visit_seq<A:SeqAccess<'de>>(self,mut s:A)->Result<Unique,A::Error>{
                let mut v=Vec::new();while let Some(x)=s.next_element::<Unique>()?{v.push(x.0);}Ok(Unique(Value::Array(v)))
            }
            fn visit_map<A:MapAccess<'de>>(self,mut a:A)->Result<Unique,A::Error>{
                let mut v=Map::new();while let Some(k)=a.next_key::<String>()?{
                    if v.contains_key(&k){return Err(de::Error::custom("duplicate JSON object key"));}
                    v.insert(k,a.next_value::<Unique>()?.0);
                }Ok(Unique(Value::Object(v)))
            }
        }d.deserialize_any(V)
    }
}
pub fn unique_json(raw:&[u8])->Result<Value,String>{
    let mut d=serde_json::Deserializer::from_slice(raw);
    let value=Unique::deserialize(&mut d).map_err(|e|e.to_string())?.0;
    d.end().map_err(|e|e.to_string())?;Ok(value) // serde's bounded recursion remains enabled.
}
pub fn json_bytes(v:&Value)->Result<Vec<u8>,String>{
    let mut b=serde_json::to_vec(v).map_err(|e|e.to_string())?;b.push(b'\n');Ok(b)
}
#[derive(Clone,Debug,PartialEq,Eq)]pub struct Fingerprint{dev:u64,ino:u64,mode:u32,uid:u32,len:u64,mtime:i64,mns:i64,ctime:i64,cns:i64}
fn fp(m:&fs::Metadata)->Fingerprint{Fingerprint{dev:m.dev(),ino:m.ino(),mode:m.mode(),uid:m.uid(),len:m.len(),mtime:m.mtime(),mns:m.mtime_nsec(),ctime:m.ctime(),cns:m.ctime_nsec()}}
#[derive(Clone)]pub struct Snapshot{pub reference:FullRef,pub bytes:Vec<u8>,fingerprint:Fingerprint}
/// Canonical regular path + O_NOFOLLOW + fd/path before/after metadata + raw SHA.
pub fn read_bound(r:&FullRef)->Result<Snapshot,String>{
    bounded_file_bytes(r.bytes)?;sha_string(&r.sha256)?;
    require(absolute(r.path.to_str().ok_or("UTF8 path")?)?==r.path,"canonical declaration path")?;
    let pm=fs::symlink_metadata(&r.path).map_err(|e|e.to_string())?;
    require(pm.is_file()&&!pm.file_type().is_symlink(),"regular non-symlink required")?;
    require(fs::canonicalize(&r.path).map_err(|e|e.to_string())?==r.path,"physical canonical path differs")?;
    let mut f=OpenOptions::new().read(true).custom_flags(O_NOFOLLOW).open(&r.path).map_err(|e|e.to_string())?;
    let start=f.metadata().map_err(|e|e.to_string())?;let before=fp(&start);
    require(start.is_file()&&before==fp(&pm)&&start.len()==r.bytes,"file fingerprint/size differs before read")?;
    let mut bytes=Vec::new();Read::by_ref(&mut f).take(r.bytes+1).read_to_end(&mut bytes).map_err(|e|e.to_string())?;
    require(bytes.len()as u64==r.bytes&&sha::hex(&bytes)?==r.sha256,"raw bytes/SHA mismatch")?;
    require(fp(&f.metadata().map_err(|e|e.to_string())?)==before
        &&fp(&fs::symlink_metadata(&r.path).map_err(|e|e.to_string())?)==before,"file mutated during read")?;
    Ok(Snapshot{reference:r.clone(),bytes,fingerprint:before})
}
pub fn verify_snapshot(s:&Snapshot)->Result<(),String>{
    let now=read_bound(&s.reference)?;require(now.fingerprint==s.fingerprint&&now.bytes==s.bytes,"snapshot changed")
}
pub fn verify_all(map:&BTreeMap<PathBuf,Snapshot>)->Result<(),String>{for s in map.values(){verify_snapshot(s)?;}Ok(())}
pub fn insert_bound(map:&mut BTreeMap<PathBuf,Snapshot>,r:&FullRef)->Result<(),String>{
    if let Some(s)=map.get(&r.path){require(same_ref(&s.reference,r),"colliding immutable path identity")?;verify_snapshot(s)}
    else {let s=read_bound(r)?;map.insert(r.path.clone(),s);Ok(())}
}
pub fn identity_map(map:&BTreeMap<PathBuf,Snapshot>)->Value{
    Value::Object(map.iter().map(|(p,s)|(p.to_string_lossy().into_owned(),info_value(&s.reference))).collect())
}
pub fn fresh_directory(path:&Path)->Result<(),String>{
    let parent=path.parent().ok_or("output parent required")?;
    require(fs::canonicalize(parent).map_err(|e|e.to_string())?==parent,"canonical output parent required")?;
    fs::DirBuilder::new().mode(0o700).create(path).map_err(|e|format!("fresh exclusive output directory: {e}"))
}
pub fn write_exclusive(path:&Path,bytes:&[u8])->Result<FullRef,String>{
    let mut f=OpenOptions::new().write(true).create_new(true).mode(0o600).custom_flags(O_NOFOLLOW).open(path).map_err(|e|e.to_string())?;
    f.write_all(bytes).and_then(|_|f.sync_all()).map_err(|e|e.to_string())?;
    let r=FullRef{path:path.to_path_buf(),bytes:bytes.len()as u64,sha256:sha::hex(bytes)?};
    let s=read_bound(&r)?;require(s.bytes==bytes,"exclusive file readback differs")?;Ok(r)
}
pub fn sync_dir(path:&Path)->Result<(),String>{File::open(path).and_then(|f|f.sync_all()).map_err(|e|e.to_string())}
#[cfg(test)]mod tests{use super::*;
    #[test]fn recursive_unique(){assert!(unique_json(br#"{"outer":{"x":1,"x":2}}"#).is_err());assert!(unique_json(b"[1e400]").is_err());assert!(unique_json(b"{\"x\":true}").is_ok());}
    #[test]fn strict_number(){assert!(!exact(&serde_json::json!(1),&serde_json::json!(1.0)));assert!(!exact(&serde_json::json!(true),&serde_json::json!(1)));}
}

pub fn bounded_file_bytes(bytes:u64)->Result<(),String>{require(bytes<=MAX_FILE_BYTES,"bounded raw file required")}
#[cfg(test)]mod zero_source_tests{use super::*;#[test]fn empty_source_raw_identity_is_allowed_semantic_json_stays_rejected(){assert!(bounded_file_bytes(0).is_ok());assert!(bounded_file_bytes(MAX_FILE_BYTES).is_ok());assert!(bounded_file_bytes(MAX_FILE_BYTES+1).is_err());assert!(unique_json(b"").is_err());assert_eq!(sha::hex(b"").unwrap(),"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");}}
