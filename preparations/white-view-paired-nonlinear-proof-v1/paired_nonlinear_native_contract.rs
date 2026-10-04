//! SOURCE ONLY standalone native contract for the selected nonlinear model.
//! No Trainer/Adam state access; no native I/O or source/build identity claim.
use sekirei_core::nnue::{NnueWeights, INPUT, L1, L2};
pub const MAGIC: &[u8;8]=b"SEKIRW03";
pub const NATIVE_BYTES:usize=1_305_356;
pub const MODEL_MODE:&str="white-view-paired-nonlinear-adam-e3-v1";
pub const RECIPE_BITS:[u32;4]=[0x3a83126f,0x3b800000,0x40800000,0x47000000];
pub const EPOCHS:u32=3;
pub const STEP:u64=338_043;
const BOARD:usize=2268;
const HAND:usize=38;
const VALUES:[i16;14]=[100,430,470,640,680,890,1040,0,600,600,600,640,1150,1300];
const HAND_OFFSETS:[usize;7]=[0,18,22,26,30,34,36];
const HAND_MAX:[usize;7]=[18,4,4,4,4,2,2];
fn require(ok:bool,msg:&'static str)->Result<(),&'static str>{if ok{Ok(())}else{Err(msg)}}
fn material_group(kind:usize)->Option<usize>{if kind==7{None}else{Some(if kind==0||kind==8{0}else{1})}}
fn same(a:f32,b:f32)->bool{a.to_bits()==b.to_bits()}
fn positive_zero(x:f32)->bool{x.to_bits()==0}
fn same_weights(a:&NnueWeights,b:&NnueWeights)->bool{
 a.ft==b.ft&&a.ft_bias==b.ft_bias&&a.l2.len()==b.l2.len()
 &&a.l2.iter().flatten().zip(b.l2.iter().flatten()).all(|(&x,&y)|same(x,y))
 &&a.l2_bias.iter().zip(&b.l2_bias).all(|(&x,&y)|same(x,y))
 &&a.out.iter().zip(&b.out).all(|(&x,&y)|same(x,y))&&same(a.out_bias,b.out_bias)
}
pub fn canonical_reference_native03() -> Result<NnueWeights, &'static str> {
    require((INPUT,L1,L2) == (2420,256,32), "wrong full architecture")?;
    let mut rng = 42_u64;
    let mut draw = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        if rng >> 63 == 1 { 1_i16 } else { -1_i16 }
    };
    let mut ft = vec![[0_i16; L1]; INPUT];
    for feature in 0..INPUT {
        for j in 2..L1 { ft[feature][j] = draw(); }
        let (kind, own) = if feature < BOARD {
            ((feature % 28)/2, feature % 2 == 0)
        } else {
            let bank = (feature-BOARD)/HAND;
            let threshold = (feature-BOARD)%HAND;
            let kind = (0..7).find(|&k| threshold >= HAND_OFFSETS[k]
                && threshold < HAND_OFFSETS[k]+HAND_MAX[k]).ok_or("hand threshold")?;
            (kind, bank/2 == bank%2)
        };
        if own { if let Some(group) = material_group(kind) { ft[feature][group] = VALUES[kind]/2; } }
    }
    let mut l2 = vec![[0_f32; L2]; 2*L1];
    l2[0][0]=1.0; l2[1][1]=1.0; l2[L1][2]=1.0; l2[L1+1][3]=1.0;
    for row in 0..2*L1 {
        if row%L1 < 2 { continue; }
        for col in 4..L2 { l2[row][col] = draw() as f32 / 256.0; }
    }
    for threshold in 0..HAND {
        ft[BOARD+3*HAND+threshold] = ft[BOARD+threshold];
        ft[BOARD+2*HAND+threshold] = ft[BOARD+HAND+threshold];
    }
    let mut bias = [4.0_f32; L2]; bias[..4].fill(0.0);
    let mut out = [0.0_f32; L2]; out[..4].copy_from_slice(&[8192.0,8192.0,-8192.0,-8192.0]);
    Ok(NnueWeights { ft, ft_bias:[64_i16;L1], l2, l2_bias:bias, out, out_bias:0.0 })
}

/// Independent ABI reconstruction of the ACTUAL core-loaded object. Full file
/// byte equality is a reader/encoding observation, not actual core save_weights.
pub fn native03_layout_bytes(w:&NnueWeights)->Result<Vec<u8>,&'static str>{
 require(w.ft.len()==INPUT&&w.l2.len()==2*L1,"full native layout shape")?;
 let mut bytes=Vec::with_capacity(NATIVE_BYTES);bytes.extend_from_slice(MAGIC);
 for row in &w.ft{for &x in row{bytes.extend_from_slice(&x.to_le_bytes());}}
 for &x in &w.ft_bias{bytes.extend_from_slice(&x.to_le_bytes());}
 for row in &w.l2{for &x in row{bytes.extend_from_slice(&x.to_le_bytes());}}
 for &x in &w.l2_bias{bytes.extend_from_slice(&x.to_le_bytes());}
 for &x in &w.out{bytes.extend_from_slice(&x.to_le_bytes());}
 bytes.extend_from_slice(&w.out_bias.to_le_bytes());require(bytes.len()==NATIVE_BYTES,"native byte layout size")?;Ok(bytes)
}

fn next_up_positive(x:f64)->Result<f64,&'static str>{
 require(x.is_finite()&&x>=0.0,"invalid finite upper bound")?;
 let y=f64::from_bits(x.to_bits()+1);require(y.is_finite(),"upper bound overflow")?;Ok(y)
}
fn add_up(a:f64,b:f64)->Result<f64,&'static str>{next_up_positive(a+b)}
fn mul_up(a:f64,b:f64)->Result<f64,&'static str>{next_up_positive(a*b)}
fn div_up(a:f64,b:f64)->Result<f64,&'static str>{require(b.is_finite()&&b>0.0,"bound divisor")?;next_up_positive(a/b)}
pub struct NativeSafety {pub q_l1_upper:f64,pub absolute_cp_upper:f64}
/// Native parameters only. Full parameter/m/v/step validation remains mandatory
/// in the dedicated checkpoint module; this is NOT a substitute for that proof.
pub fn validate_pair(candidate:&NnueWeights,reference:&NnueWeights)->Result<NativeSafety,&'static str>{
 require((INPUT,L1,L2)==(2420,256,32),"wrong full architecture")?;
 for w in [candidate,reference]{
  require(w.ft.len()==INPUT&&w.l2.len()==2*L1,"native shape")?;
  require(w.l2.iter().flatten().chain(&w.l2_bias).chain(&w.out).all(|x|x.is_finite())&&w.out_bias.is_finite(),"native nonfinite tail")?;
 }
 let expected=canonical_reference_native03()?;
 require(same_weights(reference,&expected),"reference03 differs from independent seed42 full reconstruction")?;
 require(candidate.ft_bias==reference.ft_bias,"protected q64 FT bias changed")?;
 for feature in 0..INPUT{
  require(candidate.ft[feature][..2]==reference.ft[feature][..2],"protected material FT changed")?;
  require(candidate.ft[feature][2..].iter().all(|&x|(x as i32).abs()<=797),"aux FT outside nearest797 domain")?;
 }
 for t in 0..HAND{
  require(candidate.ft[BOARD+t]==candidate.ft[BOARD+3*HAND+t]&&candidate.ft[BOARD+HAND+t]==candidate.ft[BOARD+2*HAND+t],"full256 hand tie changed")?;
 }
 for row in 0..2*L1{for o in 0..L2{
  if o<4{require(same(candidate.l2[row][o],reference.l2[row][o]),"protected material L2 changed")?;}
  if row%L1<2&&o>=4{require(positive_zero(candidate.l2[row][o]),"material input reaches aux head")?;}
 }}
 for o in 0..4{
  require(same(candidate.l2_bias[o],reference.l2_bias[o])&&same(candidate.out[o],reference.out[o]),"protected material bias/output changed")?;
 }
 require(positive_zero(candidate.out_bias),"output bias must be +zero")?;
 let mut q_l1=0.0_f64;
 for pair in 0..14{
  let plus=4+2*pair;let minus=plus+1;
  for j in 2..L1{
   require(same(candidate.l2[L1+j][minus],candidate.l2[j][plus])&&same(candidate.l2[j][minus],candidate.l2[L1+j][plus]),"paired L2 view exchange mismatch")?;
  }
  require(same(candidate.l2_bias[minus],candidate.l2_bias[plus]),"paired bias mismatch")?;
  require(candidate.out[minus].to_bits()==candidate.out[plus].to_bits()^0x8000_0000,"paired output signbit mismatch")?;
  q_l1=add_up(q_l1,candidate.out[plus].abs()as f64)?;
 }
 require(q_l1<=f32::from_bits(RECIPE_BITS[3])as f64,"selected Q32768 output L1 exceeded")?;
 let raw=add_up(1_682_688.0,mul_up(254.0,q_l1)?)?;
 let factor=div_up(1.0,1.0-2.0_f64.powi(-18))?;
 let cp=add_up(div_up(mul_up(raw,factor)?,64.0)?,1.0)?;
 require(cp<899_000.0,"selected output enters normal/mate boundary")?;
 let factor=div_up(1.0,1.0-2.0_f64.powi(-14))?;
 for o in 4..L2{
  let mut sum=0.0_f64;for row in 0..2*L1{sum=add_up(sum,candidate.l2[row][o].abs()as f64)?;}
  let c=add_up(candidate.l2_bias[o].abs()as f64,mul_up(127.0,sum)?)?;
  require(add_up(mul_up(c,factor)?,1e-30)?<f32::MAX as f64,"native L2 partial-sum bound can overflow")?;
 }
 require(64-41*797>=i16::MIN as i32&&64+41*797<=i16::MAX as i32,"41-prefix domain unsafe")?;
 Ok(NativeSafety{q_l1_upper:q_l1,absolute_cp_upper:cp})
}
#[cfg(test)]mod tests{use super::*;
 fn paired_zero(r:&NnueWeights)->NnueWeights{let mut w=r.clone();for pair in 0..14{let p=4+2*pair;let n=p+1;for j in 2..L1{w.l2[L1+j][n]=w.l2[j][p];w.l2[j][n]=w.l2[L1+j][p];}w.l2_bias[n]=w.l2_bias[p];w.out[n]=f32::from_bits(w.out[p].to_bits()^0x8000_0000);}w}

 #[test]fn protected_material_and_aux_domain_are_distinct(){let r=canonical_reference_native03().unwrap();let mut w=paired_zero(&r);validate_pair(&w,&r).unwrap();w.ft[0][2]=798;assert!(validate_pair(&w,&r).is_err());w=paired_zero(&r);w.ft[0][0]+=1;assert!(validate_pair(&w,&r).is_err());}
 #[test]fn head_and_hand_mirror_are_bit_exact(){let r=canonical_reference_native03().unwrap();let mut w=paired_zero(&r);w.l2[2][4]=0.5;w.l2[L1+2][5]=0.5;w.l2[2][5]=0.25;w.l2[L1+2][4]=0.25;w.out[4]=1.0;w.out[5]=-1.0;validate_pair(&w,&r).unwrap();w.out[5]=1.0;assert!(validate_pair(&w,&r).is_err());}
}
