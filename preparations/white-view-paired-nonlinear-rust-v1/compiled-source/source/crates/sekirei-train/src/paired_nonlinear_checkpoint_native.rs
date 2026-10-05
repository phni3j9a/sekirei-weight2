//! SOURCE ONLY. Intended sibling child module of the dedicated trainer.
//! Full in-memory checkpoint validation and independent nearest-even native03.
//! No filesystem, CLI, JSON checkpoint reader/writer, hash verifier or caller
//! integration. runtime_entry is permanently disabled. Rust was NOT compiled.

use super::{TrainWeights, INPUT, L1, L2};
use super::paired_nonlinear_adapter::{self as update, DeclaredRecipe};
use sekirei_core::nnue::NnueWeights;

pub const PROTOTYPE_ONLY: bool = true;
pub const FEATURE_TAG: &str = "flat_white_view_aux_tied_v1";
pub const MAGIC: &[u8; 8] = b"SEKIRW03";
pub const NATIVE_BYTES: usize = 1_305_356;
const BOARD: usize = 2268;
const HAND: usize = 38;
const VALUES: [i16; 14] = [100,430,470,640,680,890,1040,0,600,600,600,640,1150,1300];
const HAND_OFFSETS: [usize; 7] = [0,18,22,26,30,34,36];
const HAND_MAX: [usize; 7] = [18,4,4,4,4,2,2];

/// Explicit checkpoint position, NOT an epoch/resume provenance assertion.
#[derive(Clone, Copy)]
pub enum SnapshotStage {
    Initialized,
    AfterPosition { expected_step: u64 },
}

pub fn runtime_entry() -> Result<(), &'static str> {
    Err("PROTOTYPE_ONLY: checkpoint I/O, source/recipe SHA and caller integration missing")
}

fn require(ok: bool, message: &'static str) -> Result<(), &'static str> {
    if ok { Ok(()) } else { Err(message) }
}
fn same(a: f32, b: f32) -> bool { a.to_bits() == b.to_bits() }
fn positive_zero(value: f32) -> bool { value.to_bits() == 0 }
fn sign(value: f32) -> f32 { f32::from_bits(value.to_bits() ^ 0x8000_0000) }
fn material_group(kind: usize) -> Option<usize> {
    if kind == 7 { None } else { Some(if kind == 0 || kind == 8 { 0 } else { 1 }) }
}

/// Pure reconstruction of scripts/material_init.py seed42 followed by the
/// white-view ABI donor0->3 / donor1->2 full-row transform. Preserves every RNG
/// draw before tying. Its actual Rust/core serialization still needs proof.
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

/// All redundant TrainWeights parameter/m/v arrays are checked by the pinned
/// update validator: full shape, finite, nonnegative v, material params/m/v,
/// FT biases, 797 box, all256 hand ties, 14 head pairs, signed output/m and
/// identical v, head intermediate and explicit output L1 budget. The material
/// reference is reconstructed here, rather than trusting an arbitrary caller.
pub fn validate_checkpoint_state(w: &TrainWeights, recipe: DeclaredRecipe,
    stage: SnapshotStage, feature_tag: &str) -> Result<(), &'static str>
{
    require(feature_tag == FEATURE_TAG, "wrong feature tag")?;
    let native = canonical_reference_native03()?;
    let reference = TrainWeights::from_nnue_weights(&native);
    update::validate_state(w,&reference,recipe)?;
    match stage {
        SnapshotStage::Initialized => {
            require(w.step == 0, "fresh step must be zero")?;
            let mut expected=reference.clone();
            update::initialize_head(&mut expected,&reference,recipe)?;
            require(checkpoint_state_equal(w,&expected), "fresh full state differs from seed42/explicit init recipe")?;
            for feature in 0..INPUT {
                for j in 2..L1 {
                    let i=feature*L1+j;
                    require(positive_zero(w.ft_m[i]) && positive_zero(w.ft_v[i]), "fresh FT moments")?;
                    require(same(w.ft[i],reference.ft[i]), "fresh seed42 FT changed")?;
                }
            }
            for pair in 0..14 {
                let plus=4+2*pair;
                for j in 2..L1 {
                    for row in [j,L1+j] {
                        let i=row*L2+plus;
                        require(positive_zero(w.l2_m[i]) && positive_zero(w.l2_v[i]), "fresh head moments")?;
                    }
                }
                require(positive_zero(w.l2bias_m[plus]) && positive_zero(w.l2bias_v[plus]), "fresh bias moments")?;
                require(positive_zero(w.out[plus]) && positive_zero(w.out_m[plus])
                    && positive_zero(w.out_v[plus]), "fresh output master must be +zero")?;
            }
        }
        SnapshotStage::AfterPosition { expected_step } => {
            require(expected_step > 0 && w.step == expected_step, "checkpoint global step differs")?;
        }
    }
    Ok(())
}

/// Nearest ties-to-even for an already restored finite f32 aux FT value.
/// NO saturating cast/clamp: out-of-box input fails rather than being repaired.
pub fn nearest_aux_i16(value: f32) -> Result<i16, &'static str> {
    require(value.is_finite() && value.abs() <= 797.0/64.0, "FT outside finite 797 box")?;
    let scaled=value*64.0; // exact power-of-two scaling inside this range
    let lower=scaled.floor(); let fraction=scaled-lower;
    let rounded=if fraction < 0.5 { lower }
        else if fraction > 0.5 { lower+1.0 }
        else if (lower as i32)%2 == 0 { lower } else { lower+1.0 };
    require(rounded.is_finite() && rounded.abs() <= 797.0, "nearest FT outside prefix domain")?;
    Ok(rounded as i16)
}

fn exact_material_i16(value: f32) -> Result<i16, &'static str> {
    let scaled=value*64.0;
    require(value.is_finite() && scaled.is_finite() && scaled == scaled.trunc()
        && scaled.abs() <= 32767.0, "protected material not exact native integer")?;
    Ok(scaled as i16)
}

pub fn nearest_native03(w: &TrainWeights, recipe: DeclaredRecipe,
    stage: SnapshotStage, feature_tag: &str) -> Result<NnueWeights, &'static str>
{
    validate_checkpoint_state(w,recipe,stage,feature_tag)?;
    let mut ft=vec![[0_i16;L1];INPUT];
    for feature in 0..INPUT {
        for j in 0..L1 {
            let value=w.ft[feature*L1+j];
            ft[feature][j]=if j<2 { exact_material_i16(value)? } else { nearest_aux_i16(value)? };
        }
    }
    let mut l2=vec![[0_f32;L2];2*L1];
    for row in 0..2*L1 { l2[row].copy_from_slice(&w.l2[row*L2..(row+1)*L2]); }
    let mut bias=[0_f32;L2]; bias.copy_from_slice(&w.l2_bias);
    let mut out=[0_f32;L2]; out.copy_from_slice(&w.out);
    let native=NnueWeights { ft,ft_bias:[64_i16;L1],l2,l2_bias:bias,out,out_bias:w.out_bias };
    validate_native_layout(&native)?;
    // Explicit source-level undo_capture transient prefix domain. This is not
    // an actual incremental undo/refresh proof or a float->core bridge claim.
    require(64-41*797 >= i16::MIN as i32 && 64+41*797 <= i16::MAX as i32, "unsafe 41-prefix")?;
    Ok(native)
}

fn validate_native_layout(w: &NnueWeights) -> Result<(), &'static str> {
    require((INPUT,L1,L2)==(2420,256,32) && w.ft.len()==INPUT && w.l2.len()==2*L1, "native shape")?;
    require(w.l2.iter().flatten().chain(&w.l2_bias).chain(&w.out).all(|x|x.is_finite())
        && w.out_bias.is_finite(), "native non-finite tail")?;
    for t in 0..HAND {
        require(w.ft[BOARD+t]==w.ft[BOARD+3*HAND+t]
            && w.ft[BOARD+HAND+t]==w.ft[BOARD+2*HAND+t], "native all256 hand tie")?;
    }
    Ok(())
}

/// In-memory format encoder matching patched core save_weights byte order.
/// Layout validation is not checkpoint functional validation; use the endpoint
/// verify_native03_reexport for that. No call to the original truncate export.
pub fn encode_native03_layout(w: &NnueWeights) -> Result<Vec<u8>, &'static str> {
    validate_native_layout(w)?;
    let mut bytes=Vec::with_capacity(NATIVE_BYTES); bytes.extend_from_slice(MAGIC);
    for row in &w.ft { for &v in row { bytes.extend_from_slice(&v.to_le_bytes()); } }
    for &v in &w.ft_bias { bytes.extend_from_slice(&v.to_le_bytes()); }
    for row in &w.l2 { for &v in row { bytes.extend_from_slice(&v.to_le_bytes()); } }
    for &v in &w.l2_bias { bytes.extend_from_slice(&v.to_le_bytes()); }
    for &v in &w.out { bytes.extend_from_slice(&v.to_le_bytes()); }
    bytes.extend_from_slice(&w.out_bias.to_le_bytes());
    require(bytes.len()==NATIVE_BYTES, "native byte size")?;
    Ok(bytes)
}

/// Native header/size plus FNV-1a only. FNV is an identity checksum, NOT a
/// source/recipe SHA proof, functional validation or actual core load receipt.
pub fn native03_fnv1a(bytes: &[u8]) -> Result<u64, &'static str> {
    require(bytes.len()==NATIVE_BYTES && &bytes[..8]==MAGIC, "FNV requires native03 size/header")?;
    let mut h=14695981039346656037_u64;
    for &byte in bytes { h=(h ^ byte as u64).wrapping_mul(1099511628211); }
    Ok(h)
}

fn take_i16(bytes: &[u8], offset: &mut usize) -> i16 {
    let v=i16::from_le_bytes([bytes[*offset],bytes[*offset+1]]); *offset+=2; v
}
fn take_f32(bytes: &[u8], offset: &mut usize) -> f32 {
    let v=f32::from_le_bytes([bytes[*offset],bytes[*offset+1],bytes[*offset+2],bytes[*offset+3]]);
    *offset+=4; v
}
pub fn decode_native03_layout(bytes: &[u8]) -> Result<NnueWeights, &'static str> {
    require(bytes.len()==NATIVE_BYTES, "native byte size")?;
    require(&bytes[..8]==MAGIC, "native must be SEKIRW03; no retag")?;
    require((INPUT,L1,L2)==(2420,256,32), "wrong full architecture")?;
    let mut offset=8;
    let mut ft=vec![[0_i16;L1];INPUT];
    for row in &mut ft { for v in row { *v=take_i16(bytes,&mut offset); } }
    let mut ft_bias=[0_i16;L1]; for v in &mut ft_bias { *v=take_i16(bytes,&mut offset); }
    let mut l2=vec![[0_f32;L2];2*L1];
    for row in &mut l2 { for v in row { *v=take_f32(bytes,&mut offset); } }
    let mut l2_bias=[0_f32;L2]; for v in &mut l2_bias { *v=take_f32(bytes,&mut offset); }
    let mut out=[0_f32;L2]; for v in &mut out { *v=take_f32(bytes,&mut offset); }
    let out_bias=take_f32(bytes,&mut offset);
    require(offset==NATIVE_BYTES, "native trailing/short bytes")?;
    let weights=NnueWeights{ft,ft_bias,l2,l2_bias,out,out_bias};
    validate_native_layout(&weights)?;
    Ok(weights)
}

/// Full checkpoint state -> nearest native -> decode -> whole bytes equality.
/// Native bytes alone do not contain m/v/step; this does NOT prove a checkpoint
/// JSON roundtrip, a SHA binding, real core serialization, or runtime feature.
pub fn verify_native03_reexport(w: &TrainWeights, recipe: DeclaredRecipe,
    stage: SnapshotStage, feature_tag: &str, supplied: &[u8]) -> Result<(), &'static str>
{
    let decoded=decode_native03_layout(supplied)?;
    require(encode_native03_layout(&decoded)?.as_slice()==supplied, "native decode/encode mismatch")?;
    let expected=nearest_native03(w,recipe,stage,feature_tag)?;
    require(encode_native03_layout(&expected)?.as_slice()==supplied, "full checkpoint nearest native differs")
}

/// Bit-exact all-state in-memory comparison for a future new checkpoint reader.
/// Each vector includes every parameter/m/v slot. No JSON reader is present.
pub fn checkpoint_state_equal(a: &TrainWeights,b: &TrainWeights) -> bool {
    let pairs=[(&a.ft,&b.ft),(&a.ft_bias,&b.ft_bias),(&a.l2,&b.l2),
        (&a.l2_bias,&b.l2_bias),(&a.out,&b.out),(&a.ft_m,&b.ft_m),(&a.ft_v,&b.ft_v),
        (&a.bias_m,&b.bias_m),(&a.bias_v,&b.bias_v),(&a.l2_m,&b.l2_m),(&a.l2_v,&b.l2_v),
        (&a.l2bias_m,&b.l2bias_m),(&a.l2bias_v,&b.l2bias_v),(&a.out_m,&b.out_m),(&a.out_v,&b.out_v)];
    a.step==b.step && same(a.out_bias,b.out_bias) && same(a.obias_m,b.obias_m)
        && same(a.obias_v,b.obias_v) && pairs.iter().all(|(x,y)|
            x.len()==y.len() && x.iter().zip(y.iter()).all(|(&p,&q)|same(p,q)))
}

#[cfg(test)]
mod tests {
    use super::*;
    // Explicit SYNTHETIC fixture only; these values are NOT production defaults.
    fn recipe() -> DeclaredRecipe { DeclaredRecipe { learning_rate:0.001,
        head_init_width:0.03125,head_bias_init:4.0,output_native_l1_budget:1.0 } }
    fn fixture() -> TrainWeights {
        let reference=TrainWeights::from_nnue_weights(&canonical_reference_native03().unwrap());
        let mut state=reference.clone();
        update::initialize_head(&mut state,&reference,recipe()).unwrap(); state
    }
    #[test] fn nearest_ties_negative_zero_and_box() {
        for (x,q) in [(0.5,0),(1.5,2),(2.5,2),(-0.5,0),(-1.5,-2),(-2.5,-2)] {
            assert_eq!(nearest_aux_i16(x/64.0).unwrap(),q);
        }
        assert_eq!(nearest_aux_i16(-0.0).unwrap(),0);
        assert_eq!(nearest_aux_i16(797.0/64.0).unwrap(),797);
        for x in [f32::NAN,f32::INFINITY,798.0/64.0] { assert!(nearest_aux_i16(x).is_err()); }
        assert_eq!((64-41*797,64+41*797),(-32613,32741));
    }
    #[test] fn full_native_roundtrip_and_wrong_size_magic_tail() {
        let state=fixture(); let native=nearest_native03(&state,recipe(),SnapshotStage::Initialized,FEATURE_TAG).unwrap();
        let bytes=encode_native03_layout(&native).unwrap(); assert_eq!(bytes.len(),NATIVE_BYTES);
        assert_eq!(native03_fnv1a(&bytes).unwrap(),native03_fnv1a(&encode_native03_layout(&decode_native03_layout(&bytes).unwrap()).unwrap()).unwrap());
        verify_native03_reexport(&state,recipe(),SnapshotStage::Initialized,FEATURE_TAG,&bytes).unwrap();
        for magic in [b"SEKIRW01",b"SEKIRW02",b"JANOSW03"] {
            let mut wrong=bytes.clone(); wrong[..8].copy_from_slice(magic); assert!(decode_native03_layout(&wrong).is_err());
        }
        assert!(decode_native03_layout(&bytes[..bytes.len()-1]).is_err());
        let mut wrong=bytes.clone(); let n=wrong.len(); wrong[n-4..].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(decode_native03_layout(&wrong).is_err());
    }
    #[test] fn truncate_is_not_nearest_success() {
        let mut state=fixture(); state.step=1; state.ft[2]=1.5/64.0;
        let stage=SnapshotStage::AfterPosition{expected_step:1};
        let nearest=encode_native03_layout(&nearest_native03(&state,recipe(),stage,FEATURE_TAG).unwrap()).unwrap();
        let mut wrong=nearest.clone(); wrong[12..14].copy_from_slice(&1_i16.to_le_bytes());
        assert!(verify_native03_reexport(&state,recipe(),stage,FEATURE_TAG,&wrong).is_err());
    }
    #[test] fn protected_material_and_full_moment_shape_are_rejected() {
        let good=fixture();
        let changes: [fn(&mut TrainWeights);9]=[
            |w|w.ft[0]+=1.0, |w|w.ft_m[0]=1.0, |w|w.ft_v[0]=-0.0,
            |w|w.ft_bias[255]+=1.0, |w|w.bias_m[255]=1.0,
            |w|w.l2[0]+=1.0, |w|w.l2_m[0]=1.0,
            |w|w.out[0]+=1.0, |w|w.obias_v=1.0 ];
        for change in changes { let mut w=good.clone();change(&mut w);
            assert!(validate_checkpoint_state(&w,recipe(),SnapshotStage::Initialized,FEATURE_TAG).is_err()); }
        let mut w=good.clone();w.ft_m.pop();
        assert!(validate_checkpoint_state(&w,recipe(),SnapshotStage::Initialized,FEATURE_TAG).is_err());
    }
    #[test] fn hand_head_bias_output_and_moment_mirrors_rejected() {
        let good=fixture();let changes:[fn(&mut TrainWeights);8]=[
            |w|w.ft[(BOARD+3*HAND)*L1+255]+=0.01,
            |w|w.ft_m[(BOARD+2*HAND)*L1+2]=1.0,
            |w|w.l2[(L1+255)*L2+31]+=0.01,
            |w|w.l2_v[(L1+2)*L2+5]=1.0,
            |w|w.l2_bias[31]+=0.01, |w|w.l2bias_m[5]=1.0,
            |w|w.out[5]=0.0, |w|w.out_m[31]=0.0 ];
        for change in changes {let mut w=good.clone();change(&mut w);
            assert!(validate_checkpoint_state(&w,recipe(),SnapshotStage::Initialized,FEATURE_TAG).is_err());}
    }
    #[test] fn finite_variance_budget_step_and_full_state_equality() {
        let good=fixture();let mut w=good.clone();assert!(checkpoint_state_equal(&good,&w));
        w.step=3;assert!(!checkpoint_state_equal(&good,&w));
        assert!(validate_checkpoint_state(&w,recipe(),SnapshotStage::AfterPosition{expected_step:2},FEATURE_TAG).is_err());
        assert!(validate_checkpoint_state(&w,recipe(),SnapshotStage::AfterPosition{expected_step:3},FEATURE_TAG).is_ok());
        w.ft_v[2]=-1.0;assert!(validate_checkpoint_state(&w,recipe(),SnapshotStage::AfterPosition{expected_step:3},FEATURE_TAG).is_err());
        let mut w=good.clone();w.out[4]=2.0;w.out[5]=sign(2.0);w.step=1;
        assert!(validate_checkpoint_state(&w,recipe(),SnapshotStage::AfterPosition{expected_step:1},FEATURE_TAG).is_err());
        let mut w=good.clone();w.l2[2*L2+4]=f32::INFINITY;
        assert!(validate_checkpoint_state(&w,recipe(),SnapshotStage::Initialized,FEATURE_TAG).is_err());
        assert!(runtime_entry().is_err());
    }
}
