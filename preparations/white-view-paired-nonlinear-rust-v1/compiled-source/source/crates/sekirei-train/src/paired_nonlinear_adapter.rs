//! Disabled conditional source adapter, compiled as a child of trainer.rs.
//! No model/data/recipe I/O, no native serializer, no whole-shape allocation.
//! The parent must keep its original adam_update_scalar unchanged.
//! An update error is terminal: partially updated state MUST NOT be saved or
//! resumed. Actual caller/recipe/hash/serializer integration is not present.

use super::{adam_update_scalar, Lcg, TrainWeights, INPUT, L1, L2};

pub const PROTOTYPE_ONLY: bool = true;
pub const FT_Q_MAX: i32 = 797;
pub const FT_LIMIT: f32 = 797.0 / 64.0;
pub const FT_BIAS: f32 = 1.0; // saved q64
pub const MAX_PREFIX_FEATURES: i32 = 41;
pub const PAIRS: usize = 14;
const BOARD: usize = 2268;
const THRESHOLDS: usize = 38;
const MATERIAL_RAW_ABSOLUTE_SUM: f64 = 1_682_688.0;
const ORDINARY_SCORE_LIMIT: f64 = 899_000.0;

/// Caller-provided values from an externally SHA-pinned future recipe.
/// There are NO production defaults. This declaration is not provenance proof.
#[derive(Clone, Copy, Debug)]
pub struct DeclaredRecipe {
    pub learning_rate: f32,
    pub head_init_width: f32,
    pub head_bias_init: f32,
    pub output_native_l1_budget: f32,
}

pub fn runtime_guard() -> Result<(), &'static str> {
    // Before source/recipe/input reads, feature transforms, or training.
    Err("PROTOTYPE_ONLY: unselected recipe and missing actual adapters")
}

fn require(ok: bool, message: &'static str) -> Result<(), &'static str> {
    if ok { Ok(()) } else { Err(message) }
}

fn positive_zero(x: f32) -> bool { x.to_bits() == 0 }
fn same(a: f32, b: f32) -> bool { a.to_bits() == b.to_bits() }
fn sign_copy(x: f32, negative: bool) -> f32 {
    f32::from_bits(x.to_bits() ^ if negative { 0x8000_0000 } else { 0 })
}

// Positive outward binary64 operations. Do not silently round an exact f32
// L1 budget down. They are intentionally independent of Python/Fraction.
fn next_up_positive(x: f64) -> Result<f64, &'static str> {
    require(x.is_finite() && x >= 0.0, "invalid positive bound")?;
    let y = f64::from_bits(x.to_bits() + 1);
    require(y.is_finite(), "bound overflow")?;
    Ok(y)
}
fn add_up(a: f64, b: f64) -> Result<f64, &'static str> { next_up_positive(a + b) }
fn mul_up(a: f64, b: f64) -> Result<f64, &'static str> { next_up_positive(a * b) }
fn div_up(a: f64, b: f64) -> Result<f64, &'static str> {
    require(b.is_finite() && b > 0.0, "invalid bound divisor")?;
    next_up_positive(a / b)
}

fn score_bound(q_l1: f64) -> Result<f64, &'static str> {
    // 28 aux output terms each <=127*|q|; their absolute sum <=254*Q.
    // 32 products +32 additions covered by gamma64. One extra cp margin.
    let raw = add_up(MATERIAL_RAW_ABSOLUTE_SUM, mul_up(254.0, q_l1)?)?;
    let gamma_factor = div_up(1.0, 1.0 - 2.0_f64.powi(-18))?;
    add_up(div_up(mul_up(raw, gamma_factor)?, 64.0)?, 1.0)
}

pub fn validate_recipe(recipe: DeclaredRecipe) -> Result<(), &'static str> {
    require(recipe.learning_rate.is_finite() && recipe.learning_rate > 0.0,
            "explicit positive LR required")?;
    require(recipe.head_init_width.is_finite() && recipe.head_init_width > 0.0
            && recipe.head_bias_init.is_finite(), "explicit finite asymmetric head init required")?;
    require(recipe.output_native_l1_budget.is_finite() && recipe.output_native_l1_budget > 0.0,
            "explicit finite output budget required")?;
    require(score_bound(recipe.output_native_l1_budget as f64)? < ORDINARY_SCORE_LIMIT,
            "declared output budget enters mate domain")
}

#[derive(Default, Debug)]
pub struct UpdateStats {
    pub ft_master_calls: usize,
    pub head_master_calls: usize,
    pub bias_master_calls: usize,
    pub output_master_calls: usize,
    pub ft_projected: usize,
    pub logical_master_delta_sq: f64,
}

/// Gradients from ONE original forward/backward at the pre-update anchor.
/// d_bias is intentionally not consumed: all FT biases are protected.
pub struct DenseGradients<'a> {
    pub ft: &'a [f32],
    pub l2: &'a [f32],
    pub l2_bias: &'a [f32],
    pub out: &'a [f32],
}

fn finite_state(params: &[f32], m: &[f32], v: &[f32]) -> Result<(), &'static str> {
    require(params.len() == m.len() && params.len() == v.len(), "state length mismatch")?;
    require(params.iter().chain(m).all(|x| x.is_finite())
            && v.iter().all(|x| x.is_finite() && *x >= 0.0), "invalid optimizer state")
}

fn shape(w: &TrainWeights) -> Result<(), &'static str> {
    require((INPUT, L1, L2) == (2420, 256, 32), "wrong architecture")?;
    require(w.ft.len() == INPUT*L1 && w.ft_bias.len() == L1
            && w.l2.len() == 2*L1*L2 && w.l2_bias.len() == L2 && w.out.len() == L2,
            "parameter shape mismatch")?;
    finite_state(&w.ft, &w.ft_m, &w.ft_v)?;
    finite_state(&w.ft_bias, &w.bias_m, &w.bias_v)?;
    finite_state(&w.l2, &w.l2_m, &w.l2_v)?;
    finite_state(&w.l2_bias, &w.l2bias_m, &w.l2bias_v)?;
    finite_state(&w.out, &w.out_m, &w.out_v)?;
    require(w.out_bias.is_finite() && w.obias_m.is_finite()
            && w.obias_v.is_finite() && w.obias_v >= 0.0, "invalid output bias state")
}

fn protected_slot(p: f32, m: f32, v: f32, reference: f32) -> Result<(), &'static str> {
    require(same(p, reference) && positive_zero(m) && positive_zero(v),
            "protected param/m/v changed")
}

fn tied_slot(p: f32, m: f32, v: f32, pm: f32, mm: f32, vm: f32, neg: bool)
    -> Result<(), &'static str>
{
    require(same(p, sign_copy(pm, neg)) && same(m, sign_copy(mm, neg)) && same(v, vm),
            "dependent param/m/v mirror mismatch")
}

/// Reference is externally verified material-initializer03, never arbitrary
/// baseline state. Native SHA/source/ABI/protected bytes binding is caller work.
pub fn validate_state(w: &TrainWeights, reference: &TrainWeights,
                      recipe: DeclaredRecipe) -> Result<(), &'static str> {
    shape(w)?; shape(reference)?; validate_recipe(recipe)?;
    for feature in 0..INPUT {
        for channel in 0..L1 {
            let i=feature*L1+channel;
            if channel < 2 {
                protected_slot(w.ft[i],w.ft_m[i],w.ft_v[i],reference.ft[i])?;
            } else {
                require(w.ft[i].abs() <= FT_LIMIT, "aux FT outside prefix-safe box")?;
            }
        }
    }
    for j in 0..L1 {
        protected_slot(w.ft_bias[j],w.bias_m[j],w.bias_v[j],FT_BIAS)?;
        require(same(reference.ft_bias[j],FT_BIAS), "reference FT bias not q64")?;
    }
    for i in 0..2*L1 {
        for o in 0..L2 {
            let k=i*L2+o;
            if o < 4 || i%L1 < 2 {
                protected_slot(w.l2[k],w.l2_m[k],w.l2_v[k],reference.l2[k])?;
                if o >= 4 { require(positive_zero(reference.l2[k]), "material input into aux head")?; }
            }
        }
    }
    for o in 0..4 {
        protected_slot(w.l2_bias[o],w.l2bias_m[o],w.l2bias_v[o],reference.l2_bias[o])?;
        protected_slot(w.out[o],w.out_m[o],w.out_v[o],reference.out[o])?;
    }
    protected_slot(w.out_bias,w.obias_m,w.obias_v,0.0)?;
    for donor in 0..2 {
        for threshold in 0..THRESHOLDS {
            let d=BOARD+donor*THRESHOLDS+threshold;
            let s=BOARD+(3-donor)*THRESHOLDS+threshold;
            for j in 0..L1 {
                let a=d*L1+j; let b=s*L1+j;
                tied_slot(w.ft[b],w.ft_m[b],w.ft_v[b],w.ft[a],w.ft_m[a],w.ft_v[a],false)?;
            }
        }
    }
    for pair in 0..PAIRS {
        let plus=4+2*pair; let minus=plus+1;
        for j in 2..L1 {
            let u=j*L2+plus; let us=(L1+j)*L2+minus;
            let v=(L1+j)*L2+plus; let vs=j*L2+minus;
            tied_slot(w.l2[us],w.l2_m[us],w.l2_v[us],w.l2[u],w.l2_m[u],w.l2_v[u],false)?;
            tied_slot(w.l2[vs],w.l2_m[vs],w.l2_v[vs],w.l2[v],w.l2_m[v],w.l2_v[v],false)?;
        }
        tied_slot(w.l2_bias[minus],w.l2bias_m[minus],w.l2bias_v[minus],
                  w.l2_bias[plus],w.l2bias_m[plus],w.l2bias_v[plus],false)?;
        tied_slot(w.out[minus],w.out_m[minus],w.out_v[minus],
                  w.out[plus],w.out_m[plus],w.out_v[plus],true)?;
    }
    verify_forward_budget(w,recipe)
}

pub fn verify_forward_budget(w: &TrainWeights, recipe: DeclaredRecipe)
    -> Result<(), &'static str>
{
    let mut q_l1=0.0_f64;
    for pair in 0..PAIRS { q_l1=add_up(q_l1,w.out[4+2*pair].abs() as f64)?; }
    require(q_l1 <= recipe.output_native_l1_budget as f64, "actual output L1 exceeds declared budget")?;
    require(score_bound(q_l1)? < ORDINARY_SCORE_LIMIT, "native output enters mate domain")?;
    // 512 input products +512 additions: gamma1024. a,b clipped <=127.
    // This bounds all partial sums as well as the final preactivation.
    let factor=div_up(1.0,1.0-2.0_f64.powi(-14))?;
    for o in 4..L2 {
        let mut abs_sum=0.0_f64;
        for i in 0..2*L1 { abs_sum=add_up(abs_sum,w.l2[i*L2+o].abs() as f64)?; }
        let c=add_up(w.l2_bias[o].abs() as f64,mul_up(127.0,abs_sum)?)?;
        require(add_up(mul_up(c,factor)?,1e-30)? < f32::MAX as f64,
                "head preactivation can overflow native f32")?;
    }
    Ok(())
}

/// Tiny scalar/group kernel. All gradients (including zero) go through the
/// ORIGINAL parent adam_update_scalar, once. Protected slots never enter here.
fn step_group(params: &mut [f32], m: &mut [f32], v: &mut [f32], gradients: &[f32],
              members: &[(usize,bool)], lr: f32, t: u64, ft_box: bool)
    -> Result<(f32,bool), &'static str>
{
    require(!members.is_empty() && !members[0].1 && params.len()==m.len()
            && params.len()==v.len() && params.len()==gradients.len(), "invalid master group")?;
    require(lr.is_finite() && lr>0.0 && t>0, "invalid scalar update inputs")?;
    let first=members[0].0;
    require(first<params.len(), "master index outside state")?;
    let (old,mut pp,mut mm,mut vv)=(params[first],params[first],m[first],v[first]);
    require(pp.is_finite() && mm.is_finite() && vv.is_finite() && vv>=0.0, "nonfinite scalar state")?;
    let mut gradient=0.0_f32;
    for (ordinal,&(index,negative)) in members.iter().enumerate() {
        require(index<params.len() && members[..ordinal].iter().all(|&(i,_)|i!=index), "duplicate/outside member")?;
        tied_slot(params[index],m[index],v[index],pp,mm,vv,negative)?;
        require(gradients[index].is_finite(), "nonfinite physical gradient")?;
        gradient+=sign_copy(gradients[index],negative);
        require(gradient.is_finite(), "aggregate gradient overflow")?;
    }
    let _delta=adam_update_scalar(&mut pp,&mut mm,&mut vv,gradient,lr,t);
    require(pp.is_finite() && mm.is_finite() && vv.is_finite() && vv>=0.0,
            "master Adam overflow")?;
    let before_projection=pp;
    if ft_box { pp=pp.clamp(-FT_LIMIT,FT_LIMIT); }
    for &(index,negative) in members {
        params[index]=sign_copy(pp,negative); m[index]=sign_copy(mm,negative); v[index]=vv;
    }
    Ok((pp-old,!same(before_projection,pp)))
}

/// Caller uses the original position's t=weights.step AFTER exactly one step
/// increment, and skips ALL original layer Adam calls for this dedicated mode.
/// Backward/source teacher/clamp derivatives are unchanged. Runtime I/O remains
/// disabled. This pure function is source-integration scaffolding, not a run.
pub fn apply_dense_update(w: &mut TrainWeights, reference: &TrainWeights,
                          grads: DenseGradients<'_>, recipe: DeclaredRecipe)
    -> Result<UpdateStats, &'static str>
{
    validate_state(w,reference,recipe)?;
    require(w.step>0 && grads.ft.len()==INPUT*L1 && grads.l2.len()==2*L1*L2
            && grads.l2_bias.len()==L2 && grads.out.len()==L2, "invalid dense gradient shape/global step")?;
    require(grads.ft.iter().chain(grads.l2).chain(grads.l2_bias).chain(grads.out)
            .all(|x|x.is_finite()), "nonfinite backward gradient")?;
    let mut stats=UpdateStats::default(); let t=w.step; let lr=recipe.learning_rate;
    for feature in 0..INPUT {
        if feature>=BOARD && (feature-BOARD)/THRESHOLDS>=2 { continue; }
        for j in 2..L1 {
            let first=feature*L1+j;
            let second=if feature<BOARD { None } else {
                let (bank,th)=((feature-BOARD)/THRESHOLDS,(feature-BOARD)%THRESHOLDS);
                Some((BOARD+(3-bank)*THRESHOLDS+th)*L1+j)
            };
            let result=if let Some(s)=second {
                step_group(&mut w.ft,&mut w.ft_m,&mut w.ft_v,grads.ft,&[(first,false),(s,false)],lr,t,true)?
            } else {
                step_group(&mut w.ft,&mut w.ft_m,&mut w.ft_v,grads.ft,&[(first,false)],lr,t,true)?
            };
            stats.ft_master_calls+=1; stats.ft_projected+=usize::from(result.1);
            stats.logical_master_delta_sq+=(result.0 as f64).powi(2);
        }
    }
    for pair in 0..PAIRS {
        let plus=4+2*pair; let minus=plus+1;
        for j in 2..L1 {
            for members in [[(j*L2+plus,false),((L1+j)*L2+minus,false)],
                            [((L1+j)*L2+plus,false),(j*L2+minus,false)]] {
                let (delta,_)=step_group(&mut w.l2,&mut w.l2_m,&mut w.l2_v,grads.l2,&members,lr,t,false)?;
                stats.head_master_calls+=1; stats.logical_master_delta_sq+=(delta as f64).powi(2);
            }
        }
        let (delta,_)=step_group(&mut w.l2_bias,&mut w.l2bias_m,&mut w.l2bias_v,
            grads.l2_bias,&[(plus,false),(minus,false)],lr,t,false)?;
        stats.bias_master_calls+=1; stats.logical_master_delta_sq+=(delta as f64).powi(2);
        let (delta,_)=step_group(&mut w.out,&mut w.out_m,&mut w.out_v,
            grads.out,&[(plus,false),(minus,true)],lr,t,false)?;
        stats.output_master_calls+=1; stats.logical_master_delta_sq+=(delta as f64).powi(2);
    }
    // No FT-bias, protected material, or output-bias Adam call exists here.
    require((stats.ft_master_calls,stats.head_master_calls,stats.bias_master_calls,stats.output_master_calls)
            == (595376,7112,14,14), "master partition coverage failed")?;
    validate_state(w,reference,recipe)?; // terminal error; no export on failure
    Ok(stats)
}

/// SOURCE init adapter: existing verified seed42 material initializer03 is the
/// input. FT/board/material bytes are not generated or replaced here. The RNG
/// is the ORIGINAL upstream Lcg, seed42^upstream salt; draw order pair/j/u/v.
/// Width/bias/budget are externally supplied, not selected by this package.
pub fn initialize_head(w: &mut TrainWeights, reference: &TrainWeights,
                       recipe: DeclaredRecipe) -> Result<(), &'static str> {
    shape(w)?; shape(reference)?; validate_recipe(recipe)?;
    require(w.step==0, "fresh Adam only; no resume")?;
    for state in [&w.ft_m,&w.ft_v,&w.bias_m,&w.bias_v,&w.l2_m,&w.l2_v,
                  &w.l2bias_m,&w.l2bias_v,&w.out_m,&w.out_v] {
        require(state.iter().all(|&x|positive_zero(x)), "initial moments must be +0")?;
    }
    require(positive_zero(w.obias_m) && positive_zero(w.obias_v), "initial output-bias moments")?;
    require(w.ft.iter().zip(&reference.ft).all(|(&a,&b)|same(a,b)), "initial FT differs from verified seed42 reference")?;
    let mut rng=Lcg(42 ^ 0x9E37_79B9_7F4A_7C15);
    for pair in 0..PAIRS {
        let plus=4+2*pair; let minus=plus+1; let mut asymmetric=false;
        for j in 2..L1 {
            let u=rng.uniform(recipe.head_init_width); let v=rng.uniform(recipe.head_init_width);
            require(u.is_finite() && v.is_finite(), "head RNG overflow")?;
            asymmetric|=!same(u,v);
            for i in [j*L2+plus,(L1+j)*L2+minus] { w.l2[i]=u; }
            for i in [(L1+j)*L2+plus,j*L2+minus] { w.l2[i]=v; }
        }
        require(asymmetric, "all-symmetric zero-output head is dead")?;
        w.l2_bias[plus]=recipe.head_bias_init; w.l2_bias[minus]=recipe.head_bias_init;
        w.out[plus]=0.0; w.out[minus]=sign_copy(0.0,true);
        w.out_m[plus]=0.0; w.out_m[minus]=sign_copy(0.0,true);
    }
    validate_state(w,reference,recipe)
}

/// Guard-side FT quantizer only. Whole native03 serialization is NOT present.
pub fn nearest_aux_ft(value: f32) -> Result<i16, &'static str> {
    require(value.is_finite() && value.abs()<=FT_LIMIT, "unsafe export FT")?;
    let x=value*64.0; let lower=x.floor(); let frac=x-lower;
    let q=if frac<0.5 { lower } else if frac>0.5 { lower+1.0 }
        else if (lower as i32)%2==0 { lower } else { lower+1.0 };
    require(q.abs()<=FT_Q_MAX as f32, "unsafe saved aux FT")?;
    Ok(q as i16)
}

#[cfg(test)]
mod tiny_tests {
    use super::*;
    #[test]
    fn sum_gradient_once_and_variance_of_sum() {
        let (mut p,mut m,mut v)=([1.0_f32,1.0],[0.0_f32;2],[0.0_f32;2]);
        let (mut ep,mut em,mut ev)=(1.0_f32,0.0_f32,0.0_f32);
        adam_update_scalar(&mut ep,&mut em,&mut ev,5.0,0.001,1);
        step_group(&mut p,&mut m,&mut v,&[2.0,3.0],&[(0,false),(1,false)],0.001,1,false).unwrap();
        assert_eq!(p[0].to_bits(),ep.to_bits()); assert_eq!(m[0].to_bits(),em.to_bits());
        assert_eq!(v[0].to_bits(),ev.to_bits()); assert_eq!(v[1].to_bits(),ev.to_bits());
    }
    #[test]
    fn sign_tie_and_inactive_decay() {
        let (mut p,mut m,mut v)=([1.0_f32,-1.0],[0.2_f32,-0.2],[0.3_f32;2]);
        let old=p[0]; let old_m=m[0]; let old_v=v[0];
        step_group(&mut p,&mut m,&mut v,&[0.0,0.0],&[(0,false),(1,true)],0.001,2,false).unwrap();
        assert_eq!(m[0].to_bits(),(0.9_f32*old_m).to_bits());
        assert_eq!(v[0].to_bits(),(0.999_f32*old_v).to_bits()); assert_ne!(p[0].to_bits(),old.to_bits());
        assert_eq!(p[1].to_bits(),p[0].to_bits()^0x8000_0000);
        assert_eq!(m[1].to_bits(),m[0].to_bits()^0x8000_0000); assert_eq!(v[1].to_bits(),v[0].to_bits());
    }
    #[test]
    fn ft_projection_and_prefix() {
        let (mut p,mut m,mut v)=([FT_LIMIT],[0.0_f32],[0.0_f32]);
        step_group(&mut p,&mut m,&mut v,&[-100.0],&[(0,false)],1.0,1,true).unwrap();
        assert_eq!(p[0],FT_LIMIT); assert!(m[0]<0.0 && v[0]>0.0);
        assert_eq!(nearest_aux_ft(p[0]).unwrap(),797); assert_eq!(nearest_aux_ft(-p[0]).unwrap(),-797);
        assert_eq!(64+MAX_PREFIX_FEATURES*FT_Q_MAX,32741);
        assert_eq!(nearest_aux_ft(2.5/64.0).unwrap(),2);
        assert_eq!(nearest_aux_ft(-2.5/64.0).unwrap(),-2);
        assert!(nearest_aux_ft(798.0/64.0).is_err());
    }
    #[test]
    fn entry_is_disabled_and_budget_not_a_default() {
        assert!(PROTOTYPE_ONLY); assert!(runtime_guard().is_err());
        assert!(validate_recipe(DeclaredRecipe { learning_rate:0.0,head_init_width:0.0,
            head_bias_init:0.0,output_native_l1_budget:0.0 }).is_err());
        assert!(score_bound(1_000_000.0).unwrap()>=ORDINARY_SCORE_LIMIT);
    }
}
