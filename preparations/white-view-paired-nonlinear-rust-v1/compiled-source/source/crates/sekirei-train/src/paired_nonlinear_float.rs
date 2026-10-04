//! SOURCE ONLY read-back validator for Root's separately activated trainer.
//! No standalone entry; does not configure MXCSR or mutate the environment.
//! Existing configure_training_floats executes first in Root's main variant.
use serde::Serialize;
pub const ENV_NAME:&str="SEKIREI_TRAIN_FTZ_DAZ";
pub const ENV_REQUIRED:&str="1";
pub const STATUS_MASK:u32=0x003f;
pub const CONTROL_REQUIRED:u32=0x9fc0;

#[derive(Clone,Debug,PartialEq,Eq,Serialize)]
pub struct FloatSnapshot {
    pub schema:&'static str,
    pub float_policy:&'static str,
    pub environment_variable:&'static str,
    pub environment_value:String,
    pub mxcsr_raw_bits:u32,
    pub mxcsr_control_bits:u32,
    pub mxcsr_status_bits:u32,
    pub mxcsr_status_mask:u32,
    pub required_control_bits:u32,
}
/// Memory-only fixture/API. This validates a supplied word; it does not prove
/// that the word was read from a CPU. Use runtime_ready for actual read-back.
pub fn validate_control_word(environment:&str,raw:u32)->Result<FloatSnapshot,String>{
    if environment!=ENV_REQUIRED {return Err("SEKIREI_TRAIN_FTZ_DAZ must be exactly 1".into());}
    let control=raw & !STATUS_MASK;
    if control!=CONTROL_REQUIRED {return Err(format!("MXCSR control mismatch: raw={raw:#010x}, control={control:#010x}, expected={CONTROL_REQUIRED:#010x}"));}
    Ok(FloatSnapshot{schema:"sekirei.white-view-paired-nonlinear-float-snapshot.v1",
        float_policy:"x86-ftz-daz",environment_variable:ENV_NAME,
        environment_value:environment.into(),mxcsr_raw_bits:raw,
        mxcsr_control_bits:control,mxcsr_status_bits:raw & STATUS_MASK,
        mxcsr_status_mask:STATUS_MASK,required_control_bits:CONTROL_REQUIRED})
}
#[cfg(target_arch="x86_64")]
fn actual_mxcsr_readback()->u32{
    let mut raw:u32=0;
    // STMXCSR writes exactly four bytes to this live, aligned u32. The default
    // asm memory effects are retained: no nomem/readonly/pure claim. It neither
    // changes MXCSR nor any architectural integer condition flags.
    unsafe { core::arch::asm!("stmxcsr [{slot}]",slot=in(reg) &mut raw,
        options(nostack,preserves_flags)); }
    raw
}
/// Actual observation on the calling thread; no CPU policy is set here.
/// Root's main must configure first, and each epoch boundary re-calls this.
pub fn snapshot()->Result<FloatSnapshot,String>{
    #[cfg(not(target_arch="x86_64"))]
    {Err("paired nonlinear float policy requires x86_64 actual MXCSR read-back".into())}
    #[cfg(target_arch="x86_64")]
    {
        let environment=std::env::var(ENV_NAME).map_err(|_|"SEKIREI_TRAIN_FTZ_DAZ missing or not Unicode".to_string())?;
        if environment!=ENV_REQUIRED {return Err("SEKIREI_TRAIN_FTZ_DAZ must be exactly 1".into());}
        validate_control_word(&environment,actual_mxcsr_readback())
    }
}
pub fn runtime_ready()->Result<FloatSnapshot,String>{snapshot()}

#[cfg(test)]mod tests{
    use super::*;
    // No runtime_ready/snapshot call: all tests are supplied-word fixtures.
    #[test]fn float_policy_all_status_bits_allowed(){
        for status in 0..=STATUS_MASK {
            let v=validate_control_word("1",CONTROL_REQUIRED|status).unwrap();
            assert_eq!(v.mxcsr_raw_bits,CONTROL_REQUIRED|status);
            assert_eq!(v.mxcsr_control_bits,CONTROL_REQUIRED);
            assert_eq!(v.mxcsr_status_bits,status);
        }
    }
    #[test]fn float_policy_every_control_bit_changed_rejected(){
        for bit in 6..32 {assert!(validate_control_word("1",CONTROL_REQUIRED^(1u32<<bit)).is_err());}
    }
    #[test]fn float_policy_environment_exact(){
        for value in ["","0","true","TRUE","1 "," 1","1\n","01","１"] {
            assert!(validate_control_word(value,CONTROL_REQUIRED).is_err());
        }
        assert!(validate_control_word("1",CONTROL_REQUIRED).is_ok());
    }
    #[test]fn float_policy_snapshot_structured(){
        let v=validate_control_word("1",CONTROL_REQUIRED|1).unwrap();
        let j=serde_json::to_value(&v).unwrap();
        assert_eq!(j["mxcsr_raw_bits"].as_u64(),Some((CONTROL_REQUIRED|1)as u64));
        assert_eq!(j["mxcsr_control_bits"].as_u64(),Some(CONTROL_REQUIRED as u64));
        assert_eq!(j["environment_value"].as_str(),Some("1"));
        assert_eq!(j.as_object().unwrap().len(),9);
    }
}
