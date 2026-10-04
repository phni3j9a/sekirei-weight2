// ADD-ONLY SOURCE test module. Root appends this file to paired_nonlinear_positions.rs.
// Root alone compiles/runs with SEKIREI_TRAIN_FTZ_DAZ=1; no production API changes.
// One public fixture x three epochs is not the original 112681-position fit.
#[cfg(test)]
mod fullshape_synthetic_epoch_tests {
    use super::*;
    use crate::trainer::paired_nonlinear_checkpoint_io as checkpoint;
    use crate::trainer::paired_nonlinear_checkpoint_native as native;
    use crate::positions::PositionSample;
    use sekirei_core::sfen::board_to_sfen;
    use std::collections::HashMap;

    const PUBLIC_AFTER_7G7F: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 2";

    fn selected_recipe() -> paired::DeclaredRecipe {
        paired::DeclaredRecipe {
            learning_rate: f32::from_bits(0x3a83126f),
            head_init_width: f32::from_bits(0x3b800000),
            head_bias_init: f32::from_bits(0x40800000),
            output_native_l1_budget: f32::from_bits(0x47000000),
        }
    }

    fn actual_float_policy() {
        // The test harness does not call main. Use the existing main setting
        // then independently read the calling thread's actual control word.
        // Do not set the environment or invent a second FP implementation.
        assert_eq!(std::env::var("SEKIREI_TRAIN_FTZ_DAZ").unwrap(), "1");
        crate::configure_training_floats();
        let snapshot = crate::paired_nonlinear_float::runtime_ready().unwrap();
        assert_eq!(snapshot.environment_value, "1");
        assert_eq!(snapshot.mxcsr_control_bits, 0x9fc0);
        assert_eq!(snapshot.mxcsr_raw_bits & !0x3f, 0x9fc0);
    }

    fn public_sample() -> PositionSample {
        PositionSample {
            board: Board::from_sfen(PUBLIC_AFTER_7G7F).unwrap(),
            phase: "opening".into(), side_to_move: "white".into(), ply: 2,
            source: "public-technical-fixture-after-7g7f".into(),
        }
    }

    fn ordinary_cache(sample: &PositionSample) -> HashMap<String, i32> {
        HashMap::from([(board_to_sfen(&sample.board), 500)])
    }

    fn memory_checkpoint_context() -> checkpoint::CheckpointContext {
        // Identity-shape fixture only. No file with this path is read or made;
        // it is not a provenance declaration for an actual fit.
        let fullref = serde_json::json!({"path":"/tmp/public-synthetic-epoch-context", "bytes":1,
            "sha256":"ab".repeat(32)});
        checkpoint::CheckpointContext {
            recipe:fullref.clone(), source_binding:fullref.clone(), manifest:fullref.clone(),
            reference03:fullref.clone(), positions:fullref.clone(), labels:fullref,
            source_files:serde_json::json!({"/tmp/public-synthetic-epoch-context":{
                "bytes":1,"sha256":"ab".repeat(32)}}),
        }
    }

    #[test]
    fn fullshape_synthetic_three_epochs_float_policy_native_and_no_formal_export() {
        actual_float_policy();
        let recipe=selected_recipe();
        let (mut trainer,reference)=checkpoint::fresh_trainer(recipe).unwrap();
        assert_eq!(trainer.paired_optimizer_step(),0);
        trainer.paired_config_guard().unwrap();
        native::validate_checkpoint_state(&trainer.weights,recipe,
            native::SnapshotStage::Initialized,native::FEATURE_TAG).unwrap();
        let mut context=Context::from_verified_declaration(reference,recipe).unwrap();
        let samples=[public_sample()];
        let cache=ordinary_cache(&samples[0]);
        for epoch in 1..=3_u32 {
            actual_float_policy();
            let token=trainer.train_paired_nonlinear_epoch(&samples,&cache,epoch,&mut context).unwrap();
            assert_eq!(token.epoch(),epoch);
            assert_eq!(token.positions(),1);
            assert_eq!(token.end_step(),u64::from(epoch));
            assert_eq!(trainer.paired_optimizer_step(),u64::from(epoch));
            native::validate_checkpoint_state(&trainer.weights,recipe,
                native::SnapshotStage::AfterPosition{expected_step:u64::from(epoch)},native::FEATURE_TAG).unwrap();
            actual_float_policy();
        }
        let reference=context.into_completed_reference().unwrap();
        paired::validate_state(&trainer.weights,&reference,recipe).unwrap();
        let stage=native::SnapshotStage::AfterPosition{expected_step:3};
        let nearest=native::nearest_native03(&trainer.weights,recipe,stage,native::FEATURE_TAG).unwrap();
        let raw=native::encode_native03_layout(&nearest).unwrap();
        native::verify_native03_reexport(&trainer.weights,recipe,stage,native::FEATURE_TAG,&raw).unwrap();
        // The explicit memory diagnostic at step3 is not the production
        // checkpoint/native export, whose full-fit end remains step338043.
        assert!(checkpoint::completed_checkpoint_bytes(&trainer,recipe,&memory_checkpoint_context()).is_err());
        assert!(checkpoint::completed_nearest_bytes(&trainer,recipe).is_err());
        assert!(checkpoint::initialized_checkpoint_bytes(&trainer,recipe,&memory_checkpoint_context()).is_err());
        assert!(checkpoint::initialized_native_bytes(&trainer,recipe).is_err());
    }

    #[test]
    fn fullshape_synthetic_teacher_failure_poison_forbids_next_epoch_and_partial_export() {
        actual_float_policy();
        // Missing keyed value despite matching cardinality; both signed mate
        // thresholds and i32::MIN exercise the integer absolute-domain guard.
        for bad_teacher in [None,Some(30000_i32),Some(-30000_i32),Some(i32::MIN)] {
            let recipe=selected_recipe();
            let (mut trainer,reference)=checkpoint::fresh_trainer(recipe).unwrap();
            let mut context=Context::from_verified_declaration(reference,recipe).unwrap();
            let samples=[public_sample()];
            let good=ordinary_cache(&samples[0]);
            let token=trainer.train_paired_nonlinear_epoch(&samples,&good,1,&mut context).unwrap();
            assert_eq!(token.positions(),1); assert_eq!(token.end_step(),1);
            let before_failure=trainer.weights.clone();
            let bad=match bad_teacher {
                None=>HashMap::from([("public-synthetic-unmatched-key".to_string(),500)]),
                Some(cp)=>HashMap::from([(board_to_sfen(&samples[0].board),cp)]),
            };
            assert_eq!(bad.len(),samples.len());
            assert!(trainer.train_paired_nonlinear_epoch(&samples,&bad,2,&mut context).is_err());
            assert!(context.ensure_live().is_err());
            assert!(native::checkpoint_state_equal(&before_failure,&trainer.weights));
            assert_eq!(trainer.paired_optimizer_step(),1);
            // Repairing the cache cannot turn the terminal partial attempt
            // into a resumable one. No second update is allowed.
            assert!(trainer.train_paired_nonlinear_epoch(&samples,&good,2,&mut context).is_err());
            assert!(native::checkpoint_state_equal(&before_failure,&trainer.weights));
            assert!(checkpoint::completed_checkpoint_bytes(&trainer,recipe,&memory_checkpoint_context()).is_err());
            assert!(checkpoint::completed_nearest_bytes(&trainer,recipe).is_err());
            assert!(checkpoint::initialized_checkpoint_bytes(&trainer,recipe,&memory_checkpoint_context()).is_err());
            assert!(checkpoint::initialized_native_bytes(&trainer,recipe).is_err());
            assert!(context.into_completed_reference().is_err());
            actual_float_policy();
        }
    }
}
