# Issue #19 専用の入力契約

旧archive・compiled source・過去runを変更しない。新sourceは旧complete patchを
固定upstreamへ適用し、このpackageのactual/parent-bindingを差し替える。
CLI本体とmainの差分はMODE宣言と生成profile module登録だけ。
ニューラル演算、原典scalar Adam、保護material、FTZ/DAZ、nearest03、poison、
fresh seed42、112681×3epoch、全state/export/readbackは旧実装を維持する。

新CLIは旧19組を維持する。`recipe`, `source-binding`, `manifest`, `reference03`,
`positions`, `labels`ごとのpath/bytes/SHAと`output`を要求する。
全JSONは重複key、非finite、型混同、未知fieldを拒否する。

## plan

`schema=sekirei.weekly-nonlinear-selected-plan.v1`, `status=frozen-before-build`。
必須fieldはschema/status/mode/hypothesis/created_utc/architecture/training/
dataset_inputs/teacher_identity/selection_profile/resources/adoption/incumbent_model。
dataset_inputsはmanifest.jsonとtrain/holdoutのpositions/labelsの5fullref。
selection_profile、incumbent_modelもfullref。plan自身、dataset manifest、
selection profileのSHAをRustへcompile時に固定する。

architectureはfeature_schema=flat_white_view_aux_tied_v1、dimensions=[2420,256,32]、
native_magic=SEKIRW03、protected_material_units=[0,1,2,3]、all_ft_bias_fixed_q=64、
hand_ties=[[0,3],[1,2]]、aux_ft_channels=254、paired_heads=14。

trainingはseed=42、epochs=3、selected_epoch=3、train_count=112681、holdout_count=5895、
objective=absolute-cp-mse、optimizer=fresh-adam-tied-masters、learning_rate_schedule=constant、
learning_rate_f32_bits=3a83126f、head_init_width_f32_bits=3b800000、
head_bias_init_f32_bits=40800000、output_native_l1_budget_f32_bits=47000000、
shuffle_seed=null、resume_allowed=false、ft_saved_q_max=797、ft_bias_q=64。

resourcesはheavy_serial=true/build_jobs=2/analysis_jobs=1/threads=1、
training_wall_limit_seconds（正整数、86400以下）、added_storage_budget_bytes、
minimum_ssd_remaining_bytes（いずれも正整数）。

adoptionはdevelopment_games=5/go_nodes=1000000/mae_multipv=1/top3_multipv=3、
rule="strict MAE decrease and no Top3 decrease against latest incumbent"、
final_used_for_daily_selection=false/learning_loss_is_adoption_criterion=false。

## source bindingとparent

SourceBindingはschema=sekirei.weekly-nonlinear-source-binding.v1/status=ready、
mode/selected_plan/parent_preflight/training_build/training_binary/
training_source_files/compiler_files/engine_build/dataset_inputs/reference03/
float_policy=x86-ftz-daz。参照はfullref、source/compilerはpath→bytes/SHA map。

Parentはschema=sekirei.weekly-nonlinear-parent-preflight.v1/status=complete、
mode/selected_plan/training_build/engine_build/dataset_inputs/reference03/
dataset_origin_proof/origin_proof_policy/origin_counts/source_inputs/inputs_before/
inputs_after/checks/process_evidence/source_head。
origin_proof_policyはfresh-whole-pack-ranked-dataset-with-frozen-holdout。
origin_countsはtrain=112681/holdout=5895/acquired_pool_games=1000/
selected_pack_games=sum(profile.packs.selected_games)。
source_inputs=inputs_before=inputs_afterとし、recipe/SB/parent自分への逆参照は禁止。

checksの全fieldはstrict true：fresh_dataset_origin_raw_and_transitive_refs_valid、
fresh_dataset_schema_semantics_and_join_valid、frozen_holdout_and_pool_exclusion_verified、
reference03_fullbytes_reconstructed、new_training_source_build_compiler_abi_bound、
fixed_white_view_engine_unchanged、inputs_before_after_equal、source_membership_equal、
no_conflicting_heavy_process。

process_evidenceはschema=sekirei.weekly-nonlinear-preflight-process-evidence.v1、
status=observed-clear、producer_source/process_reader/allowlist/scan_records/
lock_records/source_head。scan_recordsは2件で各conflicts空・excluded_preexisting_services。
lock_recordsはpath/exclusive/nonblocking=true/acquired=trueの一意な非空配列。

## fresh dataset origin

schema=sekirei.weekly-nonlinear-dataset-origin.v1/status=verified、producer_source/
dataset_inputs/frozen_dataset_inputs/profile/pack_manifest/independent_pool_manifest/
input_hashes/checks/counts/inputs_before/inputs_after/
actual_replay_repeated_by_verifier=false/final_used=false。
frozen_dataset_inputsは元Oの5fullref、dataset_inputsは今回の5fullref。
input_hashesは新manifest.derivation.input_sha256と完全一致するpath→SHA map。
producer_source.SHAはmanifest.derivation.script_sha256と一致する。
inputs_before=inputs_afterは生成元の全raw inputおよびproducerを含み、
新output5件はparent source_inputsに別途含む（前後source mapへ混入しない）。
profile本文は新manifest.derivation.profileと型を含め完全一致。

checksはselection_profile_hash_body_valid/selected_replay_producer_source_bound/
whole_pack_index_membership_valid/frozen_holdout_bytes_preserved/complete_pool_excluded/
split_and_boards_disjoint/raw_input_hashes_unchanged/exact_train_label_join_valid。
countsはparent.origin_countsと一致。
このconsumerは今回producerのsource、manifest、profile、全inputを束縛し、
Rust childがcshogiの選択ゲーム全合法手再生を再実行したとは主張しない。

## training build

schemaだけsekirei.weekly-nonlinear-training-build.v1とする。旧typed buildの
fieldを維持：status=complete/mode/producer_source/source_root/upstream_commit/
source_head/source_files/compiler_files/dependency_files/engine_build/training_binary/
commands/environment/tests/process_outcomes/inputs_before/inputs_after/
scalar_power_cache_used=false。
environmentはCARGO_TARGET_DIR/RUSTC/RUSTFLAGS=-C target-cpu=x86-64-v3/
CARGO_BUILD_JOBS=2。commandはcargoのrelease/offline/locked/jobs2/明示feature付き
testとbuildの両方を要求する。
testsはlog/passed/failed=0/ignored/required_test_names。
process_outcomesはcommand/pid=pgid>0/returncode=0/waited=true/reaped=true/
timed_out=false/group_empty_scans=[true,true]/log/wall_seconds。
各required_test_namesはactual raw logのPASS行へ束縛し、reader/SHA/unique JSON/
native/updateまたはadapter/FTZまたはfloat_policyの各categoryを要求する。
