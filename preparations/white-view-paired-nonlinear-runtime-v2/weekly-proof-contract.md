# 週次candidate専用 technical proof

`scripts/weekly_nonlinear_proof.py`は週次学習親の新completionを入口にする。
旧21-role candidate gate、旧manifest固定のoriginal_rows、旧candidate proof receiptを
新MODE/datasetへ読み替えない。source-onlyの実装到達点と実proof成功を区別する。

## 実行入口

```sh
python3 scripts/weekly_nonlinear_proof.py \
  --completion /absolute/private/weekly-training-parent/completion.json \
  --expected-completion-sha256 COMPLETION_SHA \
  --engine-identity /home/server/.local/share/sekirei-weight2/suisho11beta-sekirei-v0.3.39-white-view-v1/white-view-build-identity.json \
  --expected-engine-identity-sha256 4d14a53eaea25913af586ef6ded0c26c84497e129da8844d0fa15348fe2c3e10 \
  --numeric-build-contract /home/server/.local/share/sekirei-weight2/campaign-17-autonomous-v1/white-view-runtime-build-enabled-v1/white_view_build_contract.py \
  --output /absolute/private/fresh-weekly-proof-root \
  --stock-runtime /absolute/fixed-stock-runtime
```

`--prepare-only`はcomplete E3の再検証後、activated SOURCEコピーと
`source-activation.json`と下記の公開入力relocation receiptを作り、compile/probeを開始しない。
inspection用の出力は保存する。実proofは別のfresh outputへ実行する。
reuse/resumeや失敗receiptの上書きは行わない。

## 入力とSOURCE

入口はexternally pinned `sekirei.weekly-nonlinear-training-completion.v1`。
新source binding/parent/plan/buildの実consumerを再実行し、completion exactkeys、
recipe/CLI40argument、112681×3epoch/global step338043、wait/reap/two PG scans、
source membership、before/after raw size/SHA、child contextを再検証する。
固定gate/proof/build pure source SHAは学習親と同じ3fileを保持する。
checkpointの全state/param/m/v、reference03全byte再構成、nearest03全byte再export、
FTZ/DAZ/epoch raw log、initialized positive/negative diagnosticsを再検証する。

public `white-view-paired-nonlinear-proof-v1`の3Rust sourceは個別全SHAを固定する。
core/incrementalだけfresh privateコピー内の唯一
`const PROTOTYPE_ONLY:bool=true;`をfalseへ変える。
native contractは全byte保持する。旧native contract内部のMODEL_MODE文字列は
数値probe契約のラベルとして保存し、candidate identityには使わない。
candidate identityは新outerのMODE、recipe、typed context、model/checkpoint/reference SHAへ束縛する。

SOURCE activation schemaは`sekirei.weekly-nonlinear-proof-source-activation.v1`。
exactkeys: schema/status/mode/training_completion/sources/native_contract_model_mode/
candidate_identity_bound_by_outer_context/compiled/probe_started/model_adopted/final_used。
status=source-prepared、compiled/probe_started=falseはsource準備時点の記録。
sourcesはcore/incremental/native_contractの3role。それぞれprototype/enabled fullref、
production_code_preserved、prototype_only、sole_deltaを保存する。

## Compile・process・raw consumer

固定White build identity/manifestを元の公開build contractへ通す。
旧manifest原本は変更しない。旧Issue17公開worktreeを起点とするraw入力だけを、
同じ相対パスの現在REPOへ対応させ、旧inputs_before/after両方のsize/SHA一致を要求する。
固定runtime source/compiler/binary/log/dependencyやprivate入力にはrelocationを適用しない。
旧full input mapを生成してからこの公開対応だけを適用し、generic refs collectorに
旧manifest bodyを渡して削除済み公開絶対pathを物理入力と誤認させない。
現在の全input mapをlock下で前後検証し、対応先の欠落・別byte・未固定public入力は拒否する。
`engine-public-input-relocation.json`のschemaは
`sekirei.weekly-nonlinear-engine-public-relocation.v1`、
exactkeys: schema/status/engine_manifest/engine_identity/old_public_root/current_public_root/
relocations/original_inventory/current_inventory/historical_manifest_preserved。
status=same-byte-public-inputs-bound、historical_manifest_preserved=true。
relocationsは旧pathをkey、現在のfullrefをvalueとし、旧pathは読込対象refへ偽装しない。
technical completionは`engine_input_relocation`にこのraw refを保存する。
専用consumerは旧pure full mapから対応を再計算し、receiptの全内容、現在の全byteと
original proof inventoryへのclosureを再検証する。
core rlib、実USI/core fingerprint、全release dependency/source/compilerを前後hash検証し、
fresh binaryだけへ`rustc --edition=2024 -C target-cpu=x86-64-v3 -C opt-level=3
-C panic=abort --extern sekirei_core=FIXED_RLIB -L dependency=FIXED_DEPS SOURCE -o NEW_BINARY`
を実行する。compile deadlineは1200秒。
probe/native contractは正確なmodule名で同じfresh src directoryに置くこともconsumerが確認する。

各子はliteral stdin、独立stdout/stderrを使い、session所有、timeout/cancel時の
cleanup/wait/reap、PG empty二観測を保存する。spawn中signalはhandle取得まで遅延し、
取得後に失敗して回収する。split receipt schemaは
`sekirei.weekly-nonlinear-split-child.v1`。success consumerはobserved-exit、signal=null、
execution/source streams全ref、stdin前後同一を要求する。非zero exitは成功にしない。
stock/White benchmark、source build/training、親heavyのlockを非blockingで保持する。
process allowlistは検証済みparentのraw refから取得し、二scanの除外serviceを
PID/UID/comm/kernel start time/cmdline SHAまで照合する。

compile schemaは`sekirei.weekly-nonlinear-proof-compilation.v1`、status=complete。
exactkeys: schema/status/kind/engine_manifest/engine_identity/compiler/core_rlib/
release_dependencies_before/release_dependencies_after/sources/binary/command/execution/
stdin/stdout/stderr/child_receipt/inputs_before/inputs_after。

## Technical completionとsidecar

success schemaは`sekirei.weekly-nonlinear-model-technical-proof.v1`。
new training refs/context/native/checkpoint/reference/metadata、source_activation、
core/incremental compilation、全raw streams・child receipts・観測前後map・locks・scansを保存する。
保存直前に別の専用consumerがraw byteからSOURCE/compile/lifecycle/全probe payloadを再parseする。

core scopeはtrain112681、fixed holdout5895、既存public fixture15の全118591行。
行順・reference==Python material・全L2/output product/add数・FT prefix・全intermediate有限性・
双方ordinary domain・stored-nearest f32/core bridge<1.001を旧条件のまま検査する。
各core split deadlineは600秒。incremental deadlineは1200秒。
incrementalは8185観測/fixture15/walk16/searchwalk8、capture/promotion/drop/undo/null、
global native全byte load、accumulator==refresh、parent restoration、finite/operation counts/
bridge/native functional boundを元のstrict29key parserで再検証する。
旧linear material difference<=100cpはこの非線形契約に含まれない。

sidecarはnativeの隣の`weights.nearest03.meta.json`へ排他作成する。
`format=sekirei-nnue-output-v1`、`nnue_output=absolute`、`checkpoint_hash=FNV1a(native)`、
baseline=null、native_magic=SEKIRW03を含め、モデルfull SHA/bytes、newMODE、checkpoint、
completion、recipe、source binding、E3/step338043/fresh Adamへ束縛する。
publication中のcancel/failureは今回作成したsidecar/proofだけrejected名へ保存する。
既存outputとの排他open失敗はownershipを取得せず、既存ファイルを移動しない。

technical completionはcore_fullrows_verified/incremental_verified/
protected_material_ft_bits_preserved=true、whole_board_ft_preservation_claimed/
universal_integer_bound_claimed/native_bitexact_covariance_claimed=false。
final_used/model_adopted/adoption_claimedはfalse。
正式pilot/100万node MAE/Top3/採用判断は別の段階で行う。

fixtureは標準ライブラリのsynthetic metadata/1row adapterと短いPython childだけを使う。
モデル・教師data・Rust compile・engine・search・fitを起動しない。

```sh
python3 -B -m unittest discover -s tests -p 'test_weekly_nonlinear_proof.py'
```
