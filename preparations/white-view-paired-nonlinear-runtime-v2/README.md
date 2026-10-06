# 週次改善用の非線形入力経路とfixture修復

Issue #19の新規run用ソース。第8候補の公開compiled snapshotとprivate原本は保持する。
新candidateには独立したMODE、profile/plan/manifestのcompile固定SHA、専用source/buildを使う。
`binding-contract.md`が新JSON作者とRust consumerのfield/型/status契約を定める。

## 修復と変更範囲

旧Python親は`main()`内で読み込んだローカル`g`を、外側のPhysicalBytesとepoch parser
から参照してNameErrorとなった。`scripts/weekly_nonlinear_post_training.py`はgateを
明示引数として受け取る。fullref・raw size/SHA・strict JSON・3epoch/338043更新・
MXCSRの検査を維持する。旧failureを成功に書き換えない。

`paired_nonlinear_native_contract.rs`の変更は`cfg(test)`内だけで、`&NnueWeights`の
`clone()`に依存せず全6fieldを所有するfixtureを作る。production契約関数のbyteは
旧ソースと一致する。追加fixtureは全native byte一致とFT/L2行の独立所有を検査する。
`manifest.json`はこのnative修復2fileと旧base SHAだけの一覧であり、他の新コードが
旧compiled SHAと同じという意味ではない。

`paired_nonlinear_actual.rs`と`paired_nonlinear_parent_binding.rs`は今回の入力経路。
whole-pack hash順位、packごとの全game census/選択数、選択span/rank、producer closure、
frozen holdout、独立1000局除外の新manifest/provenanceを専用schemaで読む。
新inputを旧origin証明へ読み替えない。旧numeric update/Adam/master mirror/poison/
FTZ/nearest03/checkpoint/exportを維持し、fresh seed42・112681×3epochのみに対応する。
学習lossやこの準備の成功を100万ノード比較・モデル採用へ読み替えない。

## source preparationとbuild

`scripts/prepare_weekly_nonlinear.py`は専用fresh runtimeにupstream f09c130をcloneし、
旧complete patchの全postimageをcompiled snapshotと照合してから、新reader/binderを
配置する。CLIのMODEとmainのprofile module登録だけを変更し、compile固定の
`weekly_nonlinear_profile.rs`を生成する。buildやtrainは開始しない。

```sh
python3 scripts/prepare_weekly_nonlinear.py \
  --upstream /absolute/path/to/existing/sekirei/git \
  --output /absolute/path/to/fresh/private/trainer-runtime \
  --plan /absolute/path/to/frozen-weekly-plan.json \
  --expected-plan-sha256 PLAN_SHA \
  --profile /absolute/path/to/frozen-selection-profile.json \
  --expected-profile-sha256 PROFILE_SHA
```

`scripts/weekly_nonlinear_build.py`は固定white engineのcompiler/source identityと
現registry dependencyを前後照合し、新targetだけでrelease/offline/locked/jobs2の
test/buildを実行する。各childのwait/reap/PG停止とraw logを保存し、typed build receiptを
排他作成する。deadline/failure/cancellationは失敗として残し、既存runを上書きしない。
既存stock/white benchmarkと調整する追加lockは明示する。

```sh
python3 scripts/weekly_nonlinear_build.py \
  --source-preparation /absolute/private/runtime/source-preparation.json \
  --expected-source-preparation-sha256 PREPARATION_SHA \
  --engine-manifest /absolute/fixed-white-runtime/build-manifest.json \
  --expected-engine-manifest-sha256 ENGINE_SHA \
  --dependency-reference /absolute/prior-trainer/build/release/deps \
  --coordination-lock /absolute/stock-runtime/.benchmark.lock
```

buildの成功後に新fresh-origin/parent-preflight/source-binding/recipeを作り、別の親から
`verify-paired-nonlinear-inputs`と`train-paired-nonlinear`を順に実行する。
正式候補のcore/incremental proof・own pilot・100万node comparisonは別途必要。

週次post-training technical proofの専用helperとraw consumerは
`scripts/weekly_nonlinear_proof.py`。旧Rust numerical probeのSOURCEを全SHAで固定し、
新completion/contextへ束縛して、全118591 core行・incremental8185・absolute sidecarを検証する。
CLI/receipt/activation条件は[weekly-proof-contract.md](weekly-proof-contract.md)を参照する。
このhelperの実装fixture成功は、実candidateのprobe成功を意味しない。

## 実施済みの小検証

標準ライブラリのPython fixtureはgate明示注入、改変/JSON重複/epoch欠落/型/MXCSR拒否、
builderの正常終了/timeout/cancel中spawn/PG残存/ログ分類を確認する。
新Rust metadata/binderの小harnessは10test、固定white featureの実core rlibへリンクした
native `cfg(test)`は3testがPASSした。既存runtimeはread-onlyで使い、出力はIssue #19の
fixtureディレクトリに保存した。これらはsource/header/controlの証拠であり、
専用trainer全体のcargo compile・新dataset読込・学習の成功ではない。

Issue #19の実v2 dataset/planを固定したfresh trainerでは、専用Cargo testの33件と
release buildが成功した。集計は `docs/validation/weekly-weight-improvement-2026-10-05/trainer-build.json`。
これは実source/build検証であり、新候補の入力消費・学習・正式採用は別工程である。

```sh
python3 -B -m unittest discover -s tests -p 'test_weekly_nonlinear*.py'
python3 -B -m unittest discover \
  -s preparations/white-view-paired-nonlinear-rust-v1 -p test_public_source.py
```
