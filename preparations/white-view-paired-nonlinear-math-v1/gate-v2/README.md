# 非線形 E3 gate の SOURCE 準備

v2は凍結v1を保持したFP境界だけの派生版。Rustの `next_up_positive/add_up/mul_up/div_up` と同じ順序を使い、9fc0のDAZ operand・FTZ arithmetic result・f32 abs→f64変換を独立したbit oracleで扱う。all14 output master=0、f32 subnormal、最小normal、signed-zero、f64 underflowのfixtureを追加し、零headのq上界は14×minsubnormalではなく1×minsubnormalとする。exact bound比較を範囲許容へ緩めていない。Rust targetとの実bit一致は未compile・未実測で、Rootの実probe evidenceが必須。

技術準備のみ。実データ・モデル・control・runtime・process・NAS は未読で、Rust compile、学習、probe、正式評価、採用を実施していない。`actual_entry()` は常に I/O 前に停止する。旧線形 gate、epochs0/noAdam/100cp 契約、公開 helper は変更していない。

`paired_nonlinear_gate.reconstruct_gate(raw_inputs, binding_raw, expected_binding_sha256)` は呼出側が渡した immutable bytes だけを検証する純粋な consumer である。全 input map の集合・size/SHA、重複 JSON、strict int/bool、canonical lexical path、全 fullref を検証してから body を利用する。物理ファイルの membership、symlink/inode、Git、現在の binary、実 process 状態は検証しない。Root の外側 reader が固定 source を実オブジェクトとして読み、独立した実観測と cooperating lock を持って実行する必要がある。

## 既存 SOURCE との確定接続

- 選択済み plan SHA `111f8e3f1bf404c042c9e3db3e46f46659f3f609c405c30234dbf47778e03729`、mode `white-view-paired-nonlinear-adam-e3-v1`。LR `3a83126f`、width `3b800000`、bias `40800000`、Q `47000000`、seed42、original O TRAIN112681/HOLDOUT5895、3 epochs、E3/338043 steps、shuffle/resume/cache 無しを使用する。
- Root activation SOURCE の `paired_nonlinear_checkpoint_io.rs` が出す `sekirei.white-view-paired-nonlinear-fullstate-bits-checkpoint.v1` をそのまま読む。15 vectors・3 scalars・step の19 fields を検証する。Initialized IO の別 schema/step0 は E3 として受理しない。
- Root `paired_nonlinear_actual.rs` の `sekirei.white-view-paired-nonlinear-export-stage.v1` と `float_export` 9 fields を検証する。子の cleanup/parent_reap/core/incremental flags は False のままであり、親の成功へ書き換えない。
- Root `paired_nonlinear_cli.rs` の `PAIRED_EPOCH_BEGIN` / `PAIRED_EPOCH_COMPLETE` 各3 records を外 SHA に束縛した実 log bytes から再parseする。epoch、開始・終了step、positions、各9-field FP snapshot を対応させる。raw/control/status を区別し、control `0x9fc0`、nearest、FTZ/DAZ、env exactly `1` を検証する。
- 凍結 parent validator v3 と Root SOURCE spec v2 の build/parent/source-binding/prior-origin の exact shape を使用する。1042 は replay/source **files**、1000 は raw games。Root prior-origin producer `5afb4245…` と既存 semantic validator 2本の SOURCE SHA、全推移 input raw identity を束縛する。新しい replay 実行を主張しない。
- 同梱 `paired_nonlinear_proof_contract.py` は凍結 proof v2 の byte 同一コピー（SHA `f25a6763…`）。core の候補/reference 両 stored-native float/core bridge `<1.001cp`、全 FT prefix・L2/output mul/add の finite 観測数、ordinary score `<899000` を維持する。旧 `rho100` 判定は使わない。

## この package が提示する新 producer body

次の親/proof schema は **SOURCE 提案** であり、Root の実 producer は未接続である。各関数の `obj(...exact keys...)` が機械的な契約で、未確定 fields を optional/成功 default にしない。

| producer | schema / consumer |
|---|---|
| 親 training completion | `sekirei.white-view-paired-nonlinear-training-completion.v1` / `parent_completion()` |
| E3 native sidecar | `sekirei.white-view-paired-nonlinear-native-sidecar.v1` / `reconstruct_gate()` |
| core3分割 receipt | `sekirei.white-view-paired-nonlinear-candidate-core-proof.v1` / `validate_proofs()` |
| incremental receipt | `sekirei.white-view-paired-nonlinear-candidate-incremental-proof.v1` / `validate_proofs()` |
| 全入力 binding | `sekirei.white-view-paired-nonlinear-candidate-gate-binding.v1` / `reconstruct_gate()` |
| 最終 technical gate | `sekirei.white-view-paired-nonlinear-model-technical-gate.v1` / 戻り値 |

binding は `schema, refs, inputs, source_helpers, source_head` の exact keys。`refs` は code の `ROLES` 21 keys（plan/SB/parent/training-build/engine manifest+identity/recipe/O5/reference03/checkpoint/native03/export/親completion/sidecar/core/incr/public15）。各 ref は `path/bytes/sha256`、各 input は `path -> {bytes,sha256}`。未知 SOURCE、実 model/hash/probe binary に成功 default は無い。Root は外 SHA で binding 全 bytes を凍結する。

親 completion は command の6入力 fullrefs（recipe/SB/manifest/reference03/positions/labels）と output directory を実 CLI 19 flag pairsへ再構成して一致させる。実 zero exit・wait/reap・PG消滅2回・no timeout・whole wall `<86400`、3epoch終了count、poison/resume/shuffle/final False、input/source/build 不変が必要である。これらの process 観測は Root 親だけが生成する。consumer は command/log/refs/maps/epoch records を結合するが、pure code 自体が process を観測したとは主張しない。

sidecar は `native/checkpoint/親completion/recipe/SB` の一方向 refs と native FNV、selected E3、19state由来338043stepを持つ。親 outputs は checkpoint/native/export だけを持ち、sidecar を参照しない。self hash・sidecar→parent→sidecar の循環を作らない。

core/incr 各 receipt の `compilation` は既存 proof body に埋め込み、新たな中間 receipt は不要。固定新engine manifest/identity、実 USI由来 `core_link.rlib`、全 release dependency map、実 rustc/compiler map、probe/native-contract source、output binary、literal argv、実 compile outcome/log、前後不変 input map を束縛する。旧 engine manifest の線形 probe success は流用しない。新 Rust probe は frozen prototype の `PROTOTYPE_ONLY:true -> false` 一箇所だけを戻すと既知 SOURCE SHA に一致することを要求し、隣接 native contract SOURCE SHA も固定する。Root が追加 instrumentation を入れる場合は別 SOURCE version と guard 更新が必要である。

compile argv の SOURCE 提案は `rustc --edition=2024 -C target-cpu=x86-64-v3 -C opt-level=3 -C panic=abort --extern sekirei_core=<actual rlib> -L dependency=<rlib parent> <probe> -o <binary>`。Root の実 compile producer と合意・凍結するまで actual consumer を有効化しない。

## 検証内容と限界

全 rawAdam state の finite f32/非負v、protected material params/m/v の bit 同一、FTbias64、材 input→aux +0、全256 hand ties、14 head pair params/m/v と out/m signbit、v 同一、Q budget、aux FT797・prefix41 を独立検証する。protected material channels0/1 は 797 を越える原本 row を許し、aux と区別する。nearest-even FT と保存 tail を全1305356 native03 bytesに再構成して artifact 全 byte と比較する。原 serializer truncate を nearest 成功の代用にしない。

original O5 は固定manifest SHA、全 train/holdout SFEN/cache exact join・teacher/depth/count と material 値を再parseする。局面合法性、source membership、pool exclusions、旧 replay の semantic body は Root の固定 prior-origin producer/既存 validators が担当する。pure SFEN material parserは将棋の合法性証明ではない。

core は TRAIN112681+HOLDOUT5895+public15 の順序・stdin/raw JSON・stderr・全型/観測数を再parseする。incremental は public15 ordered TSV、8185 observations、refresh/undo/parent restore、全有限観測数と最大bridgeを raw JSON と照合する。native residual の大きさは観測 statistic で、100cp 上限を課さない。mate900000/normal899000 を変更しない。

`<1.001cp` は **保存済み nearest native を floatで計算した値と実coreの差**。raw training FT→nearest native による量子化誤差の上限ではない。生学習 loss、静的診断、technical gate 成功を100万ノード改善・正式採用へ読み替えない。

tiny tests は公開 initializerからのメモリ synthetic全shape、SOURCE bytes、small controlsだけ。全22 testsが成立しても、118591実row＋8185実walkを含む end-to-end gate の正常fixtureを実行したことにはならない。Root が actual schema producers を凍結し、実 reader/writer/locks/term cleanup、全 dataset/raw membership、compiled native03 受理・全raw測定を結合する残作業がある。E5 outerへの実 port、正式比較・stop/NAS/public成果物はこの package に含まれない。

Root outer は固定専用 source/binary/recipe、実 fullstate/native/親/engine/probe provenance を外 SHA で再hashし、serial locksを開始前から子のcleanup・gate書込後まで保持する。SOURCE module imports を実 canonical file bytesと外 SHA に束縛し、同名 module cache の任意objectを受理しない。出力は inputs/既存runと双方向非overlapの新private0700 root・exclusive0600 files、原本を保持する。実選択基準・Threads1/Hash128/100万nodesを変更せず、final splitを使用しない。

E5の専用 evaluation variant は D4 `white_view_production_ports.candidate_validator/validate_preflight` の固定線形common/worker path・19refs binding を新非線形 binding/consumer へ差し替える。D4 `white_view_evaluation.validate_native_gate` の `adam_used=False, epochs=0`・solver/numeric/rho100 envelope を変更せず残し、新schema専用 envelope validatorを追加する。全 raw USI/node/lifecycle再parse、同一runtimeの8identity、equal-game exact Fraction採否は変更しない。
