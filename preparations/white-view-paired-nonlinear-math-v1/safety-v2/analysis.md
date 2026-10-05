# Paired nonlinear 14 head: SOURCE safety / proof review

SOURCE 調査と小型の純粋算術確認だけを実施した。Rust compile、実 checkpoint/native 読取、学習、engine、118591 row / 8185 incremental 測定、正式比較は未実施。既存の SOURCE preparation を実学習・実ロード・モデル採用の成功として扱わない。LR、head 初期 width、head bias、output L1 budget の4値は選択していない。以下の `219893` は上界の説明用例であり、recipe/default ではない。

## 結論と実接続前の障害

凍結 adapter の解析上界は、元 NNUE の [2420,256,32] と f32 /64 forward に合う。aux FT の nearest-even 後 |q|≤797、bias64、全 hand tie、protected material params / moments 固定、14対の head / output / moments の bit mirror を維持する限り、integer prefix と forward の有限性を別々に証明できる。ただし Rust 型検査・実 serializer・checkpoint I/O・caller source/recipe binding はまだ成功を確認していない。

旧 linear gate の再利用には確定した契約差がある。公開 `white_view_proof_common_v2.py` は noAdam / epochs0、3つの numeric certificate、candidate−material≤100cp を要求し、incremental worker の argv は `100` 固定である。新 nonlinear の学習 checkpoint をこれに偽装できない。旧 gate / raw parser / 100cp guard は不変とし、新しい学習・state・native・安全上界を扱う専用 schema / verifier / probe を設ける必要がある。

`core_pair_probe.rs` の `<1.001cp` は **保存済み i16 FT を dequantize した同一 native の f32 forward → core の整数出力** の検査である。raw TrainWeights の f32 FT → nearest native の誤差検査ではない。前者は新しい03/nonlinear候補について改めて実検証できる。後者へ旧 linear の保証や成功 receipt を流用できない。

もう一つの実観測上の差は、中間値の finite 検査である。旧 `checked_float` は最終 cp の finite だけを調べる。L2 preactivation が ±Inf になっても clamp が0/127へ隠す場合がある。新 proof では全 L2 multiply / add、output multiply / add を **clamp 前** に検査する。pure adapter の解析上界を検証することと、実 compiled forward の中間値を観測することを区別する。

## 原典と SOURCE の所在

読み取りした正確な size / SHA は `sources-before.json`、再照合結果は `review.json` に記録する。f09 原典は commit `f09c13026e9485a19b4ba41b91ed2e1bbdf5e1c9` の Root 提供 copy。旧 sekirei-weight は参照していない。

| 根拠 | 箇所 | 読み取った契約 |
|---|---|---|
| original `nnue.rs` | 488以降 / 587以降 / 648–747 / 764–798 | refresh、hand 更新、move/capture/undo の saturating i16 順序、input-first f32 L2 / output、/64 と i32 cast |
| original `trainer.rs` | 437 / 517 / 563 / 4095 | 全 TrainWeights state、FT /64 復元、旧 export は truncate、元 Adam scalar |
| original `eval.rs` | 54–69 / 103–112 / 130–139 | 材駒価、global active / absolute / residual と explicit diagnostic mode |
| original `checkpoint.rs` | 128–189 | 旧 JSON Adam の shape / finite / save/load。新 mode / recipe / ties / provenance を証明する形式ではない |
| `paired_nonlinear_adapter.rs` | 23 / 68 / 132 / 189 / 245 / 299 | 4外部必須値、state / moment / tie guard、outward forward bound、元 Adam を各 master 一回、head init |
| `paired_nonlinear_checkpoint_native.rs` | 85 / 125 / 145 / 179 / 233 / 242 | Initialized/AfterPosition、nearest-even、03 memory layout、全 native reexport、全 state bit equality |
| `paired_nonlinear_positions.rs` / `paired_nonlinear_cli.rs` | 74 / 678 / 692、125–174 | 元 forward/backward prefix、global step 一回、poison/Result、専用 reader/export trait 未実装、entry before-I/O closed |
| public `material_init.py` | 27–28 / 83–124 / 299–316 | piece inventory / values、material encoding、exact material の条件 |
| public `core_pair_probe.rs` | 50–108 / 131–150 | native quantized FT の refresh prefix、同じ保存済み f32 tail、quantized/core bridge の意味 |
| public white-view patch | feature mapper / read-save guard | feature `nnue_white_view_aux_tied`、03専用magic、全256 hand tie。search/forward の改変ではない |

Root追加許可の `/tmp/sekirei-weight2-original-f09-eval-source-v1/eval.rs`（9290B / SHA `69fce9b3522a96c68a7f6dce15a342ba34c90f0addd0fa13b7478bef4d25e839`）を直接読んだ。103–112行は global weights active時に absoluteでNNUE単独、residual-materialではM+NNUE、inactive時にMへfallbackする。130–139行の explicit diagnostic modeも同じ意味である。この新モデルはMを内蔵するため absoluteを明示し、外でMを再加算しない。実USIによる読み込みactive/FNV証拠とexplicit probeによるread_weights証拠は別々に必要。

normal threshold `899000` と mate `900000` は今回の固定 caller 条件である。このliteralは今回の eval.rs 本文には無いので、eval.rsをその宣言元として引用しない。閾値の宣言元は追加SOURCE binding確認が必要であり、変更はしない。SOURCE数式は既存adapterのORDINARY_SCORE_LIMIT899000を直接参照する。

## legal prefix の i16 安全性

将棋の physical inventory は両王を含め最大40駒。hand threshold は実個数一つ当たり一 feature なので、通常の refresh にも最大40 feature / perspective が入る。原 `undo_capture_piece` の順序は `−current,+original,+captured,−hand`。最後の hand removal 前に一時的に41項になるため、40だけの証明では不足する。move は remove→add、capture は remove→add→remove→add で40を超えない。drop / promotion / undo の Board 側 caller 順序まで実 source を束縛し、一時的な add-first は41までであることを確認する必要がある。本調査は nnue 内の操作を直接確認したが Board 全 caller の証明を済ませたとは主張しない。

aux channel2..255 は saved FT |q|≤797、bias64。41項の任意 prefix に対し

```
64 − 41×797 = −32613 ≥ −32768
64 + 41×797 =  32741 ≤  32767
```

したがってその範囲では saturating add/sub は通常整数演算と同じになる。42項は正側33538で範囲外なので、41の caller / legal inventory 前提を省けない。nearest-even は f32 x64 を丸め、範囲外を clamp で成功へ戻さず reject する。

797は protected material channel0/1に適用しない。元 initializer は駒価cpの半値を i16 FTへ保存するので、現公開 material_init の最大single qはdragonの650であり、駒価1300cp自体と区別する。現在箱以内にあっても aux797をM保護の根拠にせず、原本bytesとの完全一致を要求する。元 initializer の別の根拠は、pawn/promoted-pawn group max10800cp、other nonking group max14980cp。半値 FT / bias64 の accumulator 最大はそれぞれ5464 / 7554で、core first clamp 上限8128より低い。undo capture の一時 duplicate は captured_color と hand_color が反対で、同一 perspective の material own feature には両方同時に入らない。この保護領域は原本 bytes と bit equality を要求する。

この上界は legal40 inventoryと指定の更新順序に対する SOURCE 数学であり、全任意 SFEN / malformed hand / 任意外部 i16 tensor の保証ではない。実 row parser は両王・piece totals・hand max・integer feature index を厳格検証する。

## finite f32 と mate 領域の分離

入力 ClippedReLU は [0,127]。head o について `C_o=|bias_o|+127×Σ_512|L2_io|` とする。512 multiply +512 add を含む全 partial sum を、unit roundoff 2^-24 に対して `C_o/(1−2^-14)+1e−30 < f32::MAX` で保守的に制限する。sum/multiply/divide は positive outward f64 next-up で下方丸めを避ける。最終和が有限でも cancellation 前の partial overflow を許さない。FTも bias1 +40×797/64 程度の有界値なので、有限な input accumulation を別に確認できる。

FTZ/DAZ は相対誤差だけの定理では説明し切れない subnormal の絶対誤差を含む。上記1e−30 paddingは1024×f32 min-normal程度より十分大きい保守余裕だが、実 probe の round-to-nearest MXCSR・FTZ/DAZ設定を束縛し、各中間 finite を観測する条件を省かない。grad / Adam finite は forward boundから自動的には従わない。元 backward は大きい head coefficient と outputによって overflowし得るため、各 physical/aggregated grad と m/v/param を検査し、失敗で poison、継続/resume/成功exportを拒否する。

14対では output master q_j と slave −q_j を signbit XOR で保持し、`Q=Σ_14|q_j|`。両 neuron の ReLU は0..127なので、実数で residual≤127Q/64。一方、実 f32 output の全 intermediate の absolute-term sum は **254Q** を使う。対同士の cancellation を先に仮定して127Qで partial安全性を証明しない。

protected material 4terms の raw absolute sum は、4×bias64＋total half-material12890 を128倍した `1682688`。32 products+32 addsを含める conservative score上界は

```
B(Q) = (1682688 + 254Q) / (1 − 2^-18) / 64 + 1
B(declared_budget) < 899000
Q_actual_outward ≤ declared_budget
```

これにより ordinary/mate の既存域を維持する。219893はこの不等式を満たす説明例だが、今回の採用 recipe値ではない。LR / width / bias / Q の数値選択と事前登録は Root の後続作業。現在 source の budget guard は超過を失敗にするもので、未選択の output projection / shrink を実装済みと扱わない。

## raw学習float → nearest → core の3段階

raw TrainWeights FT は1/64の格子にない。nearest-evenでは1 feature/channelの誤差≤1/128、40項で≤40/128=0.3125（raw f32 accumulation roundoffは別）。ClippedReLUは実数で1-Lipschitzだが、L2 column L1 normと output q により誤差が増幅される。3層の有限性 / mate 安全性は、float/nativeの1cp以内を意味しない。

小型の正確な dyadic 例を `test_safety_math.py` に置いた。raw A=1+40/1024、B=1、各FT x=1/1024は native q=0。1対の headを u=64,v=0,b=−64、out=±64とすると、raw residual2.5cp、nearest residual0cp。これは純粋な2入力算術例で、実局面・実 tensor・選択recipeを生成していない。guardが許す norm条件だけから普遍<1.001を結論できないことを示す。

今後の診断には `raw_checkpoint_float_cp`、`nearest_native_quantized_float_cp`、`nearest_native_core_cp` を異なる系列として保存する。raw→nearest差は max/平均/quantile等の診断として集計し、学習品質の成功flagへ変換しない。普遍の上限が必要なら、実 norms / quantization error / f32 rounding を含む別の事前固定条件が必要で、観測結果を見て閾値を選ばない。

quantized float→coreには、**同一native weight**、同一 i16 accum、同一 loop順序、同一 float policy、finite score<899000が前提となる。原core `/64 as i32` は truncationである。実同一 f32値なら差は厳密に1cp未満であり、旧probeの1.001は実比較用余裕として新候補で再測定できる。二つのnative引数名がnative/nearestであることや self-load pair成功から、raw checkpoint→nearest proofを作らない。

## 新 actual proof / gate に必要な typed properties

以下は必要条件の提案であり、未実装の成功 receipt schema を宣言していない。booleanは整数0/1と同値にしない。全refは canonical absolute regular-file path / nonnegative strict int bytes / lowercase64 SHAを外部期待SHAへ束縛し、読んだbytes自身をparseする。FNVはUSI/native identity checksum、SHAは全bytes provenanceとして区別する。

1. **実学習/state/native binding**: plan/4 f32bits recipe/source/helpers/3-file ABI patch/upstream/compiler/build/binary/profileをpin。canonical seed42 reference03、元O manifest/cache/teacher identity、TRAIN112681のみ、label depth0/absolute MSE/freshAdam/noresume/global step各局面一回/3epoch=338043を専用型で検証する。学習 sourceと Adam sourceを別hashで束縛。15Vec+3scalar+u64step全stateを再readしbit equality、protectedM p原本+m/v+0、FTbias1 fixed、hand0=3/1=2全256 p/m/v、paired L2/bias p/m/v同一、out/m signbit XOR、v同bits、finite/v≥0と4 recipe/forward boundsを確認する。InitializedとAfterPositionをepoch provenanceと混同しない。専用 save/read-write が未接続なら失敗。
2. **nearest03実受理**: state→nearest memory encodeと実native全1305356Bの完全一致、native→core実read/save→全bytes一致、compiled feature03 / B-small排他 / oldmagic拒否、全FT/header/f32tailのbit identityとFNVを確認。旧truncate exporterを呼ばない。実 nearest export receiptはraw checkpoint SHAとrecipe SHAへ一方向bindする。学習完了runは親 wait/reap/group停止後だけcompleteにし、sidecar→completed runの向きで循環を避ける。
3. **全row static core118591**: 元O TRAIN112681、holdout5895、公開fixture15の厳密ordered input集合をSHA/countで固定、final不使用。candidate03と別file reference03を明示load。referencecore / referencequantizedfloat / PythonMが一致。row indexとint/f32型/finiteをstrict parse、raw SHA/status/exit0/no timeout/全counts完了を確認。両perspective×256のrefresh prefixをi32 shadowで検査しcandidate・referenceのsaturations0、全L2/out中間finite、candidate ordinary-domain bound、quantized/core bridge<1.001を観測する。candidate−M≤100は要求しないし、新Qを100へ読み替えない。rawAdam→nearest診断が別途無い場合 `raw_checkpoint_quantization_bridge_measured=False` と明記する。
4. **incremental8185**: 15fixture/16walk/8search-walkとcaptures/promotions/drops/undos/null_undos>0の既存公開coverageを保持する。同じ新compiledcore / candidate03を実global loaderでactiveにし、make/undo/null各段階の両perspective×256accumがrefreshとbit一致、parent-restoration一致、整数eval一致を確認する。更新の**各 add/sub prefix**をi32 shadowで観測し41上限・saturations0、finite全forward中間、ordinary-domain上界を検証する。最終accum一致だけでは saturation→後の相殺を否定できない。full-read/save証拠とwalk証拠は別scope。STM-only交換のpaired residual oddnessは実数代数であり、interleaved f32 addの順序が変わるためnative整数bitexactを未証明扱いにする。物理180度+色+STM交換に対する canonical同値も、feature invarianceと実core観測を区別する。
5. **model gate / lifecycle / 正式評価**: 専用checkpoint/native/recipe解析、coreとincremental raw再parse、親terminal/tool-reaped、入力集合before==after、source/build不変、全locks/timeout/boundedPGcleanupを外expectedSHAから再検証し、unknown/failed/missing/partialを成功にしない。旧noAdam0/3numeric certificate schemaへflagを合成しない。解析的全legal安全性、有限固定row観測、正式1M比較採用の3scopeを別フィールドにする。NNUE構造/探索/board/eval/USI/1M/Threads1/Hash128と既存評価基準は保持。既存新binary fallback対candidateの4stage・exact equal-game MAE改善/Top3維持は別の採用条件であり、このtechnical gateは adoption=False。

## 今回の検証と限界

source/mathに今回特定した新しい凍結 adapter の確定blocking bugはない。実接続の未完項と旧linear schema不互換、final-finiteだけの旧probe scope、raw→nearest誤差未保証は上記の明示障害である。小型Python Fraction/dyadic fixturesだけを実行し、SOURCE入力は前後SHAを再照合する。Rustの型検査・fullshape Rust fixture・CPU capacity harness・compiled03実受理・学習・原棋譜での測定を今回成功扱いしない。

SOURCE版履歴: v1本文/receiptは変更せず保存した。v2はRootの追加許可eval.rs copyの読取とabsolute/fallbackの根拠を加えた。数学・旧guard・実処理の条件を変更していない。
