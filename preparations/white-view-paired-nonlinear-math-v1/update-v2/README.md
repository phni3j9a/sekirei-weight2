# 未選択の条件付き次案：更新経路の source skeleton

v2 は凍結済みv1を保持する別variant。独立SOURCE peerで finite入力からpure oracleの
residual/error/gradientがoverflowして返る欠陥が見つかったため、結果をfinite検査して
拒否する修正だけを追加した。実model・recipe・update規則は変更していない。

技術準備のみ。第7候補の有効な不採用・独立監査・停止が揃った後で検討する。
未事前登録・未activation・未build・未fit・未engine・未export。
LR、head 初期化幅、保存 native output の L1 安全budget は pending であり、
この準備は値や clipping/projection recipe を選んでいない。

`paired_nonlinear_update.py` は小さな in-memory slot/group の純粋参照実装。
約60万parameter、モデル、棋譜、dataset、receipt を生成・読取する adapter はない。
`runtime_guard` は I/O や引数解釈より前に常時拒否する。`PROTOTYPE_ONLY=True`。

## 固定の仮説と native layout

- white canonical / SEKIRW03 / flat_white_view_aux_tied_v1。既存 feature ABI と
  stock search/comparison options は変えない。元 O・教師 identity・seed42 を再利用。
- protected material channels 0/1、material L2 units 0..3 と全接続、material output
  [8192,8192,-8192,-8192]、output bias0、全 FT bias64 は param/m/v 更新をskipする。
  aux head への material input rows 0/1/256/257 も固定 +0。
- aux FT channels 2..255 は学習可能。board rows は独立、hand banks 0/3 と1/2は
  master の正符号コピー。各pairにおける全254 inputについて
  `L2(us,+)=L2(them,-)=u`, `L2(them,+)=L2(us,-)=v`。
  shared bias `b(+)=b(-)`、output `out(+)=q`, `out(-)=-q`。
- pair native indices は +:4+2j / -:5+2j, j=0..13。
  aux L2 row-major offset は `(view*256+channel)*32+output`。
  手駒は board2268 の後、38 thresholds ×4 banks。white board index の180度回転と
  hand tie は既存 white feature helper に委ね、このskeletonで独自featureを増やさない。
- native scale64より `r=sum q_j*(h_j(A,B)-h_j(B,A))/64`。
  `r=(g-g_swap)/2` の g 側係数は2q。実数算術で反対称だが native f32 bitexact反対称を
  主張しない。保護M bytesと構造はそのまま維持する。

独立masterの数は FT `(2268+2*38)*254=595376`、head weights `14*508=7112`、
head bias14、output14、合計602516。これはindex算術のみでfullshapeを生成していない。

## 既存 train_position へ接続する順序

1. 元OのTRAIN順序・SFEN keyed teacher join・原manifest/replay/legal/source/exclusion
   provenance を既存経路で束縛。holdout/development/finalはfitに使わない。
   fresh Adam、3epoch、元CP-MSEのみを事前登録する（実設定は未作成）。
2. 元trainer同様、**更新前** parameterから forward/backward を一度だけ計算する。
   `e=score-T`, `d_output=2e/64`（weight=1）。clamp derivativeは0<pre<127のみ1。
   FT側ではactive us/them featuresへgradientを加算する。保護更新skipはstop-gradientではない。
3. physical gradientから各masterへ固定順で合算する。正tieはsum、output負tieはdifference。
   `du_i=G(us_i,+)+G(them_i,-)`、`dv_i=G(them_i,+)+G(us_i,-)`、
   `db=G(b+)+G(b-)`、`dq=G(out+)-G(out-)`。
   hand FTはbank0+3 / 1+2のsum。同じfeatureが両viewに現れる場合のsource梯度加算を
   先に行い、その後tieを集約する。平均にせずSUM。
4. global step tは元train_positionと同じく各位置につき一度だけ進める。各独立masterを
   そのtで一度だけ pinned `adam_update_scalar` へ渡す。vは**集約gradientの平方**。
   slaveを第二回Adamへ渡さない。protected param/m/vはscalar呼出し自体をskipする。
5. **すべてのtrainable master**を毎step更新する。inactive FTにはgradient0を渡し、
   m←β1m/v←β2vを減衰させ、既存momentによるparam更新も保持する。
   active sparse rowsだけをAdamへ渡す変更は元更新則と異なるので不可。
   同じglobal tを使ったfull inactive decayの効率化は、将来行う場合にもendpoint証明が必要。
6. aux FT masterを毎step後、float値 `[-797/64,+797/64]` へprojectする。
   m/vはprojectionで縮小・resetしない。project後paramとmomentsを正tie slaveへbyteコピー。
   head/outputは現skeletonではfinite検査のみ。選ばれていない output budgetの投影方法を
   勝手に追加しない。実学習・保存前に固定safety budget/enforcementを別途必須実装する。
7. 正tieはparam/m/v全byteをcopy。負tieはparamとmのsignbitをXORし、vを同byteコピー。
   master +0 はdependent側 -0になる明示ルール。この冗長fullshape Adam表現は新schemaで
   定義・検証する必要があり、旧checkpoint validatorを緩めて通さない。
8. checkpoint保存とexportは別の未実装adapter。Adam master/slave/protected全形状、fresh
   **logical master** m/v positivezero、protected m/v positivezero、globalstep、実新trainer/
   source/build、before/after inputhashを検証する。負符号dependentのmだけはsignbit mirrorに
   よる -0、vは +0 という冗長表現を検証する（独立Adamを起動したものではない）。
   old01へretag不可。SEKIRW03・全hand tie・保護M bytes・native float head再構成、
   nearest-even FTを全byte再構成してからcore/incremental/undoの技術proofへ進む。

## prefix と export 安全性

aux FT saved_i16はnearest-evenで |q|≤797 を再検査。bias_q64は固定。
通常standard stock≤40 feature/視点だが undo_capture は
`-current,+original,+captured,-hand` の順なので一時41featureまで存在する。
全prefixは `[64-41*797,64+41*797]=[-32613,32741]` でi16範囲内部。
param box内のpower-of-two scaling→nearest-evenだけを小fixtureで検査した。
個々のi16保存clampだけではnative saturating undoの可逆性を保証できない。

これはFT prefix条件であり、head preactivation finite、output安全域、native整数/mate域とは別。
前回 source consult のQ≤219893は数値安全の**十分条件例**で、今回budgetへ採用していない。
head preactivationのnorm boundとnative丸め込みのoutput boundを、実保存契約で必須にする。
optimizer param/gradient/m/v/lossは全finiteを要求し、NaNをclamp成功扱いしない。

## 初期化と小fixture

初期output q=0なら全局面r=0。左右master u≠vかつ活性差が非ゼロなら、
`dq=2(M-T)*(h-h_swap)/64` が最初から非ゼロになり得る。
初回head/FT梯度はq0のため0、output更新後に非飽和経路へ伝わる。
q0かつu=vはoutput勾配も0となるdead branch。finiteな異なる初期headだけでは
全TRAINの勾配を保証できず、initial activation/first-step probeを未来の技術gateで確認する。
初期化の乱数順・幅・head bias値は未選択。実O/teacher/モデルから決めない。

20 tiny tests は一回のmaster更新、sum-gradient平方、正/負mirror、inactive moment decay、
protected skip、797 projection/nearest/41prefix、index mapping、初回output→後続input勾配、
dead branch、finite difference oracle、runtime barrier、有限入力からのresidual/error/gradient
overflow拒否を検査する。モデルやSFEN scanなし。
Python Adamは元f32演算順を表すreferenceだが、Pythonのpowi/sqrtとRustのbitexact一致は未証明。
実Rust adapterは原 `adam_update_scalar` を変更せず再利用し、tiny Rust fixtureで更新と保存を
照合する必要がある。real oracleのpair odd性も実native bit証明ではない。

## 原典

固定commit f09c13026e9485a19b4ba41b91ed2e1bbdf5e1c9:

- https://github.com/kent-tokyo/sekirei/blob/f09c13026e9485a19b4ba41b91ed2e1bbdf5e1c9/crates/sekirei-train/src/trainer.rs
  SHA c86b294c35383ef25c65f209735eb1484b58dc15e43777c3f7134b5985724f53。
  2396–2410 output/L2 derivative、2493–2535 L2/FT backward、2786–2830 dense Adam、
  2888–2903 out Adam、4077–4119 fullslice/scalar m/v/bias-correction。
- https://github.com/kent-tokyo/sekirei/blob/f09c13026e9485a19b4ba41b91ed2e1bbdf5e1c9/crates/sekirei-core/src/nnue.rs
  SHA a467e8b1b75f6b82629f369a1c804476b1c610354a561822365f980e6e428561。
  710–744 undo capture順、764–792 native scale/clamp/forward。
- 公開 `scripts/white_view_paired_linear.py`
  SHA 79845db6f8ceec2e0c73488b705db526813da52bcc601ab6b10eb3f52f0ce6eb。
- 公開 `scripts/material_init.py`
  SHA d7943d5d9e01946f17300a38c143eccb0017af37f36a70554dd394a70f6bfbad。
- 公開 `scripts/fit_white_view_paired_linear.py`
  SHA d6da8cf70ddde167a7af1beeaa85d550957f2d1465fa99ed4e28c08391295eea。
- 公開 `scripts/export_nearest.py` の finite f32復元→scale64→nearest ties-evenを参照。
  SHA 124aeefea157a2036f08c17f761803bbfe694ccd6a9d2c6578ccfb7b4734bcda。

原典sourceと公開helpersのみを読取。実C/control/artifact/model/train/dev/final/runtime/T/N/Bは
不読。実Rust sourcepatch・build・fullshape checkpoint/serializer・data adapter・entry・新schema/
native/forward/proof/採用guardの統合は未実装。Rootが条件成立後に選択・事前登録・実装するまで
このsourceを実学習へ使用できない。
