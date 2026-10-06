# Raw masterとnearest03の512局面診断ソース

Issue [#19](https://github.com/phni3j9a/sekirei-weight2/issues/19)の第1候補MSE（`white-view-diverse-games-seed42-e3-v1`）について、保存されたraw master floatとその候補自身のnearest03を分けて検証する診断コードである。第2候補Huberや現採用モデルの量子化誤差を測る診断ではない。[失敗履歴と実行状態](../../docs/validation/weekly-weight-improvement-2026-10-05/raw-master-quantization-diagnostic/summary.json)を別に記録する。この公開snapshotの作成はsource確認と小fixtureだけである。別のv3実parentはRootの終端確認を通過してexit0となり、512局面の実測値を上記の別結果receiptへ記録した。診断記録のNAS保存はpendingである。

原典は[Sekirei v0.3.39 / f09c130](https://github.com/kent-tokyo/sekirei/tree/f09c13026e9485a19b4ba41b91ed2e1bbdf5e1c9)。Rustの依存は[既存のpaired nonlinear公開snapshot](../white-view-paired-nonlinear-rust-v1/README.md)、その[compiled-source](../white-view-paired-nonlinear-rust-v1/compiled-source/public-source-manifest.json)と、同じ固定White feature/runtimeである。元の[MSE全state checkpoint/nearest03/技術検証と正式結果](../../docs/validation/weekly-weight-improvement-2026-10-05/diverse-games-e3/comparison.md)を参照する。原典のライセンスは[LICENSE](LICENSE)、[MIT](LICENSE-MIT)、[Apache 2.0](LICENSE-APACHE)を変更せず同梱し、原典noticeを保持する。

公開物は相対pathのpatch、4 Rust postimagesと必要なweekly identity module、sampler/aggregator、startup parser用の合成回帰fixture、原典license、[source manifest](public-source-manifest.json)である。実completion/master/native/context、教師・棋譜・512行のrequest/生log、私有parentやcommand plan、source/input mapsは付属しない。公開sourceだけでデータ付き数値再実行を完結できる環境ではない。コード内の初期局面とunit-test盤面は合成fixtureであり、今回の512局面データを公開したものではない。

## 固定samplingと三つの差

train 112,681行と固定holdout 5,895行から、それぞれ256行を `floor(i*(N-1)/255), i=0..255` の昇順indexで選ぶ。両端を含む512行で、label値・model predictionを選択条件にしない。元のsplit membership・phase・ply・合法canonical boardを検査し、元の行を別splitへ付け替えない。development/finalを選択せず、教師再探索もしない。checkpoint contextのlabel fullrefはidentityとして維持するが、sampling/診断の計算はlabel値を使わない。

| 差（手番視点cp） | 対象 |
| --- | --- |
| raw master float − native dequant float | 保存raw masterと量子化後のfloat forwardとの差 |
| native dequant float − fixed core integer | 同じnativeのfloat/core bridge |
| raw master float − fixed core integer | 上記を含む全体の差 |

raw forwardは元の専用train-position経路から4つの数値blockを取り出し、加算・積・us/them interleave順序を維持する。clamp前に非有限中間値を拒否し、FT/L2 ClippedReLUは0〜127、outputは64で割る。汎用forwardとの同等性や再結合/fused演算を仮定しない。native dequant/coreの差は既存bridgeの1.001 cp未満を要求するが、raw masterの差を同じ上限へ読み替えず、その差自体には成功用の上限を置かない。各splitのsigned mean、absolute mean、maximum absoluteを別々に集計する。

## 状態と入力のcontrol

対象pairはraw master SHA `7c2f9793bc1e3660f0202631cce2c3ae1c85b6cf7293e9285e16e60a60772f88` とnearest03 SHA `44d1b412da22688542a406599dd40b40bc3243285aaaeae67b21d003db9a0429`。完了epoch3・step338043の元context、recipe、source binding、source files、manifest/profile、input fullrefsを束縛する。全parameter・m/v・3 scalars・global stepのbitsをbefore/after比較し、pure nearest再exportと既存nativeのfull bytes、fixed core readbackのfull bytesを照合する。optimizer updateは0、checkpoint/export/modelを上書きしない。元のcompiled initialized readerのpositive/missing-FTZ negativeは別の必須段階であり、新しい診断APIの成功で代替しない。

`nnue_white_view_aux_tied` のWhite構造（L1=256、L2=32）、同じsource/model/runtime/settings、`RUSTFLAGS=-C target-cpu=x86-64-v3`、`SEKIREI_TRAIN_FTZ_DAZ=1` と実calling-thread MXCSR control `0x9fc0`を前提とする。各rowとbefore/afterでFP readbackを要求する。Linux x86_64の同じCPU/platform、固定compiler/deps・OS・loader/shared libraries等が必要で、別platformや全OS復元へ保証を広げない。診断は直接forwardであり、100万node探索の正式MAE/Top3、モデル採用、全局面の量子化上限・汎化を証明しない。

## Source snapshotの組み立て

`source/`の4診断postimagesはprivate source-v5の原本bytesを保持し、v2/v3でRust計算本体は同じである。既存公開compiled snapshotのmainにweekly module宣言がなかったため、[baseline-weekly-profile.patch](baseline-weekly-profile.patch)でその2行と803 Bの固定identity moduleを記録した。このmoduleはSHA/count定数だけで、privatepathや局面・ラベルを含まない。既存公開compiled snapshotからこのbaseline patchを適用すると、manifestの診断preimage pinsへつながる。続いて[candidate.patch](candidate.patch)を適用すると4 postimagesへつながる。原典の全ソース・依存は別途必要である。

公開Rust snapshotは`PROTOTYPE_ONLY=true`を保持し、入口をI/O前に閉じる。実機での専用source clone・compiled fixtures・全入力/source/compiler/buildの前後pinsを確認した後、[queued-activation.patch](queued-activation.patch)の1つのbool変更で専用診断入口を開く。これは記録用snapshotを成功へretagする操作ではなく、別runtimeの実行境界である。既存source/runtimeを上書きせず、original reader、各child exit/wait/reap、timeout/cancellation、二回空scan、coordination locksの取得/全解放、fresh outputsの所有、容量・時間予算と外部SHAを確認する親が別途必要で、私有実行parentは公開物に含めない。

`sample_request.py prepare`は外部SHAに束縛したown complete planからfresh requestを作り、`aggregate`は外部SHA付きの完全logからstrict summaryを作る。私有planの実値を公開exampleへ埋めない。helpとmemory-only回帰検査は、このディレクトリで実行できる。

```sh
python3 -B sample_request.py --help
python3 -B -m unittest discover -v -p 'test_*.py'
```

samplerの22 fixtureは固定v3のうち21 methodを同じASTで保持し、source検査1 methodだけを公開relative snapshotへ接続した。3つのpublic snapshot検査はmembership・全fileのsize/SHA・4 postimages・weekly profile・activationの1 bool変更を確認する。実機P3のstartup追加回帰6 methodはsamplerの22 fixture内に同じASTで含まれ、合成512 recordsだけを使う。startupの既存行 `Training floats: FTZ/DAZ enabled (MXCSR bits 15/6)` を先頭に一度だけ要求する。欠落・重複・表記違い・途中挿入・未知追加行を拒否し、512/FP/terminal、finite/type/all-state controlsを維持する。fixtureの成功を実modelの数値結果へ読み替えない。v1はRust module path E0583で数値処理前に失敗し、v2は5段階がexit0（raw child512行）でもこのstartup行を旧aggregatorが拒否して全体exit1となった。両失敗を保存し、v3の実終端・summaryは別結果receiptで確認した。実parentの成功は512 sampleの診断完了であり、モデル採用・全局面の上限やNAS保存完了を意味しない。
