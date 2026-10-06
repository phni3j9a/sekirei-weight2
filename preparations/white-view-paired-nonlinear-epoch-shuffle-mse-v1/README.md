# Epochごとの固定shuffleで学習するMSE候補

Issue [#19](https://github.com/phni3j9a/sekirei-weight2/issues/19)の第3候補用ソース。第1候補のMSE、train 112,681行、固定holdout 5,895行、White構造、fresh初期化seed42、Adam、LR、3epoch、338,043更新を維持し、trainの行順だけをepochごとの決定的なshuffleへ変える。[量子化診断と仮説選択](../../docs/WEEKLY_WEIGHT_IMPROVEMENT.md#量子化診断と次の仮説)を参照する。このsnapshotは学習前の準備で、学習・技術検証・正式比較の完了記録ではない。

shuffle seedは初期化seedと分けて事前に `20261006` へ固定した。各epoch 1〜3で元の行番号の配列からSplitMix64とrejection付きFisher–Yatesを用いる。64 bit wrapping演算、epochとのseed混合、indexのu64 little-endian表現、三つの全順列SHAは[variant manifest](public-source-manifest.json)に記録する。holdoutの順序は変えない。

予定した順列だけで学習完了を認めない。Rustは更新が正常終了した直後に元行番号と観測したoptimizer stepを記録し、最大1,024行の連続chunkを出力する。各epochの全行一回ずつの使用、順序、count、digest、stepを確認してから終端tokenを返す。Pythonのparentとproofも束縛した実child logを読み、同じ実使用順を照合する。生の学習logは非公開に保持する。

数値position updateを含む先頭34,791 bytesは元のMSEソースと同じである。FP/FTZ/DAZ、保護parameterとAdam状態、checkpoint、nearest03、native core、探索・正式100万ノード比較と共同採用条件を維持する。学習lossや合成fixtureの成功を採用へ読み替えない。source manifestの`compiled=false`等はこの準備段階の記録として保持し、実行結果は別receiptへ記録する。

原典は[Sekirei v0.3.39 / f09c130](https://github.com/kent-tokyo/sekirei/tree/f09c13026e9485a19b4ba41b91ed2e1bbdf5e1c9)。[既存compiled snapshot](../white-view-paired-nonlinear-rust-v1/compiled-source/public-source-manifest.json)に[週次runtime-v2の入力reader/binder](../white-view-paired-nonlinear-runtime-v2/README.md)とcompile固定profileを組み合わせる。専用prepare CLIはこの順に組み立て、CLIのMODEをprofile参照へ変えて3 preimagesを照合した後、[candidate.patch](candidate.patch)を適用して同梱の3 postimagesを確認する。既存compiled snapshotへ直接candidate patchを適用する手順ではない。元の[MIT](LICENSE-MIT)・[Apache 2.0](LICENSE-APACHE)・[LICENSE](LICENSE)を保持する。教師・棋譜・モデル・私有plan・実child log・固定依存環境は同梱しない。

専用CLIは`prepare_weekly_nonlinear_epoch_shuffle_mse.py`、`weekly_nonlinear_epoch_shuffle_mse_build.py`、`weekly_nonlinear_epoch_shuffle_mse_preflight.py`、`weekly_nonlinear_epoch_shuffle_mse_run.py`、`weekly_nonlinear_epoch_shuffle_mse_proof.py`。既存MSE/Huberのsource/runtimeを変更せず、新しいcheckout/build/outputを使う。新しいorder helperもbuild/preflight/run/proofの前後SHAへ束縛する。preflight/proofの履歴互換処理には既存Issue #17 worktreeへの参照があり、別配置でそのまま実行できるという保証はない。実行には同じplatform・固定OS等と、外部SHAに束縛した元数値reader・input closureが必要である。

公開fixtureは順列の全単射、固定vector、三つの全順列、合成消費traceの欠落・重複・順序・step・digest・schema・型の拒否と、公開ソースの固定数値prefixを確認する。実modelの学習結果ではない。リポジトリrootで次を実行する。

```sh
python3 -B -m unittest discover -s tests -p test_weekly_epoch_shuffle_mse.py -v
python3 -B scripts/weekly_nonlinear_epoch_shuffle_mse_run.py --help
```

学習親の補助記録やcompletion書込後のfullref確認自体が失敗する場合に備え、completionだけを成功条件にしない。実outer/innerのexit、timeout/cancellation、wait/reap・ECHILD・二回空scan、全入力/source/model前後不変とlock解放をRootが確認し、候補自身の技術検証と四段階の正式比較へ進む。現bestとfinal5局は維持する。
