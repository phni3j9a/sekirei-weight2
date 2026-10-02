# 初回CPU学習と2指標評価

Issue #13では、固定Sekirei v0.3.39の探索を変更せず、配布された水匠11β・100万ノードの教師packから最初のweightを作る。初回候補の生成と有効な比較・採否記録が到達点で、数値改善や棋力を事前に保証しない。

## 評価指標

採否に用いる指標は次の二つ。Top1一致率や逆方向のTop3率は追加しない。

1. MAE: 従来のMultiPV=1、両者 `go nodes 1000000`。固定教師exact-cp点に候補のexact-cpが全て揃う場合のみ定義し、棋譜ごとのMAEを5局で等重み平均する。
2. Top3入り率: 同じ局面の水匠MultiPV=1のbestmoveが、Sekireiの別測定 `MultiPV=3, go nodes 1000000` で返る3候補に含まれる割合。各棋譜の割合を5局で等重み平均する。

Top3は正式MAE runの教師結果を再検証して参照する。対象はdevelopmentのnormal分類かつ合法手4手以上の局面。終端・合法手3手以下・既存のhash-bound mate-in-one shortcut局面は事前に除外し、候補の出力によって分母を変えない。3つの異なる合法root move、同一の最終完了depth、MultiPV 1/2/3、第一候補とbestmoveの一致が必要。候補不足・混在depth・bound・技術失敗・欠測は率を未定義にし、不一致として丸めたり分母から消したりしない。同点でも3候補を拡張しない。

MAEのpilot/formal gateを維持する。Top3にも別pilot（既存canonical/regression pilotと対象集合の共通部分、3反復）を置き、完全性と候補列の再現性を確認してから全対象へ進む。生USI、node evidence、lifecycle、cleanup、重み読込の検証は既存supervisorを再利用する。normal局面の合法手集合は固定cshogiで検証する。

実行失敗数・coverage・bound/mateは測定の健全性を確認する情報であり、追加のモデル性能指標にはしない。final splitの棋譜はこの評価経路で使用しない。

初回の採用方針は、両指標が有効に測定でき、同じv0.3.39 baselineよりMAEが低く、Top3入り率が下がらない場合に限って候補を採用する。片方だけ改善した場合は保留し、両指標を記録する。絶対的なMAE目標値やTop3目標率はまだ設定しない。5局の開発結果を一般的な棋力の証明とはしない。

## 教師データと学習

`scripts/pack_dataset.py` は既存の `suisho11beta-v1` runtimeに保存済みの13本をmanifest/内容hash照合して参照する。新v0.3.39 runtimeへ重複コピーする必要はない。専用の旧βvenv（cshogi 1.0.4 / NumPy 1.26.4）で復号する。

初回の抽出は各hash順packの先頭200局を上限に、16手目以降・4局面おき・最大32局面/局とする。学習規模を抑えるprefix標本で、全packの一様無作為抽出ではない。ゲームの開始局面・全指し手列からidentityを作り、重複ゲームを除く。`SHA256(seed:game-id) mod 10` の0を保留、それ以外をtrainにする。train/保留に共通する盤面は両側から除き、同じ側の盤面重複は最初の一つだけ残す。盤面照合は手番・持駒を含み手数カウンタを除く。

評価データの漏洩防止として、取得済み独立棋譜1,000局の母集団全体から盤面集合を機械的に作り、その全局面を学習・保留標本から除外する。各CSAの内容hashと1,000局のcanonical aggregateを取得時manifestへ照合する。`benchmarks/.../final`を開かず、finalの所属や成績を用いた選別をしない。配布packに記録されていないゲーム系列・変化枝の由来までは証明できない。

packの評価値は手番視点でそのまま渡す。`abs(cp)>=30000` はmate-scaleとして学習対象外、残るラベルのclipは±30000、WDL混合なし、absolute出力とする。packにexact/bound情報がないことは残る制約である。holdoutは別ファイルに隔離し、学習器の局面単位random splitを使わない。初回は事前固定の3 epochで、保留lossからepochやハイパーパラメータを選ばない。

固定upstreamの学習器は外部教師の名前空間を指定できないため、`patches/sekirei-train-external-labels.patch`を専用checkoutに適用する。これは学習器の外部入力とCPU数値処理設定だけを変更し、探索エンジンには適用しない。外部teacher identityを明記したstrict/cache-onlyモードを要求し、ラベルが一件でも欠ける場合は停止する。`label_depth=0`は外部キャッシュ用の識別値で、元の探索深さを示さない。学習器・patch・入力・recipe・weightのhash、全ラベルcache hit、処理局面数を保存する。

## CPUでの数値処理

最初の55,404局面の実行は18分経過してもepochを完了せず、失敗記録を残して停止した。256局面の短い試走からの線形外挿が不適切だった。固定学習器は勾配0の要素も含めてAdamのmomentを減衰するため、長い学習でsubnormalが蓄積し得る。専用学習器では `SEKIREI_TRAIN_FTZ_DAZ=1` を明示し、x86のMXCSR FTZ/DAZを有効にする。これはsubnormalを0として扱う学習recipeの変更であり、IEEEのgradual underflowと完全同一とは主張しない。設定をbuild/epoch/runに記録する。探索バイナリやUSI比較設定へは適用しない。一般的な作用は [IntelのFTZ/DAZ資料](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2025-0/set-the-ftz-and-daz-flags.html) を参照。

## 実行例と資源上限

内蔵SSDに実行用のbuild/data/runを置く。開始時空き約30 GiB、初回学習は最大1時間、今回の追加作業領域8 GiB以下。短い256局面の疎通確認に加え、4,096局面の試走で速度/RSS/保存容量を確認し、上限内の規模に調整する。エンジン比較はjobs=1/Threads=1、ビルドはjobs=2。NAS保管は環境文書の照合手順に従い、生データ・重み・raw logは公開しない。

```sh
python3 scripts/prepare_training.py --output "$TRAIN_RUNTIME"
"$AUDIT_PYTHON" scripts/pack_dataset.py \
  --corpus-runtime "$CORPUS_RUNTIME" --quest-runtime "$QUEST_RUNTIME" \
  --output "$DATASET" --games-per-pack 200
python3 scripts/train_cpu.py --dataset "$DATASET" --trainer "$TRAIN_RUNTIME" \
  --output "$TRAIN_RUN" --epochs 3 --seconds 3600
"$AUDIT_PYTHON" scripts/benchmark.py pilot --config "$BENCHMARK_CONFIG" --run-id "$PILOT_ID"
# pilotを検証し、同じconfigのformal.pilot_evidenceを固定してからformalを実行する。
"$AUDIT_PYTHON" scripts/benchmark.py formal --config "$BENCHMARK_CONFIG" --run-id "$FORMAL_ID"
"$AUDIT_PYTHON" scripts/top3.py pilot --mae-config "$BENCHMARK_CONFIG" \
  --mae-run "$MAE_RUN" --run-id "$TOP3_PILOT_ID"
"$AUDIT_PYTHON" scripts/top3.py formal --mae-config "$BENCHMARK_CONFIG" \
  --mae-run "$MAE_RUN" --run-id "$TOP3_FORMAL_ID" --pilot-run-id "$TOP3_PILOT_ID"
```

候補用configは固定development configのcopyに `candidate_model.kind=nnue` とweightの絶対pathを指定する。実行時にEvalFileへ同じpathを渡し、ハッシュと明示的な読込成功応答を照合する。baselineのpilotを別weightのformalへ流用しない。weightなしのfallbackは別のbaselineとして保持する。

検証結果・採否は実機測定完了後に追記する。
