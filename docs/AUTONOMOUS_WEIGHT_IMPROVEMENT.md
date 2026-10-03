# 採用可能なウェイトを得るまでの自律改善（Issue #17）

## 目標と現在地

ユーザーの依頼により、採用可能な評価モデルが得られるまでgoalで改善を進める。数日という見込みを達成条件にはせず、規定のdevelopment 5局・100万ノード正式比較で現行最良よりMAEが下がり、Top3入り率が下がらないことを要求する。最初の比較基準は固定v0.3.39のmaterial fallbackで、MAEは1084.4786006022964 cp、Top3は0.5538875742592971。表示上の丸め値ではなく保存済み正式結果の有理数で採否を照合する。

Issue #15 / PR #16の実装と実験記録はマージ済みで、今回の専用worktreeは最新mainから開始した。実験コードの統合とモデル採用を区別し、今回のPRは別途許可があるまでマージしない。

## 最初の候補

- 112,681局面・constant LR=0.0001・3エポック・駒得初期化・fresh Adam・seed42。
- 前回の事前選択規則で選択済みのepoch3を、FT最近傍丸めでexportした重みをそのまま使用する。
- 重みSHA-256: `81b1c80a884349fbe8c32d74053e1b3d70a69a191611d3a3d9d2ee0febd1305f`。
- 静的holdout MAE 644.599 cpは診断値で、正式探索後のMAEや採用成功を示さない。
- 同stemのmetadataは`nnue_output=absolute`。過去の学習・診断・export・比較runは書き換えない。

候補自身のMAE pilot（17局面×両エンジン×3反復）を検証・凍結してからMAE formal（570局面×両エンジン）を行う。その同一重みのMAE formalを参照してTop3 pilot（12局面×3反復）、Top3 formal（551局面）を実施する。weight identity、明示的読込成功、全attempt、型付き結果、node evidence、USI lifecycle、cleanup、固定Teacher-E 266点とTop3対象を既存validatorで照合する。

直近の1候補全体の実測は約69〜72分。今回の所要時間は記録して見積りを更新する。既存のfallback正式runは読み取り再検証して比較基準にするが、fallbackや別候補のpilotを今回の候補のformal gateへ流用しない。

## 継続方針と資源

正式結果を得てから次の仮説を一つ選ぶ。学習量、量子化、教師データ、学習目的、表現力のどれが制約かを今回の実測と原典から調べる。単純な学習延長やdevelopmentの特定局面に合わせる調整を反復しない。候補生成・静的選択規則は正式比較前に記録する。固定holdoutは既に候補選択に使われた検証集合であり、新しい汎化の証明として扱わない。final 5局は日々の仮説選択・局面採掘に使わない。

最初の112k epoch3最近傍候補の正式比較が有効に完了し、採用条件を満たさなかった場合に限り、既存の112k・13駒価値候補ridge=1を次に評価する。最初の候補が採用された場合は評価しない。データ・特徴・prior・solver・制約・丸めを維持し、既評価のridge=0.01から正則化だけを強める。正式結果を見てlambdaを選び直さず、候補自身のpilot・正式MAE・Top3で判断する。技術的にinvalidな測定の場合はこの分岐へ進まず、証拠の問題を解消する。

次候補のSHA-256は`db1e4c4c910194709a3b7edbbb601fa1c0962de9314f7f6c7b39e8fd0ad1672e`。前回の生成・core一致記録と実体hashを再確認した。元の13係数からのL2距離は851.140→297.610 cpと小さくなる一方、固定holdout MAEは662.717→719.088 cpへ悪化する。これは静的誤差を犠牲にして駒得保持を強めたとき探索Top3が回復するかを調べる新しい仮説であり、前回のholdout選択規則の書換えではない。位置的な優劣を表せないモデルの限界と、同じ5局での反復選択による過適合を踏まえて一候補だけを試す。

探索実装・教師・100万ノード条件・二指標の採用基準を維持する。重み・特徴量・NNUE構造・学習方法・学習データは改善対象。現在のMac mini CPU・32 GiB RAM・既存SSD/NASだけを使い、重い実験は直列、解析jobs=1/Threads=1、build jobs=2とする。開始時SSD空き約27 GiB。初期の追加作業領域は8 GiB以内を目安にし、候補ごとに容量と時間見積りを確認する。既存原本を削除して容量を作らない。

今回のprivate campaignは`campaign-17-autonomous-v1`。前回の実行・比較補助を新campaignへコピーし、削除済みIssue #15 worktreeへの参照だけを今回のworktreeへ変更した。比較を担う既存6スクリプトの内容hashは前回の正式比較と同じで、固定cshogi 1.0.4 / NumPy 1.26.4のvenvで実行する。適応補助のsource・変更後hashと候補identityはprivate receiptに残す。途中停止時は既存の厳密resumeで欠落attemptだけを再開し、技術失敗・破損・identity不一致を削除や書換えで救済しない。

生局面・ラベル・weight・raw log・詳細receiptはGit外に保持する。完了成果物はSSD原本を維持し、ファイル集合・サイズ・SHA-256を照合して既存NASへ保管する。公開Git/Actionsにはコード・集計・条件・ハッシュだけを載せる。利用上限や技術的な障害は率直に報告し、未達のままgoal完了とは扱わない。

## 比較の実行経路

前回private運用に使った測定・厳密比較・公開化の補助を、それぞれ`scripts/evaluate_candidate.py`、`scripts/compare_candidates.py`、`scripts/publish_comparison.py`へ移した。repository参照をスクリプト位置から解決し、公開Markdownから今回の自律改善文書へのリンクを加えた。既存の測定・比較・公開境界を保持し、今回の最初のrunは開始時に保存したprivateコピーを使う。実行中のsourceは差し替えない。

固定audit venvのPythonで、次の順に実行できる。`CAMPAIGN`と`WEIGHT`はそれぞれ新しいprivate保存先、候補重みの絶対パスとする。測定は既定の固定runtimeを使う。

```sh
python scripts/evaluate_candidate.py --weight "$WEIGHT" \
  --output "$CAMPAIGN/evaluation" --prefix development-candidate-unique-v1
python scripts/compare_candidates.py "$CAMPAIGN/evaluation" \
  --output-json "$CAMPAIGN/comparison.json"
python scripts/publish_comparison.py --comparison "$CAMPAIGN/comparison.json" \
  --title '候補の正式比較' --candidate-label '候補モデル' \
  --output "$CAMPAIGN/public"
```

測定コマンドは新しい出力先を要求し、採用は行わない。比較コマンドのexit 0は比較の有効性で、採用には`adopt=true`が必要。公開化はこのevaluation-directory経路に対応する。既定の比較基準は今回のfallback正式runであり、最良モデルが更新された後に追加比較する場合は`--baseline-evaluation`を明示し、公開化にも`--baseline-label`を渡す。途中再開は`benchmark.py resume`または`top3.py --resume`を使い、全体補助を同じ出力先へ再実行しない。

## 実行状態

最初の候補は2026-10-03 04:12:16〜05:22:25 UTC、4,208.758秒（約70.1分）で全段階を完了した。MAE pilotは102/102・両engine17/17局面×3反復安定、正式MAEは1,140/1,140・Teacher-E 266/266、Top3 pilotは36/36・安定、正式Top3は551/551。全1,829 attemptで技術失敗0、期限内終了、supervisor/cleanup成功を確認した。最大報告1,001,086 nodesは規定上限1,010,000以内だった。

| 採用指標（5局等重み） | 現行fallback | 112k epoch3・FT最近傍 |
| --- | ---: | ---: |
| MAE | 1084.478601 cp | 949.416682 cp |
| Top3入り率 | 55.388757% | 54.067494% |

厳密比較は8項目のidentity一致、入力hash不変、developmentのみ使用を確認し、有効に完了した。MAEは約135.062 cp改善したが、Top3は約1.321ポイント低下した。採用条件は満たさず、最良モデルはfallbackを維持する。独立監査でも各局の保存値から有理数を再集計し、同じ採否を確認した。[正式比較・identity](validation/autonomous-weight-2026-10-03/expanded-e3-nearest/comparison.md)と[候補グラフ](validation/autonomous-weight-2026-10-03/expanded-e3-nearest/mae/validation.md)を公開した。これはdevelopmentでの比較であり、final評価や一般的な棋力改善を示さない。

有効な採用未達という事前分岐条件を満たしたため、次は上記のridge=1候補を評価する。今回の結果を見てlambdaを選び直すことはしない。完了候補のNAS保存と次候補の容量・identity確認を済ませてから、候補自身のpilotを開始する。goalは継続中。

完了候補のevaluation・4run・モデル/metadata・凍結したsource/receipt/publication snapshotを既存NASへ保存し、source-before/source-after/destinationの集合・サイズ・SHA-256、directory集合を照合した。3,726 files・30 directories・0 symlinks・33,657,398 bytesで一致し、SSD原本は保持した。[保存集計](validation/autonomous-weight-2026-10-03/expanded-e3-nearest/archive.json)だけを公開する。このコピーは固定runtime・audit venv・fallback比較・前回の学習archiveへの依存を持ち、単独で全環境を復元できるものではない。

関連: [Issue #17](https://github.com/phni3j9a/sekirei-weight2/issues/17)、[PR #18](https://github.com/phni3j9a/sekirei-weight2/pull/18)（下書き・未マージ）、[前回の実験](WEIGHT_IMPROVEMENT.md)、[研究方針](RESEARCH.md)、[環境](ENVIRONMENT.md)。
