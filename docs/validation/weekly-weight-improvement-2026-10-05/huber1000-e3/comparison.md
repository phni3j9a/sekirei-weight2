# 第2候補: Huber1000・固定3epoch

固定development 5局・100万ノード比較は有効で、候補は**不採用**。現採用モデルを維持する。

| 指標 | 現採用E3 | Huber1000 E3 |
| --- | ---: | ---: |
| MAE（cp、低いほどよい） | 907.671621 | 1022.504590 |
| Top3入り率 | 59.951320% | 58.590726% |

両指標は各局の整数集計から算出した5局等重みの有理数で比較した。候補MAEは `1086513377/1062600`、Top3は `941756597/1607347560`。MAEの厳密低下とTop3非低下の共同条件を満たさない。8項目の比較identity一致、両モデル自身のMAE/Top3各pilot・formalの全raw再検証、比較入力と最新best登録の前後不変を確認した。[分子・分母と各局の集計](comparison.json)を参照。

第1候補と同じ多様化データ、fresh seed42、固定3epoch、構造、scalar Adam、LR、保護パラメータ、export、探索条件を維持し、損失だけをδ=1,000 cpの2倍Huberへ変更した。[学習完了集計](training-completion.json)と[技術検証集計](technical-proof.json)を保存する。学習は338,043更新・13,466.234221秒で完了し、core 118,591行とincremental 8,185観測を確認した。保存nativeを復号したfloatと同じnativeの整数coreとのbridgeであり、raw masterからの量子化差や全局面の整数bit共変性を証明した結果ではない。

[正式親処理の集計](formal-terminal.json)には専用raw proof consumer・評価・比較・guardの実exit、wait/reap、cleanup、入力/source pinsの前後一致を記録する。候補自身の4段階は102 / 1,140 / 36 / 551 attemptを完了した。全段階で同じHuber重みを確認し、現在のbest revision 0と同じ採用モデルを維持した。

[評価値の公開グラフ](mae/reviewed.svg)と[MAEの検証集計](mae/validation.md)は元の4-file exportをそのまま保持する。元reportの共通見出しは「baseline validation」だが、この候補の正式MAEに対応する。表示floatの最下位桁の差を採用条件にせず、上の整数分数で判断した。final5局は未使用であり、一般棋力や未評価棋譜への汎化を証明しない。

NAS保存15,451ファイル・2,486,486,375 bytes、独立verify、2回目のSSD物理復元、コピー環境の5 smoke、コピーだけからの正式比較再集計、元Huber数値reader v4の再検証を完了した。復元readerは7,416 logical filesを消費し、19 source roleとコピー由来decoderのoriginを確認した。失敗・修復・完了記録126ファイル・11,984,951 bytesも別の補足NAS packetへ保存し、base archiveとSSD原本を保持した。[保存・復元集計](archive-recovery.json)を参照。

初回復元は1ファイルの1バイトSHA不一致で失敗した。後の観測ではそのstageの内容が変化してNASと同じになったが、原因は未確定。観測した不一致bytesの保管物は、記録した差から明示的に再構成したもので、失敗stageそのものの不変コピーとは扱わない。元reader v3は数値処理前にclosure metadataのmodeを含む型比較で失敗し、v4でそのbootstrapだけを修正した。旧62 fixtureを保持した74 fixtureと、実際の復元コピーだけを読む数値consumerが通過した。各失敗は成功へ読み替えず保持する。同じplatformの固定OS・Python stdlib・loader/shared libraries・linkerを前提とし、OS全体の独立復元を主張しない。
