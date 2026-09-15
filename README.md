# sekirei-weight2

水匠11β（Suisho Concerto 202512）と Sekirei をそれぞれ **100万ノード指定**で解析し、未学習棋譜の評価値グラフをできるだけ重ねるための研究プロジェクト。Suisho11Plus は主目標からは外し、節目で確認する参考教師として残す。

探索実装を当面固定し、評価モデルの重み・特徴量・NNUE 構造・学習データ・学習方法を改善する。旧 sekirei-weight の方針や実験履歴を前提にしない。

教師解析・学習は現在の Mac mini の CPU・32 GiB RAM・既存ストレージの範囲で進める。今後も追加機材は導入せず、外付け GPU・別 PC・クラウド GPU・有料計算基盤への移行を計画に含めない。公開リポジトリと GitHub Actions の軽量 CI を利用する。

## 現在の段階

Issue [#1](https://github.com/phni3j9a/sekirei-weight2/issues/1) / [PR #2](https://github.com/phni3j9a/sekirei-weight2/pull/2) で初期環境を整備し、Issue [#3](https://github.com/phni3j9a/sekirei-weight2/issues/3) / [PR #4](https://github.com/phni3j9a/sekirei-weight2/pull/4) で主教師を水匠11βへ切り替えて生成済み教師データを監査した。Issue [#5](https://github.com/phni3j9a/sekirei-weight2/issues/5) では独立評価棋譜を固定した。

- Sekirei と付属学習器、shogiesa、水匠11β用やねうら王V9.20のソースを commit 単位で固定。
- ビルド・重み照合・USI疎通はPython標準ライブラリで実行。`.pack` の局面復号と外部CSAの合法手確認だけは専用venvに固定したcshogi / NumPyを使う。
- β・100万ノードとして配布された `.pack` 15本を取り込み、同一内容の2本を除いた13本（534,175,084 bytes）をローカルで管理。ゲーム境界を保つストリーム復号と標本再解析が可能。
- 実機で両エンジンの100万ノード指定探索と shogiesa の一局面ラベル生成を確認済み。[β環境と教師監査](docs/validation/suisho11beta-2026-09-15.md)。旧Plus環境の結果は[初期検証](docs/validation/environment-2026-09-14.md)に残す。
- 10標本の固定V9.20再解析では、確定値6件のMAE 3.167 cp（最大11 cp）、境界値4件、保存指し手一致6件。互換性の小規模確認であり、元の生成環境との完全同一性の証明ではない。
- 正式ベンチマーク用に、将棋クエストの公開棋譜から人間同士・平手・合法手・重複なしの1,000局をローカルへ固定し、その中から解析前に development 5局 / final 5局を選定済み。[取得・分割の検証記録](docs/validation/shogiquest-corpus-2026-09-15.md)。
- `.pack` から学習器への入力経路、本格学習、モデル採用判定、定期自動実行は次の段階。
- Sekirei の今回の初期疎通は **駒得評価へのフォールバック**。学習済みモデルはまだない。

## 開発用 100万ノード baseline

Issue #7 の実装は、`benchmarks/shogiquest-v1/development` の5局だけを固定入力にする。初期局面は含めず、各棋譜の記録指し手の直後を一つの occurrence として扱うため、合計570局面になる。同じ指し手列が別の棋譜に現れても occurrence は共有しない。`final` を選べるパス引数は用意していない。

標準ライブラリのboard trackerは固定hashのCSAについて、手番・所有者・成り・駒取り・打ち駒を含む厳密な移動遷移／擬似合法性を検査してUSI履歴へ変換する。盤面分類は `config/development-position-classifications.json` に入力・履歴・position hashを束縛して保持する。これは pinned cshogi 1.0.4 で全570手を再生して生成・検証した監査メタデータで、強制1手5局面（development-04 p106/p108、development-05 p115/p117/p121）と終端詰み development-05 p123 を含む。CIの標準ライブラリ検査だけから完全合法性を主張しない。計画とhashを確認するコマンドは次のとおり。実行結果は外部runtimeへ保存され、CIはネットワークや教師重みを必要としない。

```sh
python3 scripts/benchmark.py plan --run-type pilot
python3 scripts/benchmark.py plan --run-type formal
```

pilotは各棋譜の `{1, ceil(L/2), L}` の15局面を両エンジン・3回ずつ（90 attempt）計画する。formalは570局面を各1回ずつ（1,140 attempt）計画する。plan時に固定audit runtimeの cshogi 1.0.4 が利用できれば、変換済み570手と分類をすべて照合して結果を記録する（`--cshogi` は互換用に受理する）。これはCIの必須依存ではない。

実機実行は `pilot`、その全90 attemptから再計算した `observed_max_reported_nodes` と、採用する絶対値 `max_reported_nodes` を `formal.pilot_evidence` に凍結した後に `formal` を使う。受理する非終端スコアの node evidence は存在する整数 `>=0` とし、0 は分類済み強制1手の Sekirei が sole legal move を PV head と bestmove の両方に出した場合だけ許容する。終端詰みでは teacher の同一要求内 `score mate -1 nodes 0 pv resign` + `bestmove resign` だけを mate として受理し、Sekirei の裸の resign は no_score のままにする。通常の説明不能な0、欠落・負値・malformed、timeout/cleanup/protocol failure は成功に救済しない。current-go の `go nodes ...` から対応する `bestmove` まで、`info string` と `pv` の payload を境界として structured `info` の全 `nodes` token を収集する。各 structured info 行は `nodes` の値組を高々一つとする厳密文法を採用し、重複は曖昧な node evidence として失敗にする。選択score行の値、最後の有効値、全有効値の最大を別々に保存し、後二者で負値・malformedを隠さず、最大値を ceiling と pilot gate の観測値に使う。`go nodes` は上限なので、正当に早期完了した pilot の observed max が requested nodes 未満でも許容するが、formal gate は両エンジンそれぞれに少なくとも一つの正の node observation を要求する。formal gate は pilot と formal の canonical `execution_identity`（runner/parser、development hash、分類hash、requested nodes、timeout、環境、全option、バイナリ・build/toolchain、教師重み、候補モデル、host）を完全一致させ、freeze用の gate/config と文書・repo commit の変更だけは identity に含めない。現在はpilot証跡と上限が未確定なので、formal実行は意図的に拒否される。各attemptはrunner所有のLinux supervisor/subreaperをsession anchorにしてengineを直接argv execする新しいUSIプロセスで、`RAYON_NUM_THREADS=1`、`go nodes 1000000`、設定済みoptionの広告確認、`usi` → `setoption` → `isready` → `usinewgame` → `position` → `go` → `bestmove` → `quit` の順で行う。supervisorはengine親の即時終了後も子孫cleanupまで残り、`/proc/<supervisor>/task/<supervisor>/children` と `waitid(WNOWAIT)` で再親化を確認してからpidfdを開き、TERM/KILLを `signal.pidfd_send_signal` 経由で送り、全子のreap/ECHILDとrelay閉鎖を確認する。pidfd・列挙・signal・reapの証明に失敗した場合、またはrunnerが監督anchorを強制killした場合はcleanupを成功扱いにせず、未知の同一PGIDもsignalしない。pilotのheadlineは正式基準ではなく、3 repetitionすべての再現性診断を保持する。

```sh
python3 scripts/benchmark.py pilot --run-id development-pilot-20260916-v2
python3 scripts/benchmark.py resume --run-id development-pilot-20260916-v2
python3 scripts/benchmark.py status
python3 scripts/benchmark_report.py report \
  --run-dir ~/.local/share/sekirei-weight2/suisho11beta-v1/runs/development-pilot-20260916-v2 \
  --output /tmp/development-pilot-report
python3 scripts/benchmark_report.py export \
  --run-dir ~/.local/share/sekirei-weight2/suisho11beta-v1/runs/development-pilot-20260916-v2 \
  --output /tmp/development-pilot-export
```

raw USI log、attempt JSON、manifest、重み・モデル・絶対パスを含むレポートはruntime外へ出さない。raw log には engine の send/receive 行をそのまま保持し、最後に分類後・cleanup後の一つの structured runner outcome（status、failure reason、phase、deadline完了、cleanup状態、engine returncode、supervisor status/failure）を追加する。attemptとreportには position type、分類別のcoverage、engine別のpositive node evidenceを保持するが、公開exportには履歴を出さない。engineの終了コードと監督cleanup failureは別フィールドで保持し、監督のterminal status欠落やforce-killはcleanup成功に昇格させない。resume/report/export は raw lifecycle（広告されたoption、送信順、position、go nodes、対応bestmove、正常終了時のquit）、recorded binary identity、runner outcome と保存結果を再照合する。`export` は空の出力ディレクトリ直下に `validation.md` / `reviewed.svg` / `validation.json` / `manifest.json` の4ファイルだけを書き、`local/` や局面別ファイルを作らない。aggregateだけを残した明示的な公開境界で、ローカルパス、ユーザー名、source game ID、position履歴、モデルパスをレビュー時に拒否する。旧 `development-pilot-20260916` は旧parser semanticsの診断証跡として immutable に扱い、formal evidenceには使わない。今回の段階では新しい実機pilot/formalの結果や公開result artifactはまだ生成していない。

## 使い始める

USI node grammar は、`go nodes <[0-9]+>`（ASCII 十進数字列、符号なし）から対応する `bestmove` までの各 structured `info` 行を左から解釈する。`info string` は全体を free text として無視し、`pv`・`string`・`refutation`・`currline` の可変長 payload に入った後の token は解釈しない。payload 前の各行には `nodes <[0-9]+>` を高々一組だけ許容し、欠落・重複・負値・符号付き／非整数値は無効な node evidence として扱う。

Linux x86_64 / AVX2・BMI2 対応 CPU、Rust/Cargo、Python 3.11+、GCC C++、make、7z が必要。準備スクリプトは sudo や OS パッケージ変更を行わない。Ubuntu 24.04 の現ホストには必要なツールがある。

専用 worktree 内から実行する。

```sh
python3 scripts/prepare.py doctor
python3 scripts/prepare.py audit-deps
python3 scripts/prepare.py build --jobs 2
python3 scripts/prepare.py import-teacher \
  --archive '/absolute/path/to/suisho11beta-suisho_concerto202512.7z'
python3 scripts/prepare.py import-corpus --archive '/absolute/path/to/kif20260630-pack-1M.7z'
python3 scripts/prepare.py import-corpus --archive '/absolute/path/to/kif20260731-pack-1M.7z'
python3 scripts/smoke.py
~/.local/share/sekirei-weight2/suisho11beta-v1/venv/bin/python \
  scripts/audit_pack.py --samples 10
~/.local/share/sekirei-weight2/suisho11beta-v1/venv/bin/python \
  scripts/acquire_quest.py crawl --dry-run
~/.local/share/sekirei-weight2/suisho11beta-v1/venv/bin/python \
  scripts/acquire_quest.py crawl
~/.local/share/sekirei-weight2/suisho11beta-v1/venv/bin/python \
  scripts/acquire_quest.py verify
~/.local/share/sekirei-weight2/suisho11beta-v1/venv/bin/python \
  scripts/acquire_quest.py snapshot
~/.local/share/sekirei-weight2/suisho11beta-v1/venv/bin/python \
  scripts/acquire_quest.py verify-snapshot
```

`--runtime /absolute/path` で保存先を変更できる。既定は `~/.local/share/sekirei-weight2/suisho11beta-v1`。旧Plus環境を上書きせず、複数 worktree で固定したβ環境を共有する。実験でソース版・構造を変える場合は別 runtime を使う。

`smoke.py` は短い自作棋譜の一局面を両エンジンで100万ノード指定解析し、さらに shogiesa から教師を起動してラベルを一件生成する。結果と USI ログは runtime の `runs/environment-*/` に保存される。これはモデル品質や棋譜全体の一致率を測るベンチマークではない。

`import-corpus` はアーカイブ全体を既知SHA-256で照合し、β用の `1000000a/` / `1000000b/` だけを抽出する。各ファイルを内容SHAで保存するため重複は一つの実体になる。アーカイブ、重み、生の `.pack`、実行結果はGitやActionsへ入れない。

`audit_pack.py` は各固有ファイルから有限評価値を1局面ずつ選び、局面・履歴・手番を復元して固定教師で再解析する。元データに探索の境界種別はないため、再解析が upperbound / lowerbound の場合は点差を計算しない。また `go nodes` は上限であり、合法手を読み切るなどして上限前に完了した探索の実ノード数もそのまま記録する。

`acquire_quest.py` は、現在公開されている第三者の棋譜検索画面から履歴とCSAを直列・既定2秒間隔で取得し、公式棋譜ページの `opp:human` 属性、一覧上のBot印、平手初期局面、cshogiによる全手再生を照合する。取得状態と応答cacheは `~/.local/share/sekirei-weight2/shogiquest-human-v1` に1局ごとに保存され、同じコマンドで再開できる。Webサービスの非公開通信を解析・利用しない。公開画面の仕様は変わり得るため、異常な応答では停止し、取得済みcacheを再利用する。

`snapshot` は1,000局が揃った後、対局者の重複、手数、対局時レーティング差を制約し、固定hash順位だけで development 5局 / final 5局を選ぶ。CSA内の対局者名とレーティングは置換し、出典を追跡するため対局IDはmanifestに残す。最終評価用5局はモデルや閾値の選択には使わない。

## 次の到達点

まず、固定した独立棋譜で現状の重ね合わせグラフを作り、時間・再現性・誤差を把握する。同時に、生成済み `.pack` をゲーム単位で train / development に分割し、全量JSONL展開を避けた学習入力経路を作る。その後、CPUで小規模学習を一周通す。

主指標は棋譜ごとの MAE を棋譜間で平均する案。教師のみ詰み／候補のみ詰みなどで採点対象が候補依存に変わらない仕様を、正式な採用判定の前に確定する。

## 文書

- [研究方針・比較条件](docs/RESEARCH.md)
- [固定環境・再現手順・制約](docs/ENVIRONMENT.md)
- [将棋クエスト独立棋譜の取得・分割](docs/validation/shogiquest-corpus-2026-09-15.md)
- [開発運用](AGENTS.md)
- [外部ソフト・資料の出典](docs/PROVENANCE.md)

コード・設定・検証手順を公開 Git に置き、配布記事・アーカイブ・重み・大規模データ・実行結果はローカルに置く。Issue/PR には変更理由と検証結果を残す。モデル公開・定期実行の有効化は今回の準備に含まれない。
