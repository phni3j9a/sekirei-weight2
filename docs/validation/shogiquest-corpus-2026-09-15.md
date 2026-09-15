# 将棋クエスト独立棋譜corpusの固定（2026-09-15）

## 結果

将棋クエストの公開棋譜から、教師用 `.pack` と独立した人間対局候補をGit外に1,000局固定した。全局について、平手初期局面、公開一覧の既知Bot印なし、公式棋譜ページの `opp:human` 属性、指し手列の重複なし、cshogiによる全手合法再生を確認した。

教師・Sekireiで解析する前に、固定した母集団だけを入力としてdevelopment 5局 / final 5局を決定論的に選び、名前とレーティングを置換したCSAを [`benchmarks/shogiquest-v1`](../../benchmarks/shogiquest-v1) に保存した。

| 項目 | 実測 |
| --- | --- |
| 取得時刻 | 2026-09-15 08:57:42〜10:18:56 JST（81分14秒） |
| 採用 | 1,000局 / 94,982手 / 867利用者 |
| 時間設定 | 10分 679局、5分 321局。2分は対象外 |
| 手数 | 最小9、中央値97、平均94.982、最大207 |
| 対局日時範囲 | 2026-08-14 07:53:04〜2026-09-15 09:41:44 JST |
| 終局表示 | 投了577、時間切れ199、詰み198、切断23、千日手2、トライ1 |
| 除外 | 434件（既知Bot印252、公式人間属性なし35、空CSA147） |
| 取得要求 | 2,417回、全要求を直列化し2秒以上の間隔 |
| ローカル容量 | 28 MiB。採用CSA本体1,080,802 bytes |
| corpus指し手集合SHA-256 | `0616c867ba6ccc2a6c1d9a469bec444f7e992b6e5bc19229efa9066b19a7ec72` |
| ローカルmanifest SHA-256 | `371047953423f7b72877b9e10caf94f8be5fbddcb356b3aa1778f235a88a2670` |

ローカルmanifestの除外名 `csa_invalid_or_illegal` 147件を個別確認すると、すべてダウンロード応答が0 byteだった。内容が存在するCSAを合法性エラーで除外した例はない。取得中に判明したため、現行コードは今後の0 byte応答を `csa_empty` として区別する。

## 取得元とアクセス方針

- [将棋クエスト利用規約](https://d26termck8rp2x.cloudfront.net/static/questterms/term_ja.html)を2026-09-15に確認した。対局結果・棋譜がアプリ/Webで公開されること、およびサービスから取得した他ユーザーによる棋譜利用を制限しない旨の記載に基づく。規約は変更され得るため確認日を固定した。
- 候補探索とCSA取得には、第三者が利用者向けに公開する[将棋クエスト履歴検索](https://www.c-loft.com/shogi/quest/)のWeb画面が使うHTTP経路だけを利用した。同画面は2025-10-01以降、各利用者の最新30件だけを表示すると告知している。
- 人間対局属性は[将棋クエスト棋譜WEB](https://kifu.questgames.net/shogi/)の公開ページで局ごとに確認した。認証情報、アプリの非公開通信、WebSocket、内部プロトコルの解析は使っていない。
- `sekirei-weight2/0.1` と公開repository URLをUser-Agentで明示し、履歴、公式ページ、CSAを一つのprocessで直列取得した。成功応答はcacheし、再実行時には再取得しない。
- 取得日に `https://www.c-loft.com/robots.txt` はHTTP 404、`https://kifu.questgames.net/robots.txt` はHTTP 502で、利用可能なrobots規則は取得できなかった。robotsがないことを許諾とは解釈せず、上記規約・公開画面・低頻度・有限件数を判断根拠にした。

## 保存と再開

生データは `~/.local/share/sekirei-weight2/shogiquest-human-v1` に保存した。ディレクトリとcacheはmode 700、`state.json` / `manifest.json` はmode 600で、Git対象外である。

- `state.json`: 利用者queue、取得済み履歴、候補の採否、利用者名を含むローカル再開状態。
- `cache/history`: 検索応答。利用者IDのSHA-256をファイル名に使う。
- `cache/official`: 公式ページ応答。
- `cache/csa`: 対局ID単位のダウンロード応答。空応答も再要求防止のため保持する。
- `games`: 採用した生CSAを内容SHA-256名で保存。
- `manifest.json`: 件数、除外理由、取得条件、要求数、corpus hashだけをまとめたローカル集計。

取得開始時の設定SHA-256は `fcd3f0bed22eddd29a1960de8e89ab10cbfe7794fcc17b8c682c4cc66a2f120c`。途中で設定を変えた状態への追記を拒否する。初回取得のcache hitは0で、目標到達時点に未処理queueが1,016利用者分残ったが、1,000局で停止した。

## 固定benchmark

選定条件は取得・エンジン解析前に [`config/quest-corpus.json`](../../config/quest-corpus.json) へ固定した。

- 60〜160手、両者の対局時レーティングが1,200以上、差500以内。
- 公開seedと正規化指し手hashによる昇順。エンジン評価値は選定入力に含めない。
- developmentは10分3局・5分2局、finalは10分2局・5分3局。
- 10局の対局者20人に重複なし。
- CSAの対局者名とレーティングを `black` / `white` に置換。対局IDは出典追跡のためmanifestに残す。

| split | 局数 | 10分 / 5分 | 手数合計 | 終局表示 |
| --- | ---: | ---: | ---: | --- |
| development | 5 | 3 / 2 | 570 | 投了2、時間切れ2、詰み1 |
| final | 5 | 2 / 3 | 513 | 投了2、時間切れ1、詰み2 |

snapshot manifest SHA-256は `d7ee0cce928be4a45b43733dc8d34d568583b0451586078e19df98892ad9e5fc`。final 5局は改善仮説、モデル、閾値の選択に使わず節目だけで確認する。公開repositoryに収録するため秘密のtest setではなく、運用上のholdoutである。

名前をCSAから除いただけで、完全な匿名化ではない。manifestに残した公開対局IDから元ページをたどれる。再現性・出典追跡と利用者情報最小化の折衷として明記する。

## 検証

```sh
~/.local/share/sekirei-weight2/suisho11beta-v1/venv/bin/python \
  scripts/acquire_quest.py verify
~/.local/share/sekirei-weight2/suisho11beta-v1/venv/bin/python \
  scripts/acquire_quest.py verify-snapshot
```

- `verify`: 採用1,000ファイルを読み直し、保存size/SHA-256、公式属性記録、正規化hash、重複、cshogi 1.0.4による全手合法再生を確認。`PASS: 1000 unique, standard, human-marked legal games`。
- `verify-snapshot`: tracked CSA 10局のsizeに依存しないSHA-256、正規化hash、手数、名前置換、split/time bucket、重複、全手合法再生を確認。
- 固定shogiesa 0.9.2の `extract` でもdevelopment 5局から570局面、final 5局から513局面を抽出し、0局skipだった。
- ローカル状態にある人間利用者ID・表示名をtoken境界でtracked候補全体と照合し、一致0件を確認した。取得開始点の公式Bot名2件は設定に意図して残す。
- Python構文、JSON、18件のネットワーク不要な単体テスト、`git diff --check` が成功した。GitHub Actionsは外部サイトやローカル生データへアクセスしない。

## 制約

- `opp:human` はサービスが公開ページに付けた属性であり、自然人による対局や対局中の外部ソフト不使用を独立に証明するものではない。利用規約は禁止しているが、今回の取得から不正利用を判定することはできない。
- 最新30局から利用者間を幅優先にたどった標本で、全利用者・全期間からの無作為標本ではない。期間、活動頻度、入口利用者による偏りがある。
- 1,000局の候補corpusには切断23局とトライ1局を含む。固定benchmark 10局にはどちらも含まれない。
- 第三者検索と公式ページの仕様・提供継続は保証されない。今回ローカル固定した理由の一つであり、既存corpusやsnapshotの再検証はWebアクセスなしで行える。
- このIssueでは棋譜取得と分割だけを行った。水匠11β / Sekireiの100万ノード解析、グラフ、採点仕様は次のIssueで実装する。
