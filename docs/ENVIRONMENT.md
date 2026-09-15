# 実行環境

## 利用範囲

今後も現在の Mac mini と GitHub Actions の軽量 CI の範囲で進める。追加機材（外付け GPU・別 PC・ストレージ増設等）は導入せず、クラウド GPU・有料計算基盤への移行も計画に含めない。教師解析・学習は CPU で実行し、32 GiB RAM と既存ストレージに収まるよう実測に基づいて規模を調整する。詳細は [計算資源の固定条件](RESEARCH.md#計算資源の固定条件) を参照。

## 配置

| パス | 用途 |
| --- | --- |
| `/home/server/projects/sekirei-weight2` | main 同期用 checkout、既存資料の保持 |
| `/home/server/worktrees/sekirei-weight2/<作業名>` | Issueごとの開発worktree。統合後は安全確認して削除 |
| `/home/server/projects/sekirei-weight2/docs/pixiv_fanbox_yaneurao` | ユーザー提供のローカル資料。Git 対象外 |
| `~/.local/share/sekirei-weight2/suisho11beta-v1/sources` | β固定版の upstream ソース |
| `~/.local/share/sekirei-weight2/suisho11beta-v1/build` | β固定環境のビルド生成物 |
| `~/.local/share/sekirei-weight2/suisho11beta-v1/bin` | β固定環境の実行ファイルへのリンク |
| `~/.local/share/sekirei-weight2/suisho11beta-v1/models/suisho11beta-concerto-202512` | 主教師重み |
| `~/.local/share/sekirei-weight2/suisho11beta-v1/data/teachers/suisho11beta-1m` | 内容ハッシュで重複除外した教師 `.pack` とmanifest |
| `~/.local/share/sekirei-weight2/suisho11beta-v1/venv` | `.pack` 監査・CSA合法手確認用の固定Python環境 |
| `~/.local/share/sekirei-weight2/suisho11beta-v1/runs` | smoke・監査・今後の個別実験成果物 |
| `~/.local/share/sekirei-weight2/shogiquest-human-v1` | 公開棋譜1,000局、取得cache、再開状態、ローカルmanifest |

大規模資料を worktree にコピーしない。独立した研究実験では専用のソース・出力先を使い、共通 runtime を改造しない。`prepare.py` は排他ロック、`smoke.py` と `audit_pack.py` は共有ロックを取り、スクリプト同士のビルド／解析の競合を防ぐ。手動でのソース変更や直接ビルドは別途利用状況を確認する。

## 固定ソフト

完全な commit とハッシュは [toolchain.lock.json](../config/toolchain.lock.json) にある。

| ソフト | 固定版 | 役割 |
| --- | --- | --- |
| Sekirei | v0.3.36 / `aeb6ea3` | 改善対象エンジン。初期は default 特徴量・SEKIRW01 形式 |
| sekirei-train | 同上 | upstream の学習器。実ファイル名 `train` を `bin/sekirei-train` にリンク |
| shogiesa | 0.9.2 / `dd03174` | 棋譜からの局面抽出・教師ラベル・データ整形 |
| やねうら王 | V9.20 / `a81730f` | 水匠11βに対応する公開版の教師エンジン |
| 水匠11β | Suisho Concerto 202512 | SFNNwoP1536、FV_SCALE=28。第1目標 |
| Suisho11Plus | 2026-08-01 配布記事に対応する手元アーカイブ | 旧固定環境を残す参考教師 |

水匠11βは重みであり、やねうら王の実行ファイルとは別。配布記事はV9.20以降と `NNUE_SFNNwoP1536` を指定しているため、公開履歴のV9.20版更新commitをLinux向けにビルドした。配布バイナリとのビット単位の同一性は主張しない。旧Plusのruntimeは上書きせず、新しい比較系列としてβ専用profileへ分離した。

Sekirei は最新というラベルではなく、必要な USI 機能の実装と実機結果から固定した。調査時の `main` (`08107a2`, package 0.3.5) は `go nodes` の解釈と探索のノード上限がなく、smoke が120秒でタイムアウトした。公開タグ v0.3.36 はその機能を持つ。main、GitHub の latest release、最大のタグ番号は一致していないため、自動的に main/latest へ追従しない。

### ビルド条件

- Ubuntu 24.04 / Intel i5-8500B（6 core）、32 GiB RAM。
- 初回確認: rustc/cargo 1.96.0、Python 3.12.3、g++ 13.3.0。
- Rust: `--release --locked`、default features、`RUSTFLAGS='-C target-cpu=x86-64-v3'`。既存プロジェクトのグローバル設定を変更しない。
- やねうら王: `normal COMPILER=g++ PYTHON=python3 TARGET_CPU=AVX2 YANEURAOU_EDITION=YANEURAOU_ENGINE_NNUE_SFNNwoP1536`。
- コンパイラ自体はインストール／自動更新しない。実際のバージョンと各バイナリの SHA-256 を `build-manifest.json` に記録する。同じソースでもコンパイラ変更後の完全なバイナリ同一性は主張しない。
- β環境構築前の空きは約57 GiB。ビルド・venv・重み・重複除外済みpackを含むβ runtimeは実測998 MiB。大規模な中間形式への全量展開やGPU学習環境は今回導入しない。

### 主教師の重み

```sh
python3 scripts/prepare.py import-teacher --archive \
  '/home/server/projects/sekirei-weight2/docs/pixiv_fanbox_yaneurao/suisho11beta-suisho_concerto202512.7z'
```

アーカイブSHA-256 `989d292d...b69aab8a8` を照合して `eval/nn.bin` だけを抽出し、112,887,658 bytes、SHA-256 `d1b16f0a...e8e67785` を再照合する。

配布記事の指定どおり、USIで `FV_SCALE=28` と絶対パスの `EvalDir` を渡す。V9.20の固定ソースにある `sfnnwop-1536.h` は HalfKA_hm、変換後1536次元、隠れ層15/32、LayerStacks=9。アーキテクチャをファイル名から推測せず、当時のMakefileにある専用editionを使う。

### 生成済み教師 `.pack`

```sh
python3 scripts/prepare.py import-corpus --archive \
  '/home/server/projects/sekirei-weight2/docs/pixiv_fanbox_yaneurao/kif20260630-pack-1M.7z'
python3 scripts/prepare.py import-corpus --archive \
  '/home/server/projects/sekirei-weight2/docs/pixiv_fanbox_yaneurao/kif20260731-pack-1M.7z'
```

それぞれのアーカイブ全体を固定SHA-256で照合し、記事で水匠11β・100万ノードと説明された `1000000a/` と `1000000b/` の `.pack` だけを標準出力経由で安全に抽出する。個々のファイルはSHA-256名で保存し、同じ内容を複数回保持しない。実機では収録15本のうち2本が重複し、13本、534,175,084 bytesになった。由来と重複関係はローカルの `manifest.json` に残る。

`.pack` の復号と取得したCSAの合法手再生には cshogi 1.0.4 / NumPy 1.26.4 を専用venvへ固定する。これはCPU用で、GPU環境は導入しない。

```sh
python3 scripts/prepare.py audit-deps
~/.local/share/sekirei-weight2/suisho11beta-v1/venv/bin/python \
  scripts/audit_pack.py --samples 10
```

## 疎通検証

```sh
python3 scripts/smoke.py
```

- 専用の空の run ディレクトリから起動し、呼出元にある `engine_options.txt` 等の混入を避ける。
- バイナリ・重みを照合し、広告された USI オプションを確認してから設定する。
- `usi` / `isready` を待ち、局面履歴を渡して100万ノード指定で解析する。
- cp 形式、最終 PV と bestmove の一致、node 指定の実行、正常終了を確認する。最後が upperbound/lowerbound の場合はその種別を保持し、確定評価値を `null` にする。以前の exact スコアへのすり替えはしない。
- 自作の3手の CSA fixture を shogiesa で一局面に抽出し、教師ラベルを生成する。
- 両エンジンの USI ログ、shogiesa の JSONL・manifest、全体の summary.json を保存する。

### 現時点の制約

Sekirei の `isready` は重み読込失敗後でも応答するため、将来のモデル評価ではファイルハッシュと読込成功の確認が必要。今回の smoke は明示的な駒得評価であり、旧モデルやランダム重みを学習済みとして扱わない。

ノード上限到達が aspiration 探索の途中になると、やねうら王の最終 `info` に上限・下限が付くことがある。`OutputFailLHPV=false` でも最後の報告には付く場合がある。これはノード指定の疎通失敗ではないが、確定値の採点には使えない。smoke と監査では境界の向きと生の値を保存し、先後反転では上限／下限も反転する。また `go nodes` は上限であり、探索が完了すれば100万より手前で正常終了しうる。監査では正の値かつ上限超過が2%以内であることを確認し、実ノード数を保存する。

GenSfen `.pack` はゲーム境界を保持するが、評価関数SHA、エンジンcommit、全option、score boundを持たない。今回の対象ファイルにはV9.20公開前の日付のものもあり、生成に使った開発版を特定できない。GenSfenは通常、1局につき先後用のプロセスを再利用する一方、監査は局面ごとに新しいプロセスと空のhashを使う。したがって監査は符号・尺度・大きな取り違えの検出であり、同一生成環境の証明ではない。

shogiesa の固定版は `position sfen ...` で局面を渡し、同じプロセスを再利用する。棋譜履歴を保持し、局面ごとに探索状態を独立させる正式なグラフ比較と意味が異なる。学習用のラベル生成には活用するが、今の label コマンドをそのまま最終採点器にはしない。

固定版の `sekirei-train` と `shogiesa` はGenSfen `.pack` を直接は読まない。ストリーム復号から学習入力への接続、ゲーム単位の分割、正式棋譜の選定、複数棋譜の比較、CPUでの小規模学習は次の到達点。βへの切替と監査結果は [β環境と教師監査](validation/suisho11beta-2026-09-15.md)、旧Plus環境は [初期検証](validation/environment-2026-09-14.md) に記録する。

公開 GitHub Actions では Python の構文、USI スコア受理のテスト、設定 JSON を検証する。教師重みを CI へアップロードせず、実エンジンと教師の smoke は Mac mini で実行する。
