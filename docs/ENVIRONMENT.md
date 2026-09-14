# 実行環境

## 配置

| パス | 用途 |
| --- | --- |
| `/home/server/projects/sekirei-weight2` | main 同期用 checkout、既存資料の保持 |
| `/home/server/worktrees/sekirei-weight2/issue-1-environment` | Issue #1 の開発 worktree |
| `/home/server/projects/sekirei-weight2/docs/pixiv_fanbox_yaneurao` | ユーザー提供のローカル資料。Git 対象外 |
| `~/.local/share/sekirei-weight2/sources` | 固定版の upstream ソース |
| `~/.local/share/sekirei-weight2/build` | Rust ビルド生成物 |
| `~/.local/share/sekirei-weight2/bin` | 実行ファイルへのリンク |
| `~/.local/share/sekirei-weight2/models/suisho11plus` | 教師重み |
| `~/.local/share/sekirei-weight2/runs` | smoke・今後の個別実験成果物 |

大規模資料を worktree にコピーしない。独立した研究実験では専用のソース・出力先を使い、共通 runtime を改造しない。`prepare.py` は排他ロック、`smoke.py` は共有ロックを取り、スクリプト同士のビルド／解析の競合を防ぐ。手動でのソース変更や直接ビルドは別途利用状況を確認する。

## 固定ソフト

完全な commit とハッシュは [toolchain.lock.json](../config/toolchain.lock.json) にある。

| ソフト | 固定版 | 役割 |
| --- | --- | --- |
| Sekirei | v0.3.36 / `aeb6ea3` | 改善対象エンジン。初期は default 特徴量・SEKIRW01 形式 |
| sekirei-train | 同上 | upstream の学習器。実ファイル名 `train` を `bin/sekirei-train` にリンク |
| shogiesa | 0.9.2 / `dd03174` | 棋譜からの局面抽出・教師ラベル・データ整形 |
| やねうら王 | 9.80git / `c1b80ea` | Suisho11Plus の教師エンジン |
| Suisho11Plus | 2026-08-01 配布記事に対応する手元アーカイブ | SFNN_halfka2_1024_7_64_k3k3、FV_SCALE=40 |

Suisho11Plus は重みであり、やねうら王の実行ファイルとは別。手元の実行ファイル配布物は macOS 用だったため、公開されている Linux 対応ソースの現時点の commit を固定した。これは配布記事の V9.70 バイナリと同一ではない。教師の変更時は新しい比較系列として扱う。

Sekirei は最新というラベルではなく、必要な USI 機能の実装と実機結果から固定した。調査時の `main` (`08107a2`, package 0.3.5) は `go nodes` の解釈と探索のノード上限がなく、smoke が120秒でタイムアウトした。公開タグ v0.3.36 はその機能を持つ。main、GitHub の latest release、最大のタグ番号は一致していないため、自動的に main/latest へ追従しない。

### ビルド条件

- Ubuntu 24.04 / Intel i5-8500B（6 core）、32 GiB RAM。
- 初回確認: rustc/cargo 1.96.0、Python 3.12.3、g++ 13.3.0。
- Rust: `--release --locked`、default features、`RUSTFLAGS='-C target-cpu=x86-64-v3'`。既存プロジェクトのグローバル設定を変更しない。
- やねうら王: `normal COMPILER=g++ PYTHON=python3 TARGET_CPU=AVX2 YANEURAOU_EDITION=YANEURAOU_ENGINE_SFNN_halfka2_1024_7_64_k3k3`。
- コンパイラ自体はインストール／自動更新しない。実際のバージョンと各バイナリの SHA-256 を `build-manifest.json` に記録する。同じソースでもコンパイラ変更後の完全なバイナリ同一性は主張しない。
- 初回の空きは約15 GB、手元資料は約6.9 GB。大規模棋譜展開や GPU 学習環境は今回導入しない。

### 教師重み

```sh
python3 scripts/prepare.py import-teacher --archive \
  '/home/server/projects/sekirei-weight2/docs/pixiv_fanbox_yaneurao/Suisho11Plus-SFNN_halfka2_1024_7_64_k3k3-20260525 (1).7z'
```

アーカイブをハッシュ照合して `nn.bin` だけを抽出し、再度ハッシュ照合する。初期手動確認で同梱ファイルも同じモデルディレクトリに保存されているが、実行には不要。

同梱ファイル名は `engine_options.txt`（内容 `FV_SCALE 40`）。記事中の `eval_options.txt` とは異なるため、自動読込に依存せず USI で `FV_SCALE=40` と絶対パスの `EvalDir` を渡す。`sfnnwop-1536.h` という名前の同梱ヘッダも、内部の特徴量次元は1024、隠れ層は7/64、LayerStacks=9。ファイル名だけから構造を決めない。公開ソースの該当構造を指定してビルドし、同梱ヘッダはソースへコピーしない。

## 疎通検証

```sh
python3 scripts/smoke.py
```

- 専用の空の run ディレクトリから起動し、呼出元にある `engine_options.txt` 等の混入を避ける。
- バイナリ・重みを照合し、広告された USI オプションを確認してから設定する。
- `usi` / `isready` を待ち、局面履歴を渡して100万ノード指定で解析する。
- 有限 cp、最終 PV と bestmove の一致、node 指定の実行、正常終了を確認する。
- 自作の3手の CSA fixture を shogiesa で一局面に抽出し、教師ラベルを生成する。
- 両エンジンの USI ログ、shogiesa の JSONL・manifest、全体の summary.json を保存する。

### 現時点の制約

Sekirei の `isready` は重み読込失敗後でも応答するため、将来のモデル評価ではファイルハッシュと読込成功の確認が必要。今回の smoke は明示的な駒得評価であり、旧モデルやランダム重みを学習済みとして扱わない。

shogiesa の固定版は `position sfen ...` で局面を渡し、同じプロセスを再利用する。棋譜履歴を保持し、局面ごとに探索状態を独立させる正式なグラフ比較と意味が異なる。学習用のラベル生成には活用するが、今の label コマンドをそのまま最終採点器にはしない。

正式棋譜の選定・分割、複数棋譜の比較、CPU での小規模学習・重み読込の検証は別の到達点。クラウド GPU は前提にしない。初回 smoke の結果は [初期検証](validation/environment-2026-09-14.md) に記録する。

公開 GitHub Actions では Python の構文、USI スコア受理のテスト、設定 JSON を検証する。教師重みを CI へアップロードせず、実エンジンと教師の smoke は Mac mini で実行する。
