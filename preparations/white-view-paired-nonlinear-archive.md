# 非線形候補の準備ソースと履歴fixture

[compiled source・完全patchと9件のportable検査](white-view-paired-nonlinear-rust-v1/README.md)を、実学習で使用したソースの再現記録として保持する。9件の検査を公開CIに接続した。

math-v1、proof-v1、rust-v1/source-validationは準備段階の数式・proof・試験ソースである。私有runtimeや/tmpの元ファイル、一部の未同梱依存を使う履歴fixtureは `test_*.py.txt` として保存し、この公開リポジトリから実行できるテストとは扱わない。Python本文のバイト列は維持した。[archive index](white-view-paired-nonlinear-archive-index.json)に元の相対名、公開名、サイズ、SHA-256を記録した。

これらの履歴fixtureをportableな9件へ置き換えた、Rust cfg(test)のClone問題を解消した、または学習・正式比較を再実行したという意味ではない。元の実行環境とprivate成果物はSSD/NASに保持する。
