# 外部ソフト・資料の出典

| 対象 | 原典 | このリポジトリでの扱い |
| --- | --- | --- |
| Sekirei / 学習器 | https://github.com/kent-tokyo/sekirei | 固定 upstream を外部 runtime に取得。独自コード変更なし |
| shogiesa | https://github.com/kent-tokyo/shogiesa | 固定 upstream を外部 runtime に取得。独自コード変更なし |
| やねうら王 | https://github.com/yaneurao/YaneuraOu | GPL-3.0 ソースを外部 runtime でビルド。バイナリを Git に入れない |
| Suisho11Plus | https://www.fanbox.cc/@yaneurao/posts/12349386 | ユーザー提供アーカイブをローカルで利用。重み・記事・同梱コードを Git に入れない |
| 開発ループの参考 | https://github.com/phni3j9a/rfkit-rs | 目標・検証・一つの課題・レビューという運用方針を参考にした |

各ソフトの原著作権・ライセンスは取得したソース内に保持する。バージョン・ハッシュは config/toolchain.lock.json。今回の Python スクリプトと短い CSA fixture はこのプロジェクト用に新規作成した。

データ・重み・派生モデルの公開条件は個別の利用条件に従う。今回のローカル環境構築は公開や再配布の判断を含まない。
