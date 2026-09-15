# 外部ソフト・資料の出典

| 対象 | 原典 | このリポジトリでの扱い |
| --- | --- | --- |
| Sekirei / 学習器 | https://github.com/kent-tokyo/sekirei | 固定 upstream を外部 runtime に取得。独自コード変更なし |
| shogiesa | https://github.com/kent-tokyo/shogiesa | 固定 upstream を外部 runtime に取得。独自コード変更なし |
| やねうら王 | https://github.com/yaneurao/YaneuraOu | GPL-3.0 ソースを外部 runtime でビルド。バイナリを Git に入れない |
| 水匠11β / V9.20 | https://www.fanbox.cc/@yaneurao/posts/11335845 | ユーザー提供アーカイブを主教師としてローカル利用。重み・記事・配布バイナリを Git に入れない |
| β・100万ノード教師（`1000000a/`） | https://www.fanbox.cc/@yaneurao/posts/12184084 | 記事上の由来とアーカイブSHAを記録。生 `.pack` はそのまま再配布しない |
| β・100万ノード教師（`1000000b/`） | https://www.fanbox.cc/@yaneurao/posts/12338760 | 記事上の由来とアーカイブSHAを記録。生 `.pack` はそのまま再配布しない |
| GenSfen / pack形式 | https://github.com/yaneurao/YaneuraOu-ScriptCollection/tree/main/GenSfen | 公開仕様を参照してストリーム監査器を実装。上流コード自体は同梱しない |
| cshogi | https://github.com/TadaoYamaoka/cshogi | GPL-3.0パッケージをローカルvenvで局面復号に使用。GitやCIへ同梱しない |
| Suisho11Plus | https://www.fanbox.cc/@yaneurao/posts/12349386 | 旧固定環境を参考教師として保持。重み・記事・同梱コードを Git に入れない |
| 将棋クエスト利用規約 | https://d26termck8rp2x.cloudfront.net/static/questterms/term_ja.html | 棋譜の公開と取得した棋譜の利用に関する現行記載を取得前に確認 |
| 将棋クエスト棋譜WEB | https://kifu.questgames.net/shogi/ | 各候補の公開ページにある人間対局属性を確認。ページ本文や利用者名はGitへ保存しない |
| 将棋クエスト履歴検索 | https://www.c-loft.com/shogi/quest/ | 第三者が公開する検索・CSAダウンロード画面を低頻度で利用。生の履歴・CSA・cacheはローカルだけに保存 |
| 開発ループの参考 | https://github.com/phni3j9a/rfkit-rs | 目標・検証・一つの課題・レビューという運用方針を参考にした |

各ソフトの原著作権・ライセンスは取得したソース内に保持する。バージョン・ハッシュは config/toolchain.lock.json。今回の Python スクリプトと短い CSA fixture はこのプロジェクト用に新規作成した。

データ・重み・派生モデルの公開条件は個別の利用条件に従う。配布教師データは自身の評価関数学習・実験・研究への利用が許諾されている一方、そのままの二次配布は禁止と明記されているため、ローカルruntimeだけで扱う。今回の環境構築はモデルや変換済みデータの公開可否を決めるものではない。
