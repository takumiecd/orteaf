# ドキュメントガイド

ORTEAF に関するドキュメントの索引です。目的に応じて以下から参照してください。

## 利用者向け
- プロジェクト概要: `../README.md`
- API レイヤ概要: `developer/design.md#user-層`

## 拡張開発者向け
- コアデザイン: [docs/developer/design.md](developer/design.md)
- 拡張ガイド: [docs/developer/extension-guide.md](developer/extension-guide.md)

## プロジェクト運営
- 開発ロードマップ (下書き): [docs/roadmap.md](roadmap.md)
- チャレンジログ テンプレート: [docs/challenge-log.md](challenge-log.md)
- 開発環境ガイド: [docs/developer/environment.md](developer/environment.md)

## テスト
- TDD チェックリスト: [docs/developer/testing-strategy.md](developer/testing-strategy.md)

## ドキュメント生成
- ユーザー向け Doxygen:
  - 英語版（デフォルト）: `docs/Doxyfile.user` → `docs/api-user/`
  - 日本語版: `docs/Doxyfile.user.ja` → `docs/api-user/ja/`
- 開発者向け Doxygen:
  - 英語版（デフォルト）: `docs/Doxyfile.developer` → `docs/api-developer/`
  - 日本語版: `docs/Doxyfile.developer.ja` → `docs/api-developer/ja/`
- テスト専用 Doxygen:
  - 英語版（デフォルト）: `docs/Doxyfile.tests` → `docs/api-tests/`
  - 日本語版: `docs/Doxyfile.tests.ja` → `docs/api-tests/ja/`
- ソースコード内のコメントは `\if JA` / `\else` / `\endif` を使って英語と日本語を切り替えます。

## CI
- GitHub Actions ワークフロー: `.github/workflows/ci.yml`

> ドキュメントが増えた場合は、この索引を更新して導線を維持してください。
