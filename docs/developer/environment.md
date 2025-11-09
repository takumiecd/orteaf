# 開発環境ガイド

## Linux / 共通 (Docker)
- Docker を利用して共通のツールチェーンを提供。
- イメージビルド:
  ```bash
  docker build -f docker/dev/Dockerfile -t orteaf-dev .
  ```
- 作業フォルダをマウントしてシェルを起動:
  ```bash
  docker run --rm -it -v "$(pwd)":/workspace -w /workspace orteaf-dev
  ```
- イメージには `clang`, `cmake`, `ninja`, `doxygen`, `graphviz` などが含まれる。
- `yaml-cpp` を使うコード生成ツールのために、コンテナでは `libyaml-cpp-dev` も事前にインストールされています。
- CUDA を利用する場合は別途 `nvidia/cuda` ベースの派生イメージを作成する予定（未実装）。

## macOS (MPS)
- Metal (MPS) 用の依存関係は Docker で扱えないため、ホスト macOS 上でセットアップする。
- `scripts/setup-mps.sh` を実行すると Xcode Command Line Tools の検出と Homebrew での依存関係 (`cmake`, `ninja`, `doxygen`, `yaml-cpp`) のインストールを行います。手動で必要なパッケージを追加する場合は `brew install <package>` を利用してください。
- スクリプトが完了したら、`cmake -S . -B build -DORTEAF_FETCH_GTEST=ON` などでビルド環境を構築できます。

## Windows
- 推奨する方法は WSL2 + Docker。WSL 上で Linux 版コンテナを利用し、同じツールチェーンを共有する。
- ネイティブ Windows 手順は今後必要に応じて追加する。

## Doxygen
- Docker イメージに Doxygen を含めているため、コンテナ内で以下を実行すればドキュメント生成が可能:
  ```bash
  doxygen docs/Doxyfile.user         # 公開 API ドキュメント
  doxygen docs/Doxyfile.developer    # 開発者向けドキュメント
  doxygen docs/Doxyfile.tests        # テストスイート専用ドキュメント
  ```
- 出力先はそれぞれ: `docs/api-user/`、`docs/api-developer/`、`docs/api-tests/`

> このガイドは暫定版です。環境構築手順が固まり次第アップデートしてください。
