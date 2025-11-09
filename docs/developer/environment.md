# 開発環境ガイド

## Linux / macOS (CPU)

### ネイティブ
- `scripts/setup-cpu.sh` を実行すると、プラットフォームごとに以下を自動化します。  
  - macOS: Homebrew を経由して `cmake`, `ninja`, `doxygen`, `graphviz`, `llvm`, `yaml-cpp` を導入。  
  - Linux (Debian/Ubuntu): `apt-get` で `build-essential`, `clang`, `llvm`, `cmake`, `ninja-build`, `doxygen`, `graphviz`, `pkg-config` などをインストールし、`yaml-cpp 0.8.0` をソースから `/usr/local` に導入。  
- セットアップ完了後に、`cmake -S . -B build -DENABLE_CPU=ON` → `cmake --build build` → `ctest` の流れを案内します。

### Docker
- CPU 用に `docker/run-cpu.sh` を用意しています。実行すると `docker/cpu/Dockerfile` から `orteaf-dev-cpu` イメージをビルドし、カレントリポジトリを `/workspace` にマウントしてシェルを起動します。
- イメージには `clang`, `cmake`, `ninja`, `doxygen`, `graphviz`, `pkg-config`, `yaml-cpp 0.8.0` などが含まれ、ホスト依存を排除できます。

## Linux (CUDA)

### ネイティブ
- `scripts/setup-cuda.sh` は Linux 前提です。内部で `scripts/setup-cpu.sh` の処理を呼び出した上で、以下を確認します。  
  - `nvidia-smi` が利用できる (ドライバ + GPU が認識されている)。  
  - `nvcc` と `CUDA_HOME` (`/usr/local/cuda` など) が有効。  
  - `llvm-objcopy` がインストール済み（CUDA カーネル埋め込みで利用）。  
- すべて揃っていれば `-DENABLE_CUDA=ON` でビルドする手順を表示します。

### Docker
- `docker/run-cuda.sh` は `docker/cuda/Dockerfile` を使って `nvidia/cuda:13.0.0-devel-ubuntu22.04` ベースの開発用イメージを構築し、`docker run --gpus=all` で起動します。  
- 実行前に NVIDIA Container Toolkit が導入済みか (`docker info` に `nvidia` runtime があるか) をチェックし、無い場合は警告を表示します。
- コンテナ内には CPU 版と同じ共通ツールに加えて CUDA Toolkit が含まれています。

## macOS (MPS)
- Metal (MPS) 用の依存関係は Docker で扱えないため、ホスト macOS 上でセットアップする。
- `scripts/setup-mps.sh` を実行すると Xcode Command Line Tools の検出、Homebrew での依存関係 (`cmake`, `ninja`, `doxygen`, `yaml-cpp`) のインストールに加え、`sudo xcodebuild -downloadComponent MetalToolchain` を使った Metal CLI の導入まで自動で行います（途中でパスワード入力が必要です）。  
  完了後は `xcrun -sdk macosx metal -help` が成功することを確認してから `cmake -S . -B build -DENABLE_MPS=ON` でビルドしてください。
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
