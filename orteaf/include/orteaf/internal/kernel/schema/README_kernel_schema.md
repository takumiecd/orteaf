# Kernel Schema (Parameter & Storage)

型安全なカーネルパラメータ・ストレージ抽出機構。カーネル引数（`CpuKernelArgs`, `MpsKernelArgs`, `CudaKernelArgs`など）から特定のパラメータ・ストレージセットを構造体として一括取得できます。

## 目次
- [パラメータスキーマ (`ParamSchema`)](#パラメータスキーマ-paramschema)
- [ストレージスキーマ (`StorageSchema`)](#ストレージスキーマ-storageschema)
- [組み合わせ使用](#組み合わせ使用)

---

# パラメータスキーマ (`ParamSchema`)

カーネル引数から必要なパラメータを構造体として一括取得します。

## 基本的な使い方

### 1. スキーマ構造体を定義

```cpp
#include <orteaf/internal/kernel/schema/kernel_param_schema.h>

namespace orteaf::internal::kernel::cpu {

// カーネルに必要なパラメータを構造体として定義
struct MyKernelParams : ParamSchema<MyKernelParams> {
  Field<ParamId::Alpha, float> alpha;
  Field<ParamId::Beta, float> beta;
  Field<ParamId::Dim, std::size_t> dim;
  
  // マクロで自動抽出ロジックを生成
  ORTEAF_EXTRACT_FIELDS(alpha, beta, dim)
};

} // namespace
```

### 2. カーネル実装で使用

#### CPU Kernel
```cpp
#include <orteaf/internal/kernel/cpu/cpu_kernel_args.h>

void executeMyCpuKernel(CpuKernelArgs& args) {
  // パラメータを一括抽出
  auto params = MyKernelParams::extract(args);
  
  float alpha = params.alpha;
  float beta = params.beta;
  std::size_t dim = params.dim;
  
  // カーネル実行処理...
}
```

#### MPS Kernel
```cpp
#include <orteaf/internal/kernel/mps/mps_kernel_args.h>

void executeMyMpsKernel(MpsKernelBase& base, MpsKernelArgs& args) {
  // 同じスキーマを使用
  auto params = MyKernelParams::extract(args);
  
  float alpha = params.alpha;
  float beta = params.beta;
  std::size_t dim = params.dim;
  
  // Metal Kernel実行...
}
```

#### CUDA Kernel (将来対応)
```cpp
#include <orteaf/internal/kernel/cuda/cuda_kernel_args.h>

void executeMyCudaKernel(CudaKernelArgs& args) {
  // 同じスキーマを使用
  auto params = MyKernelParams::extract(args);
  
  // CUDA Kernel実行...
}
```

## オプショナルパラメータ

```cpp
struct NormalizationParams : ParamSchema<NormalizationParams> {
  Field<ParamId::Epsilon, float> epsilon;       // 必須
  Field<ParamId::Axis, int> axis;               // 必須
  OptionalField<ParamId::Scale, double> scale{1.0};  // オプショナル（デフォルト値: 1.0）
  
  ORTEAF_EXTRACT_FIELDS(epsilon, axis, scale)
};

void executeNormalization(auto& args) {  // CPU/MPS/CUDA どれでもOK
  auto params = NormalizationParams::extract(args);
  
  float eps = params.epsilon;
  int axis = params.axis;
  
  // オプショナルパラメータの処理
  if (params.scale) {  // 存在チェック
    double scale = params.scale.value;
  }
  
  // またはデフォルト値を使用
  double scale = params.scale.valueOr(1.0);
}
```

## マクロなし版（手動実装）

```cpp
struct MyKernelParams : ParamSchema<MyKernelParams> {
  Field<ParamId::Alpha, float> alpha;
  Field<ParamId::Beta, float> beta;
  
  // extractAllFields()を手動で実装
  template <typename KernelArgs>
  void extractAllFields(const KernelArgs& args) {
    alpha.extract(args);
    beta.extract(args);
  }
};
```

## バックエンド共通スキーマの定義

全バックエンド（CPU/MPS/CUDA）で共通のスキーマを定義できます。

```cpp
// orteaf/include/orteaf/internal/kernel/schema/common_schemas.h
namespace orteaf::internal::kernel {

// 汎用的なスケーリングパラメータ
struct ScaleParams : ParamSchema<ScaleParams> {
  Field<ParamId::Alpha, float> alpha;
  Field<ParamId::Beta, float> beta;
  
  ORTEAF_EXTRACT_FIELDS(alpha, beta)
};

// 正規化パラメータ
struct NormParams : ParamSchema<NormParams> {
  Field<ParamId::Epsilon, float> epsilon;
  Field<ParamId::Axis, int> axis;
  OptionalField<ParamId::Scale, double> scale{1.0};
  
  ORTEAF_EXTRACT_FIELDS(epsilon, axis, scale)
};

} // namespace
```

各バックエンドで使用：
```cpp
// CPU実装
void cpuScale(CpuKernelArgs& args) {
  auto params = ScaleParams::extract(args);
  // ...
}

// MPS実装
void mpsScale(MpsKernelBase& base, MpsKernelArgs& args) {
  auto params = ScaleParams::extract(args);  // 同じスキーマ
  // ...
}
```

## 利点

1. **型安全**: コンパイル時に型チェック、ParamIdと型の不一致を防止
2. **可読性**: 必要なパラメータが構造体として明示的
3. **保守性**: パラメータ追加時にFieldを1行追加するだけ
4. **バックエンド非依存**: CPU/MPS/CUDAで同じスキーマを使い回せる
5. **エラーハンドリング**: 存在チェックと型検証が自動
6. **パフォーマンス**: ゼロコストアブストラクション、インライン展開

## 従来との比較

### 従来の方法（手動取得）
```cpp
void executeKernel(auto& args) {
  const auto* alpha_param = args.findParam(ParamId::Alpha);
  if (!alpha_param) throw std::runtime_error("Missing alpha");
  const auto* alpha_val = alpha_param->tryGet<float>();
  if (!alpha_val) throw std::runtime_error("Type mismatch");
  float alpha = *alpha_val;
  
  // 他のパラメータも同様に... (繰り返し)
}
```

### 新しい方法（スキーマベース）
```cpp
struct MyParams : ParamSchema<MyParams> {
  Field<ParamId::Alpha, float> alpha;
  ORTEAF_EXTRACT_FIELDS(alpha)
};

void executeKernel(auto& args) {
  auto params = MyParams::extract(args);
  float alpha = params.alpha;
}
```

## API リファレンス

### `Field<ParamId ID, typename T>`
- 必須パラメータを表すフィールド型
- `extract(args)`: 抽出（存在しない場合や型不一致で例外）
- `operator T()`: 暗黙的型変換
- `get()`: 明示的値取得

### `OptionalField<ParamId ID, typename T>`
- オプショナルパラメータを表すフィールド型
- `extract(args)`: 抽出（存在しなくてもOK）
- `present`: パラメータが存在するか
- `valueOr(T defaultVal)`: 値またはデフォルト値を取得
- `operator bool()`: 存在チェック

### `ParamSchema<Derived>`
- スキーマベースクラス（CRTP）
- `static Derived extract(const KernelArgs&)`: 一括抽出（テンプレート）

### `ORTEAF_EXTRACT_FIELDS(...)`
- フィールドリストから`extractAllFields()`を自動生成するマクロ

## 実装の詳細

### テンプレートによる汎用性
全ての型（`Field`, `OptionalField`, `ParamSchema`）がテンプレートで実装されており、`KernelArgs`型に依存しません。これにより：

- CPU/MPS/CUDA用の個別実装が不要
- 新しいバックエンド追加時に自動対応
- ヘッダーオンリーで実装が完結

### コンパイル時最適化
- `constexpr`による定数畳み込み
- インライン展開による関数呼び出しオーバーヘッドの削減
- パラメータパック展開による効率的なフィールド抽出

---

# ストレージスキーマ (`StorageSchema`)

カーネル引数から必要なストレージバインディングを構造体として一括取得します。

## 基本的な使い方

### 1. ストレージスキーマ構造体を定義

```cpp
#include <orteaf/internal/kernel/schema/kernel_storage_schema.h>

namespace orteaf::internal::kernel::mps {

// カーネルに必要なストレージを構造体として定義
struct MyKernelStorages : StorageSchema<MyKernelStorages> {
  StorageField<StorageId::Input0> input;
  StorageField<StorageId::Output> output;
  OptionalStorageField<StorageId::Workspace> workspace;
  
  // マクロで自動抽出ロジックを生成
  ORTEAF_EXTRACT_STORAGES(input, output, workspace)
};

} // namespace
```

### 2. カーネル実装で使用

#### MPS Kernel
```cpp
#include <orteaf/internal/kernel/mps/mps_kernel_args.h>

void executeMyMpsKernel(MpsKernelBase& base, MpsKernelArgs& args) {
  // ストレージを一括抽出
  auto storages = MyKernelStorages::extract(args);
  
  // ストレージリースへのアクセス
  auto& input_lease = storages.input.lease<MpsStorageBinding>();
  auto& output_lease = storages.output.lease<MpsStorageBinding>();
  
  // オプショナルストレージのチェック
  if (storages.workspace) {
    auto& workspace_lease = storages.workspace.lease<MpsStorageBinding>();
    // Workspaceを使用...
  }
  
  // Metal Kernel実行...
}
```

#### CPU Kernel
```cpp
#include <orteaf/internal/kernel/cpu/cpu_kernel_args.h>

void executeMyCpuKernel(CpuKernelArgs& args) {
  auto storages = MyKernelStorages::extract(args);
  
  auto& input_lease = storages.input.lease<CpuStorageBinding>();
  auto& output_lease = storages.output.lease<CpuStorageBinding>();
  
  // カーネル実行処理...
}
```

## オプショナルストレージ

```cpp
struct ConvolutionStorages : StorageSchema<ConvolutionStorages> {
  StorageField<StorageId::Input0> input;      // 必須
  StorageField<StorageId::Output> output;     // 必須
  OptionalStorageField<StorageId::Workspace> workspace;  // オプショナル
  OptionalStorageField<StorageId::Temp> temp;            // オプショナル
  
  ORTEAF_EXTRACT_STORAGES(input, output, workspace, temp)
};

void executeConvolution(auto& args) {  // CPU/MPS どれでもOK
  auto storages = ConvolutionStorages::extract(args);
  
  // 必須ストレージは常にアクセス可能
  auto& input = storages.input.lease<decltype(args)::StorageListType::Storage::value_type>();
  auto& output = storages.output.lease<decltype(args)::StorageListType::Storage::value_type>();
  
  // オプショナルストレージの存在チェック
  if (storages.workspace.present()) {
    auto* workspace = storages.workspace.leaseOr(nullptr);
    // Workspaceを使用...
  }
}
```

## バックエンド共通ストレージスキーマ

```cpp
// orteaf/include/orteaf/internal/kernel/schema/common_schemas.h
namespace orteaf::internal::kernel {

// 汎用的な入出力ストレージ
struct BasicIOStorages : StorageSchema<BasicIOStorages> {
  StorageField<StorageId::Input0> input;
  StorageField<StorageId::Output> output;
  
  ORTEAF_EXTRACT_STORAGES(input, output)
};

// 複数入力ストレージ
struct MultiInputStorages : StorageSchema<MultiInputStorages> {
  StorageField<StorageId::Input0> input0;
  StorageField<StorageId::Input1> input1;
  StorageField<StorageId::Output> output;
  
  ORTEAF_EXTRACT_STORAGES(input0, input1, output)
};

} // namespace
```

## 従来との比較

### 従来の方法（手動取得）
```cpp
void executeKernel(auto& args) {
  const auto* input_binding = args.findStorage(StorageId::Input0);
  if (!input_binding) throw std::runtime_error("Missing input storage");
  auto& input_lease = input_binding->lease;
  
  const auto* output_binding = args.findStorage(StorageId::Output);
  if (!output_binding) throw std::runtime_error("Missing output storage");
  auto& output_lease = output_binding->lease;
  
  // 他のストレージも同様に... (繰り返し)
}
```

### 新しい方法（スキーマベース）
```cpp
struct MyStorages : StorageSchema<MyStorages> {
  StorageField<StorageId::Input0> input;
  StorageField<StorageId::Output> output;
  ORTEAF_EXTRACT_STORAGES(input, output)
};

void executeKernel(auto& args) {
  auto storages = MyStorages::extract(args);
  auto& input_lease = storages.input.lease<decltype(args)::StorageListType::Storage::value_type>();
  auto& output_lease = storages.output.lease<decltype(args)::StorageListType::Storage::value_type>();
}
```

## Storage API リファレンス

### `StorageField<StorageId ID, StorageRole Role = Data>`
- 必須ストレージバインディングを表すフィールド型
- `Role` は同一テンソル内のストレージ役割（Data/Index など）。省略時は `Data`
- `extract(args)`: 抽出（存在しない場合は例外）
- `binding<StorageBinding>()`: バインディング取得
- `lease<StorageBinding>()`: リース取得（const/非const版）
- `operator bool()`: 存在チェック

### `OptionalStorageField<StorageId ID, StorageRole Role = Data>`
- オプショナルストレージバインディングを表すフィールド型
- `Role` は同一テンソル内のストレージ役割（Data/Index など）。省略時は `Data`
- `extract(args)`: 抽出（存在しなくてもOK）
- `present()`: ストレージが存在するか
- `bindingOr<StorageBinding>(default)`: バインディングまたはデフォルト値
- `leaseOr<StorageBinding>(nullptr)`: リースまたはnullptr
- `operator bool()`: 存在チェック

### `StorageSchema<Derived>`
- ストレージスキーマベースクラス（CRTP）
- `static Derived extract(const KernelArgs&)`: 一括抽出（テンプレート）

### `ORTEAF_EXTRACT_STORAGES(...)`
- ストレージフィールドリストから`extractAllStorages()`を自動生成するマクロ

## 実装の詳細

### テンプレートによる汎用性
- `StorageField`, `OptionalStorageField`, `StorageSchema`は全てテンプレート実装
- バックエンド非依存で、CPU/MPS/CUDA全てに対応
- `KernelArgs`型から自動的に`StorageBinding`型を推論

### 型推論の仕組み
```cpp
template <typename KernelArgs>
void extract(const KernelArgs &args) {
  // KernelArgs::StorageListType::Storage::value_type から
  // CpuStorageBinding / MpsStorageBinding を自動推論
  using StorageBinding = typename KernelArgs::StorageListType::Storage::value_type;
  extract<StorageBinding>(args.storageList());
}
```

---

# 組み合わせ使用

パラメータとストレージを同時に使用する実践例。

## 完全なカーネル実装例

```cpp
#include <orteaf/internal/kernel/schema/kernel_param_schema.h>
#include <orteaf/internal/kernel/schema/kernel_storage_schema.h>

namespace orteaf::internal::kernel::mps {

// パラメータスキーマ
struct NormalizationParams : ParamSchema<NormalizationParams> {
  Field<ParamId::Epsilon, float> epsilon;
  Field<ParamId::Axis, int> axis;
  OptionalField<ParamId::Scale, double> scale{1.0};
  
  ORTEAF_EXTRACT_FIELDS(epsilon, axis, scale)
};

// ストレージスキーマ
struct NormalizationStorages : StorageSchema<NormalizationStorages> {
  StorageField<StorageId::Input0> input;
  StorageField<StorageId::Output> output;
  OptionalStorageField<StorageId::Workspace> workspace;
  
  ORTEAF_EXTRACT_STORAGES(input, output, workspace)
};

// カーネル実装
void executeNormalization(MpsKernelBase& base, MpsKernelArgs& args) {
  // パラメータとストレージを一括抽出
  auto params = NormalizationParams::extract(args);
  auto storages = NormalizationStorages::extract(args);
  
  // パラメータへのアクセス
  float eps = params.epsilon;
  int axis = params.axis;
  double scale = params.scale.valueOr(1.0);
  
  // ストレージへのアクセス
  auto& input_lease = storages.input.lease<MpsStorageBinding>();
  auto& output_lease = storages.output.lease<MpsStorageBinding>();
  
  // Workspaceの条件付き使用
  if (storages.workspace) {
    auto& workspace_lease = storages.workspace.lease<MpsStorageBinding>();
    // Workspaceを使った高速パス...
  } else {
    // Workspaceなしの通常パス...
  }
  
  // Metal Kernel実行...
  base.dispatch(/* ... */);
}

} // namespace
```

## バックエンド共通カーネル定義

```cpp
// orteaf/include/orteaf/internal/kernel/schema/normalization_schema.h
namespace orteaf::internal::kernel {

// 全バックエンド共通のパラメータ
struct NormalizationParams : ParamSchema<NormalizationParams> {
  Field<ParamId::Epsilon, float> epsilon;
  Field<ParamId::Axis, int> axis;
  ORTEAF_EXTRACT_FIELDS(epsilon, axis)
};

// 全バックエンド共通のストレージ
struct NormalizationStorages : StorageSchema<NormalizationStorages> {
  StorageField<StorageId::Input0> input;
  StorageField<StorageId::Output> output;
  ORTEAF_EXTRACT_STORAGES(input, output)
};

} // namespace

// CPU実装
namespace orteaf::internal::kernel::cpu {
void cpuNormalization(CpuKernelArgs& args) {
  auto params = NormalizationParams::extract(args);
  auto storages = NormalizationStorages::extract(args);
  
  auto& input = storages.input.lease<CpuStorageBinding>();
  auto& output = storages.output.lease<CpuStorageBinding>();
  // CPU実装...
}
}

// MPS実装
namespace orteaf::internal::kernel::mps {
void mpsNormalization(MpsKernelBase& base, MpsKernelArgs& args) {
  auto params = NormalizationParams::extract(args);
  auto storages = NormalizationStorages::extract(args);
  
  auto& input = storages.input.lease<MpsStorageBinding>();
  auto& output = storages.output.lease<MpsStorageBinding>();
  // MPS実装...
}
}
```

## エラーハンドリング

```cpp
void safeKernelExecution(auto& args) {
  try {
    auto params = MyParams::extract(args);
    auto storages = MyStorages::extract(args);
    
    // カーネル実行...
    
  } catch (const std::runtime_error& e) {
    // パラメータまたはストレージの欠落/型不一致
    ORTEAF_LOG_ERROR("Kernel parameter/storage extraction failed: {}", e.what());
    return;
  }
}
```

## まとめ

### ParamSchema の利点
- ✅ 型安全なパラメータ抽出
- ✅ バックエンド非依存（CPU/MPS/CUDA共通）
- ✅ 可読性の向上（必要なパラメータが構造体として明示）
- ✅ ボイラープレート削減

### StorageSchema の利点  
- ✅ 型安全なストレージバインディング抽出
- ✅ バックエンド非依存（CPU/MPS/CUDA共通）
- ✅ リースへの簡潔なアクセス
- ✅ オプショナルストレージの直感的な処理

### 組み合わせの利点
- ✅ パラメータとストレージを統一的に管理
- ✅ カーネル実装の可読性・保守性向上
- ✅ エラーハンドリングの一元化
- ✅ ゼロコストアブストラクション（実行時オーバーヘッドなし）
