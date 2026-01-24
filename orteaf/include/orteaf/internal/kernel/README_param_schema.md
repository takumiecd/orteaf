# Kernel Parameter Schema

型安全なカーネルパラメータ抽出機構。カーネル引数（`CpuKernelArgs`, `MpsKernelArgs`, `CudaKernelArgs`など）から特定のパラメータセットを構造体として一括取得できます。

## 基本的な使い方

### 1. スキーマ構造体を定義

```cpp
#include <orteaf/internal/kernel/kernel_param_schema.h>

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
// orteaf/include/orteaf/internal/kernel/common_schemas.h
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
