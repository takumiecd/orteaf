# Float16 型メモ

`orteaf/include/orteaf/internal/dtype/float16.h` に定義されている `orteaf::internal::Float16` は、CPU / CUDA 双方で同じインターフェイスを提供する 16bit 浮動小数フォーマットです。以下に主なユースケースをまとめます。FP8 変換については `docs/notes/float8.md` も参照してください。

---

## 1. CPU 側 (`.cpp`) での利用

- `float` / `double` からそのままコンストラクタで生成できます。
- `ToFloat32()` / `ToFloat64()` によって IEEE754 half を 32/64bit 浮動小数に復元できます。
- 内部的には 16bit の生ビット（`std::uint16_t`）を保持するだけなので trivially copyable です。

```cpp
#include <orteaf/internal/dtype/float16.h>

using ::orteaf::internal::Float16;

Float16 MakeHalf(float value) {
    Float16 h(value);             // float -> Float16
    float f32 = h.ToFloat32();    // Float16 -> float
    double f64 = h.ToFloat64();   // Float16 -> double

    // 演算する場合はいったん float に戻してから計算する
    Float16 sum = Float16(f32 + f32);
    return sum;
}
```

> **Note**: 加算などの算術演算はライブラリ側では提供していません。必要に応じて `float` に変換して計算するか、ラッパーを追加してください。

---

## 2. CUDA カーネル内 (`.cu`) での利用

- すべてのメソッドが `__host__ __device__` 指定されているため、ホストとデバイスで同じコードが使用できます。
- CUDA ビルド時は `__half` との相互変換 (`Float16(__half)`, `ToCudaHalf()`) が利用できます。

```cpp
// sample_kernel.cu
#include <orteaf/internal/dtype/float16.h>

using ::orteaf::internal::Float16;

__global__ void ToAndFromHalf(const float* input, Float16* output, float* restored) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    Float16 h(input[idx]);          // float -> Float16
    output[idx] = h;

    // 計算したい場合は float32 に戻す
    float value = h.ToFloat32();
    restored[idx] = value * 2.0f;

#if defined(__CUDACC__)
    // ネイティブ half が欲しい場合は __half へ変換
    __half native_half = h.ToCudaHalf();
    Float16 round_trip(native_half);
#endif
}
```

ホスト側でも同じ `Float16` 型が使えるため、テンプレートを利用した CPU/GPU 共通処理に組み込みやすくなっています。

---

## 3. まとめ / 注意点

- サイズ・アラインメントは常に 16bit (`sizeof(Float16) == 2`, `alignof(Float16) == alignof(std::uint16_t)`) を保証します。
- C++20 でコンパイル可能になるよう `constexpr` 計算に対応し、`std::bitset::operator[]` 等で `constexpr` を維持しています。
- CUDA の `__half`、ホストコンパイラが `_Float16` を提供する場合はそれぞれのネイティブ型にアクセス可能です。
- 算術演算は含まれないため、必要なら `float` に変換して実装するか、アプリケーション側でユーティリティを追加してください。

このインターフェイスにより、ARM / CUDA / CPU それぞれの環境で half precision データを同じ型で受け渡しつつ、プラットフォームに応じた最適化や後付けの演算実装を行いやすい構造になっています。
