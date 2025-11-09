# Float8 型メモ

`orteaf/include/orteaf/internal/dtype/float8.h` では、FP8 フォーマットとして広く利用される E4M3 / E5M2 の両形式をサポートしています。CPU と CUDA のどちらでも同じ API で扱えるように設計されており、用途に応じて精度・ダイナミックレンジを選択できます。

---

## 1. E4M3 (`Float8E4M3`)

- 符号 1bit / 指数 4bit / 仮数 3bit。指数バイアスは 7。
- 無限大は表現せず、オーバーフロー時は最大有限値にクリップ。
- 精度が高めで、重み・活性化の格納用途に適したフォーマット。

```cpp
#include <orteaf/internal/dtype/float8.h>

using ::orteaf::internal::Float8E4M3;

Float8E4M3 PackE4M3(float value) {
    Float8E4M3 fp8(value);       // float -> FP8
    float restored = fp8.ToFloat32();
    // 8bit表現として出力する場合は Bits() を利用
    std::uint8_t raw = fp8.Bits();
    (void)restored;
    return Float8E4M3::FromBits(raw);
}
```

---

## 2. E5M2 (`Float8E5M2`)

- 符号 1bit / 指数 5bit / 仮数 2bit。指数バイアスは 15。
- 無限大・NaN を表現可能。広いダイナミックレンジを必要とする勾配などに向く。

```cpp
#include <orteaf/internal/dtype/float8.h>

using ::orteaf::internal::Float8E5M2;

Float8E5M2 PackE5M2(double value) {
    Float8E5M2 fp8(value);       // double からも直接変換可能（内部では float32 化）
    float restored = fp8.ToFloat32();
    return fp8;
}
```

---

## 3. CUDA カーネルでの利用

`__host__ __device__` 指定済みのため、ホスト / デバイスの両方で同じコードが利用できます。E4M3 / E5M2 どちらの型も `ToFloat32()` 経由で IEEE754 32bit に復元し、必要があれば演算後に再量子化するフローが基本です。

```cpp
// sample_fp8_kernel.cu
#include <orteaf/internal/dtype/float8.h>

using ::orteaf::internal::Float8E4M3;
using ::orteaf::internal::Float8E5M2;

__global__ void Fp8RoundTrip(const float* input, Float8E4M3* out_e4m3, Float8E5M2* out_e5m2) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    Float8E4M3 e4(input[idx]);
    Float8E5M2 e5(input[idx]);

    out_e4m3[idx] = e4;
    out_e5m2[idx] = e5;

    // 実際の計算は float32 に戻して行う
    float value = e4.ToFloat32() + e5.ToFloat32();
    out_e4m3[idx] = Float8E4M3(value * 0.5f);
}
```

---

## 4. 注意点 / 仕様

- 2 種類のフォーマットはいずれも `sizeof(...) == 1` / `alignof(...) == 1` を保証し、POD として利用できます。
- 算術演算は提供していないため、必要に応じて `float` / `double` に戻して計算してください。
- E4M3 はオーバーフロー時に最大有限値へクリップ、E5M2 は無限大を保持します。
- `configs/dtype/dtypes.yml` では `F8E4M3` / `F8E5M2` として登録されており、暗黙キャストは `F16` 以上へ許可しています。

---

## 5. 今後の拡張アイデア

- 量子化 / デ量子化をユーティリティ化して、スケール適用を組み込みたい場合は `Float8` ラッパーの薄いヘルパーを追加してください。
- CUDA の `__nv_fp8_*` 型と直接やり取りするラッパーが必要になった場合は、`float8.h` 内で条件付きincludeを追加することで対応できます。
