#pragma once

#include "orteaf/internal/backend/cpu/cpu_stats.h"

#include <cstdlib>
#include <new>
#include <cstdint>

namespace orteaf::internal::backend::cpu {

/**
 * @brief CPU メモリ割り当てのデフォルトアライメント値。
 *
 * `std::max_align_t` のアライメント要件に基づく。
 */
constexpr std::size_t kCpuDefaultAlign = alignof(std::max_align_t);

/**
 * @brief 指定された値が2のべき乗かどうかを判定する。
 *
 * @param x 判定対象の値。
 * @return `x` が2のべき乗であれば `true`、そうでなければ `false`。
 *         0 の場合は `false` を返す。
 */
inline bool is_pow2(std::size_t x) { return x && ((x & (x-1))==0); }

/**
 * @brief 指定された値以上の最小の2のべき乗を計算する。
 *
 * @param x 基準となる値。
 * @return `x` 以上の最小の2のべき乗値。
 *         0 または 1 の場合は 1 を返す。
 */
inline std::size_t next_pow2(std::size_t x){
    if (x<=1) return 1u;
    --x; x|=x>>1; x|=x>>2; x|=x>>4; x|=x>>8; x|=x>>16;
    if constexpr (sizeof(std::size_t)==8) x|=x>>32;
    return x+1;
}

/**
 * \if JA
 * @brief デフォルトアライメントでメモリを割り当てる。
 *
 * `alloc_aligned(size, kCpuDefaultAlign)` のラッパー関数。
 * 割り当て時に統計情報を自動的に更新する。
 *
 * @param size 割り当てるメモリのサイズ（バイト）。
 * @return 割り当てられたメモリへのポインタ。失敗時は `std::bad_alloc` を投げる。
 * @throws std::bad_alloc メモリ割り当てに失敗した場合。
 * \else
 * @brief Allocate memory with default CPU alignment.
 *
 * Wrapper for `alloc_aligned(size, kCpuDefaultAlign)`.
 * Statistics are automatically updated on allocation.
 *
 * @param size Size of memory to allocate in bytes.
 * @return Pointer to allocated memory; throws std::bad_alloc on failure.
 * @throws std::bad_alloc If memory allocation fails.
 * \endif
 */
inline void* alloc(std::size_t size) {
    return alloc_aligned(size, kCpuDefaultAlign);
}

/**
 * @brief 指定されたアライメントでメモリを割り当てる。
 *
 * プラットフォームに応じて以下のAPIを使用する：
 * - Windows (`_MSC_VER`): `_aligned_malloc`
 * - その他: `posix_memalign`
 *
 * アライメントが2のべき乗でない場合は、自動的に次の2のべき乗に調整される。
 * また、`std::max_align_t` より小さいアライメントは最小値に調整される。
 *
 * 割り当て時に統計情報を自動的に更新する。
 *
 * @param size 割り当てるメモリのサイズ（バイト）。
 * @param alignment 要求するアライメント（バイト）。2のべき乗である必要がある。
 * @return 割り当てられたメモリへのポインタ。`size` が 0 の場合は `nullptr`。
 *         失敗時は `std::bad_alloc` を投げる。
 * @throws std::bad_alloc メモリ割り当てに失敗した場合。
 */
inline void* alloc_aligned(std::size_t size, std::size_t alignment) {
    if (size == 0) return nullptr;

    const std::size_t min_align = alignof(std::max_align_t);
    if (alignment < min_align) alignment = min_align;
    if (!is_pow2(alignment)) alignment = next_pow2(alignment);

#if defined(_MSC_VER)
    void* p = _aligned_malloc(size, alignment);
    if (!p) throw std::bad_alloc();
#else
    // aligned_alloc は size が alignment の倍数要件あり→posix_memalign優先
    void* p = nullptr;
    const int rc = ::posix_memalign(&p, alignment, size);
    if (rc != 0 || !p) throw std::bad_alloc();
#endif

    update_alloc(size);
    return p;
}

/**
 * @brief 割り当てられたメモリを解放する。
 *
 * プラットフォームに応じて以下のAPIを使用する：
 * - Windows (`_MSC_VER`): `_aligned_free`
 * - その他: `free`
 *
 * `ptr` が `nullptr` の場合は何も行わない。
 * 解放時に統計情報を自動的に更新する。
 *
 * @param ptr 解放するメモリへのポインタ。`nullptr` の場合は何も行わない。
 * @param size 解放するメモリのサイズ（バイト）。統計情報の更新に使用される。
 */
inline void dealloc(void* ptr, std::size_t size) noexcept {
    if (!ptr) return;
    update_dealloc(size);
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    ::free(ptr);
#endif
}

/**
 * \if JA
 * @brief テスト用関数：メモリ使用量を取得する。
 *
 * この関数はテスト目的で追加されたものです。
 * 現在のメモリ使用量をバイト単位で返します。
 *
 * @return 現在のメモリ使用量（バイト）。
 * @note この関数はテスト用のため、実際の実装は含まれていません。
 * \else
 * @brief Test function: Get memory usage.
 *
 * This function is added for testing purposes.
 * Returns the current memory usage in bytes.
 *
 * @return Current memory usage in bytes.
 * @note This function is for testing purposes and does not contain actual implementation.
 * \endif
 */
inline std::size_t get_memory_usage() {
    // TODO: 実装が必要
    return 0;
}

} // namespace orteaf::internal::backend::cpu