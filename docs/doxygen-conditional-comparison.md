# Doxygen条件分岐の比較

## 方式1: デフォルト英語 + `\if JA`で日本語追加（推奨）

```cpp
/**
 * @brief Allocate memory with default CPU alignment.
 *
 * Wrapper for `alloc_aligned(size, kCpuDefaultAlign)`.
 * Statistics are automatically updated on allocation.
 *
 * @param size Size of memory to allocate in bytes.
 * @return Pointer to allocated memory; throws std::bad_alloc on failure.
 * @throws std::bad_alloc If memory allocation fails.
 *
 * \if JA
 * @details
 * CPUメモリのデフォルトアライメント（std::max_align_t）で確保します。
 * 統計情報は自動的に更新されます。
 * \endif
 */
inline void* alloc(std::size_t size);
```

**メリット:**
- ✅ 英語が基本で、日本語は補足として自然
- ✅ コメントが読みやすく、メンテナンスしやすい
- ✅ 英語版は日本語部分が無視されるだけなのでシンプル
- ✅ 国際的なプロジェクトでは英語が標準として扱える

**デメリット:**
- ⚠️ 英語と日本語で内容が重複する可能性がある
- ⚠️ 完全に異なる説明をしたい場合には不向き

---

## 方式2: `\if DOXYGEN_JA`/`\else`/`\endif`で完全分離

```cpp
/**
 * \if DOXYGEN_JA
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
inline void* alloc(std::size_t size);
```

**メリット:**
- ✅ 英語と日本語を完全に独立して記述できる
- ✅ 言語ごとに最適化された説明が可能
- ✅ 文化やコンテキストに応じた説明が書ける

**デメリット:**
- ⚠️ コメントが長く、可読性が下がる
- ⚠️ メンテナンスが大変（両方の言語を更新する必要がある）
- ⚠️ コードレビュー時に両方の言語を確認する必要がある

---

## 推奨: 方式1（デフォルト英語 + `\if JA`）

**理由:**
1. **可読性**: コードレビュー時に英語が基本として見える
2. **メンテナンス性**: 基本説明は英語で一度書けば良い
3. **国際標準**: 英語が標準で、日本語は補足として扱える
4. **柔軟性**: 必要に応じて日本語部分だけ追加・修正できる

**使用例:**
```cpp
/**
 * @brief Allocate memory with default CPU alignment.
 * @param size Size of memory to allocate in bytes.
 * @return Pointer to allocated memory; throws std::bad_alloc on failure.
 *
 * \if JA
 * @details
 * CPUメモリのデフォルトアライメント（std::max_align_t）で確保します。
 * 統計情報は自動的に更新されます。size==0 の場合は nullptr を返します。
 * \endif
 */
```

この方式なら、英語版では日本語部分が無視され、日本語版では両方が表示されます。
