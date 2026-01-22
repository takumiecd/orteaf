#pragma once

#include <cstddef>
#include <functional>
#include <type_traits>

namespace orteaf::internal::kernel {

/**
 * @brief Lightweight view over a contiguous array of POD elements.
 *
 * ArrayView is a non-owning, trivially copyable structure that holds a pointer
 * to data and a count. It is designed to be passed by value to CUDA/Metal
 * kernels.
 *
 * Unlike std::span, ArrayView does not involve runtime checks and guarantees
 * trivial copyability, making it suitable for device code.
 *
 * @tparam T Element type (must be const-qualified)
 */
template <typename T> struct ArrayView {
  static_assert(std::is_const_v<T>,
                "ArrayView element type must be const-qualified");

  const T *data;
  std::size_t count;

  /**
   * @brief Default constructor creates an empty view.
   */
  constexpr ArrayView() noexcept : data(nullptr), count(0) {}

  /**
   * @brief Construct from pointer and count.
   */
  constexpr ArrayView(const T *ptr, std::size_t n) noexcept
      : data(ptr), count(n) {}

  /**
   * @brief Get the number of elements.
   */
  constexpr std::size_t size() const noexcept { return count; }

  /**
   * @brief Check if the view is empty.
   */
  constexpr bool empty() const noexcept { return count == 0; }

  /**
   * @brief Access element at index (no bounds checking).
   */
  constexpr const T &operator[](std::size_t idx) const noexcept {
    return data[idx];
  }

  /**
   * @brief Equality comparison (compares pointers and counts).
   */
  friend constexpr bool operator==(const ArrayView &lhs,
                                   const ArrayView &rhs) noexcept {
    return lhs.data == rhs.data && lhs.count == rhs.count;
  }

  /**
   * @brief Inequality comparison.
   */
  friend constexpr bool operator!=(const ArrayView &lhs,
                                   const ArrayView &rhs) noexcept {
    return !(lhs == rhs);
  }
};

// Verify that ArrayView is trivially copyable for all common types
static_assert(std::is_trivially_copyable_v<ArrayView<const int>>,
              "ArrayView<const int> must be trivially copyable");
static_assert(std::is_trivially_copyable_v<ArrayView<const float>>,
              "ArrayView<const float> must be trivially copyable");
static_assert(std::is_trivially_copyable_v<ArrayView<const double>>,
              "ArrayView<const double> must be trivially copyable");
static_assert(std::is_trivially_copyable_v<ArrayView<const std::size_t>>,
              "ArrayView<const std::size_t> must be trivially copyable");

} // namespace orteaf::internal::kernel

// Hash support for std::unordered_map and std::unordered_set
namespace std {

template <typename T> struct hash<::orteaf::internal::kernel::ArrayView<T>> {
  std::size_t operator()(
      const ::orteaf::internal::kernel::ArrayView<T> &view) const noexcept {
    // Combine hash of pointer and count (boost::hash_combine style).
    std::size_t h1 = std::hash<const T *>{}(view.data);
    std::size_t h2 = std::hash<std::size_t>{}(view.count);
    constexpr std::size_t kHashMix =
        static_cast<std::size_t>(0x9e3779b97f4a7c15ULL);
    std::size_t seed = h1;
    seed ^= h2 + kHashMix + (seed << 6) + (seed >> 2);
    return seed;
  }
};

} // namespace std
