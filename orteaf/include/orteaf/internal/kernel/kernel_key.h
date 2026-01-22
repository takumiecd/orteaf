#pragma once

#include <cstdint>
#include <functional>

#include <orteaf/internal/dtype/dtype.h>
#include <orteaf/internal/execution/execution.h>
#include <orteaf/internal/kernel/layout.h>
#include <orteaf/internal/kernel/op_id.h>
#include <orteaf/internal/kernel/variant.h>

namespace orteaf::internal::kernel {

/**
 * @brief KernelKey enum for packed kernel identification.
 *
 * A bit-packed key that combines OpId, Execution, Layout, DType, and Variant
 * into a single 64-bit value for efficient storage and hashing.
 *
 * Bit layout (64 bits total):
 *   [63:56] Variant   (8 bits)  - Optimization variant (256 variants)
 *   [55:40] DType     (16 bits) - Data type (65536 types)
 *   [39:32] Layout    (8 bits)  - Memory layout (256 patterns)
 *   [31:28] Execution (4 bits)  - Execution backend (16 backends)
 *   [27:12] OpId      (16 bits) - Operation ID (65536 operations)
 *   [11:0]  Reserved  (12 bits) - Reserved for future use
 */
enum class KernelKey : std::uint64_t {};

namespace kernel_key {

// Bit shift constants
inline constexpr int kOpIdShift = 12;
inline constexpr int kExecutionShift = 28;
inline constexpr int kLayoutShift = 32;
inline constexpr int kDTypeShift = 40;
inline constexpr int kVariantShift = 56;

// Bit mask constants
inline constexpr std::uint64_t kOpIdMask = 0xFFFF;
inline constexpr std::uint64_t kExecutionMask = 0xF;
inline constexpr std::uint64_t kLayoutMask = 0xFF;
inline constexpr std::uint64_t kDTypeMask = 0xFFFF;
inline constexpr std::uint64_t kVariantMask = 0xFF;

/**
 * @brief Create a KernelKey from individual components.
 *
 * @param op_id Operation identifier
 * @param execution Execution backend
 * @param layout Memory layout pattern
 * @param dtype Data type
 * @param variant Optimization variant
 * @return Packed KernelKey
 */
constexpr KernelKey make(OpId op_id,
                         ::orteaf::internal::execution::Execution execution,
                         Layout layout, ::orteaf::internal::DType dtype,
                         Variant variant) noexcept {
  const std::uint64_t op_bits = (static_cast<std::uint64_t>(op_id) & kOpIdMask)
                                << kOpIdShift;
  const std::uint64_t exec_bits =
      (static_cast<std::uint64_t>(execution) & kExecutionMask)
      << kExecutionShift;
  const std::uint64_t layout_bits =
      (static_cast<std::uint64_t>(layout) & kLayoutMask) << kLayoutShift;
  const std::uint64_t dtype_bits =
      (static_cast<std::uint64_t>(dtype) & kDTypeMask) << kDTypeShift;
  const std::uint64_t variant_bits =
      (static_cast<std::uint64_t>(variant) & kVariantMask) << kVariantShift;

  return static_cast<KernelKey>(op_bits | exec_bits | layout_bits | dtype_bits |
                                variant_bits);
}

/**
 * @brief Extract OpId from a KernelKey.
 */
constexpr OpId getOpId(KernelKey key) noexcept {
  const std::uint64_t value = static_cast<std::uint64_t>(key);
  return static_cast<OpId>((value >> kOpIdShift) & kOpIdMask);
}

/**
 * @brief Extract Execution from a KernelKey.
 */
constexpr ::orteaf::internal::execution::Execution
getExecution(KernelKey key) noexcept {
  const std::uint64_t value = static_cast<std::uint64_t>(key);
  return static_cast<::orteaf::internal::execution::Execution>(
      (value >> kExecutionShift) & kExecutionMask);
}

/**
 * @brief Extract Layout from a KernelKey.
 */
constexpr Layout getLayout(KernelKey key) noexcept {
  const std::uint64_t value = static_cast<std::uint64_t>(key);
  return static_cast<Layout>((value >> kLayoutShift) & kLayoutMask);
}

/**
 * @brief Extract DType from a KernelKey.
 */
constexpr ::orteaf::internal::DType getDType(KernelKey key) noexcept {
  const std::uint64_t value = static_cast<std::uint64_t>(key);
  return static_cast<::orteaf::internal::DType>((value >> kDTypeShift) &
                                                kDTypeMask);
}

/**
 * @brief Extract Variant from a KernelKey.
 */
constexpr Variant getVariant(KernelKey key) noexcept {
  const std::uint64_t value = static_cast<std::uint64_t>(key);
  return static_cast<Variant>((value >> kVariantShift) & kVariantMask);
}

} // namespace kernel_key

} // namespace orteaf::internal::kernel

// Hash support for std::unordered_map and std::unordered_set
namespace std {
template <> struct hash<::orteaf::internal::kernel::KernelKey> {
  std::size_t
  operator()(const ::orteaf::internal::kernel::KernelKey &key) const noexcept {
    return std::hash<std::uint64_t>{}(static_cast<std::uint64_t>(key));
  }
};
} // namespace std
