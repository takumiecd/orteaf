#pragma once

#include <cstdint>
#include <functional>

#include <orteaf/internal/architecture/architecture.h>
#include <orteaf/internal/dtype/dtype.h>
#include <orteaf/internal/execution/execution.h>
#include <orteaf/internal/kernel/core/layout.h>
#include <orteaf/internal/kernel/core/variant.h>
#include <orteaf/internal/ops/ops.h>

namespace orteaf::internal::kernel {

/**
 * @brief KernelKey enum for packed kernel identification.
 *
 * A bit-packed key that combines Version, Variant, Layout, Architecture, Op,
 * and DType into a single 64-bit value for efficient storage and hashing.
 *
 * Bit layout (64 bits total):
 *   [63:60] Version      (4 bits)  - Key format version (0 = initial)
 *   [59:52] Variant      (8 bits)  - Optimization variant (256 variants)
 *   [51:44] Layout       (8 bits)  - Memory layout (256 patterns)
 *   [43:28] Architecture (16 bits) - Execution architecture (includes Execution
 * info) [27:12] Op           (16 bits) - Operation (65536 operations) [11:0]
 * DType        (12 bits) - Data type (4096 types)
 *
 * Note: Execution is not stored directly; use getExecution() which derives it
 * from Architecture.
 */
enum class KernelKey : std::uint64_t {};

namespace kernel_key {

// Current key format version
inline constexpr std::uint8_t kCurrentVersion = 0;

// Bit shift constants
inline constexpr int kDTypeShift = 0;
inline constexpr int kOpShift = 12;
inline constexpr int kArchitectureShift = 28;
inline constexpr int kLayoutShift = 44;
inline constexpr int kVariantShift = 52;
inline constexpr int kVersionShift = 60;

// Bit mask constants
inline constexpr std::uint64_t kDTypeMask = 0xFFF;         // 12 bits
inline constexpr std::uint64_t kOpMask = 0xFFFF;           // 16 bits
inline constexpr std::uint64_t kArchitectureMask = 0xFFFF; // 16 bits
inline constexpr std::uint64_t kLayoutMask = 0xFF;         // 8 bits
inline constexpr std::uint64_t kVariantMask = 0xFF;        // 8 bits
inline constexpr std::uint64_t kVersionMask = 0xF;         // 4 bits

/**
 * @brief Create a KernelKey from individual components.
 *
 * @param op Operation identifier
 * @param architecture Execution architecture (includes Execution info)
 * @param layout Memory layout pattern
 * @param dtype Data type
 * @param variant Optimization variant
 * @param version Key format version (default: kCurrentVersion)
 *
 * Values are masked to the bit widths shown above; higher bits are truncated
 * in the packed key.
 * @return Packed KernelKey
 */
constexpr KernelKey
make(::orteaf::internal::ops::Op op,
     ::orteaf::internal::architecture::Architecture architecture, Layout layout,
     ::orteaf::internal::DType dtype, Variant variant,
     std::uint8_t version = kCurrentVersion) noexcept {
  const std::uint64_t version_bits =
      (static_cast<std::uint64_t>(version) & kVersionMask) << kVersionShift;
  const std::uint64_t variant_bits =
      (static_cast<std::uint64_t>(variant) & kVariantMask) << kVariantShift;
  const std::uint64_t layout_bits =
      (static_cast<std::uint64_t>(layout) & kLayoutMask) << kLayoutShift;
  const std::uint64_t arch_bits =
      (static_cast<std::uint64_t>(architecture) & kArchitectureMask)
      << kArchitectureShift;
  const std::uint64_t op_bits = (static_cast<std::uint64_t>(op) & kOpMask)
                                << kOpShift;
  const std::uint64_t dtype_bits =
      (static_cast<std::uint64_t>(dtype) & kDTypeMask) << kDTypeShift;

  return static_cast<KernelKey>(version_bits | variant_bits | layout_bits |
                                arch_bits | op_bits | dtype_bits);
}

/**
 * @brief Extract Version from a KernelKey.
 */
constexpr std::uint8_t getVersion(KernelKey key) noexcept {
  const std::uint64_t value = static_cast<std::uint64_t>(key);
  return static_cast<std::uint8_t>((value >> kVersionShift) & kVersionMask);
}

/**
 * @brief Extract Op from a KernelKey.
 */
constexpr ::orteaf::internal::ops::Op getOp(KernelKey key) noexcept {
  const std::uint64_t value = static_cast<std::uint64_t>(key);
  return static_cast<::orteaf::internal::ops::Op>((value >> kOpShift) &
                                                  kOpMask);
}

/**
 * @brief Extract Architecture from a KernelKey.
 */
constexpr ::orteaf::internal::architecture::Architecture
getArchitecture(KernelKey key) noexcept {
  const std::uint64_t value = static_cast<std::uint64_t>(key);
  return static_cast<::orteaf::internal::architecture::Architecture>(
      (value >> kArchitectureShift) & kArchitectureMask);
}

/**
 * @brief Extract Execution from a KernelKey (derived from Architecture).
 */
constexpr ::orteaf::internal::execution::Execution
getExecution(KernelKey key) noexcept {
  return ::orteaf::internal::architecture::executionOf(getArchitecture(key));
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
