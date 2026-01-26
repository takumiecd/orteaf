#pragma once

#include <orteaf/internal/architecture/architecture.h>
#include <orteaf/internal/dtype/dtype.h>
#include <orteaf/internal/execution/execution.h>
#include <orteaf/internal/kernel/core/kernel_key.h>
#include <orteaf/internal/kernel/core/layout.h>
#include <orteaf/internal/kernel/core/variant.h>
#include <orteaf/internal/ops/ops.h>

namespace orteaf::internal::kernel {

/**
 * @brief Fixed key components that cannot be changed without altering
 * semantics.
 *
 * Changing these would change the meaning of the computation:
 * - Op: The operation being performed
 * - DType: Data type (changing requires cast)
 * - Execution: Backend (changing requires device transfer)
 */
struct FixedKeyComponents {
  ops::Op op;
  DType dtype;
  execution::Execution execution;

  constexpr bool operator==(const FixedKeyComponents &other) const noexcept {
    return op == other.op && dtype == other.dtype &&
           execution == other.execution;
  }

  constexpr bool operator!=(const FixedKeyComponents &other) const noexcept {
    return !(*this == other);
  }
};

/**
 * @brief Variable key components that can be relaxed for fallback.
 *
 * These can be changed to find a compatible kernel:
 * - Architecture: M3 → MpsGeneric
 * - Layout: ContiguousNHWC → Contiguous
 * - Variant: Vectorized → Default
 */
struct VariableKeyComponents {
  architecture::Architecture arch;
  Layout layout;
  Variant variant;

  constexpr bool operator==(const VariableKeyComponents &other) const noexcept {
    return arch == other.arch && layout == other.layout &&
           variant == other.variant;
  }

  constexpr bool operator!=(const VariableKeyComponents &other) const noexcept {
    return !(*this == other);
  }
};

/**
 * @brief Create a KernelKey from fixed and variable components.
 */
constexpr KernelKey makeKey(const FixedKeyComponents &fixed,
                            const VariableKeyComponents &variable) noexcept {
  return kernel_key::make(fixed.op, variable.arch, variable.layout, fixed.dtype,
                          variable.variant);
}

// Forward declaration for KernelArgs
class KernelArgs;

/**
 * @brief Predicate function type for rule verification.
 *
 * Returns true if the rule is applicable for the given args.
 */
using KeyPredicate = bool (*)(const KernelArgs &);

/**
 * @brief Rule combining variable components with optional custom predicate.
 *
 * If predicate is nullptr, the default verification logic is used.
 * Otherwise, the custom predicate takes precedence.
 */
struct KeyRule {
  VariableKeyComponents components;
  KeyPredicate predicate = nullptr; // null = use default logic

  constexpr bool operator==(const KeyRule &other) const noexcept {
    return components == other.components && predicate == other.predicate;
  }

  constexpr bool operator!=(const KeyRule &other) const noexcept {
    return !(*this == other);
  }
};

} // namespace orteaf::internal::kernel
