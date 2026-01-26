#pragma once

#include <optional>

#include <orteaf/internal/base/small_vector.h>
#include <orteaf/internal/kernel/core/kernel_args.h>
#include <orteaf/internal/kernel/core/key_components.h>

namespace orteaf::internal::kernel::key_resolver {

/**
 * @brief Get candidate rules for the given fixed components.
 *
 * Phase 1: Returns a list of KeyRules in priority order (highest priority
 * first). This does NOT check the KernelArgs - that happens in verify().
 *
 * @param fixed The fixed key components (Op, DType, Execution)
 * @return Priority-ordered list of candidate rules
 */
base::SmallVector<KeyRule, 8> getRules(const FixedKeyComponents &fixed);

/**
 * @brief Default verification logic based on variable components.
 *
 * Checks if the candidate is applicable based on Layout/Variant requirements.
 * Used when KeyRule::predicate is nullptr.
 *
 * @param components The variable components to verify
 * @param args The kernel arguments to check against
 * @return true if the candidate is applicable
 */
bool defaultVerify(const VariableKeyComponents &components,
                   const KernelArgs &args);

/**
 * @brief Verify if a rule is applicable for the given args.
 *
 * Phase 2: Uses custom predicate if provided, otherwise falls back to
 * defaultVerify.
 *
 * @param rule The rule to verify
 * @param args The kernel arguments to check against
 * @return true if the rule is applicable
 */
inline bool verify(const KeyRule &rule, const KernelArgs &args) {
  if (rule.predicate != nullptr) {
    return rule.predicate(args);
  }
  return defaultVerify(rule.components, args);
}

/**
 * @brief Resolve the best matching KernelKey.
 *
 * Combines Phase 1 and Phase 2:
 * 1. Get rules from fixed components
 * 2. For each rule (in priority order):
 *    - Verify against args (custom predicate or default)
 *    - Check if key exists in registry
 *    - Return first match
 *
 * @tparam Registry Type that supports contains(KernelKey) method
 * @param registry The kernel registry to check for existence
 * @param fixed The fixed key components
 * @param args The kernel arguments for verification
 * @return The resolved KernelKey, or nullopt if none found
 */
template <typename Registry>
std::optional<KernelKey> resolve(const Registry &registry,
                                 const FixedKeyComponents &fixed,
                                 const KernelArgs &args) {
  auto rules = getRules(fixed);

  for (const auto &rule : rules) {
    if (verify(rule, args)) {
      auto key = makeKey(fixed, rule.components);
      if (registry.contains(key)) {
        return key;
      }
    }
  }

  return std::nullopt;
}

} // namespace orteaf::internal::kernel::key_resolver
