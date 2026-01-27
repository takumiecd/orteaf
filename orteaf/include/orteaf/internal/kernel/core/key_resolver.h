#pragma once

#include <optional>

#include <orteaf/internal/base/small_vector.h>
#include <orteaf/internal/kernel/core/kernel_args.h>
#include <orteaf/internal/kernel/core/key_components.h>

namespace orteaf::internal::kernel::key_resolver {

/**
 * @brief Context for key resolution, built from a KeyRequest.
 *
 * Contains the fixed components and the prioritized list of rules to try.
 */
struct ResolveContext {
  FixedKeyComponents fixed;
  base::SmallVector<KeyRule, 8> rules;
};

/**
 * @brief Build a resolution context from a key request.
 *
 * Extracts fixed components and generates prioritized rules based on
 * the request's architecture (specific → generic fallback).
 *
 * @param request The key request (Op, DType, Architecture)
 * @return Context containing fixed components and rules
 */
ResolveContext buildContext(const KeyRequest &request);

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
 * Uses custom predicate if provided, otherwise falls back to defaultVerify.
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
 * 1. Build context from request
 * 2. For each rule (in priority order):
 *    - Verify against args (custom predicate or default)
 *    - Check if key exists in registry
 *    - Return first match
 *
 * @tparam Registry Type that supports contains(KernelKey) method
 * @param registry The kernel registry to check for existence
 * @param request The key request (Op, DType, Architecture)
 * @param args The kernel arguments for verification
 * @return The resolved KernelKey, or nullopt if none found
 */
template <typename Registry>
std::optional<KernelKey> resolve(const Registry &registry,
                                 const KeyRequest &request,
                                 const KernelArgs &args) {
  auto context = buildContext(request);

  for (const auto &rule : context.rules) {
    if (verify(rule, args)) {
      auto key = makeKey(context.fixed, rule.components);
      if (registry.contains(key)) {
        return key;
      }
    }
  }

  return std::nullopt;
}

} // namespace orteaf::internal::kernel::key_resolver
