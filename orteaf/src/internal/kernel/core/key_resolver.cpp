#include "orteaf/internal/kernel/core/key_resolver.h"

#include <orteaf/internal/architecture/architecture.h>

namespace orteaf::internal::kernel::key_resolver {

namespace arch = ::orteaf::internal::architecture;

base::SmallVector<KeyRule, 8> getRules(const FixedKeyComponents &fixed) {
  base::SmallVector<KeyRule, 8> rules;

  // Get all architectures for this execution backend
  auto architectures = arch::architecturesOf(fixed.execution);

  // Add rules in reverse order (specific architectures first, then generic)
  // For each architecture, add Default layout and variant first
  // TODO: Expand with more Layout/Variant combinations based on Op

  // Specific architectures first (skip index 0 which is Generic)
  for (std::size_t i = architectures.size(); i > 1; --i) {
    auto architecture = architectures[i - 1];
    rules.pushBack({
        {architecture, static_cast<Layout>(0), static_cast<Variant>(0)},
        nullptr // Use default verify logic
    });
  }

  // Generic architecture last (fallback)
  if (!architectures.empty()) {
    auto generic = architectures[0]; // Index 0 is always Generic
    rules.pushBack({
        {generic, static_cast<Layout>(0), static_cast<Variant>(0)},
        nullptr // Use default verify logic
    });
  }

  return rules;
}

bool defaultVerify(const VariableKeyComponents & /*components*/,
                   const KernelArgs & /*args*/) {
  // TODO: Implement actual verification logic based on Layout/Variant
  // For now, always return true (accept all candidates)
  // Future: Check layout compatibility, tensor contiguity, etc.
  //
  // Example logic:
  // switch (components.layout) {
  //   case Layout::ContiguousNHWC: return isNHWCContiguous(args);
  //   case Layout::Contiguous: return isContiguous(args);
  //   default: return true;
  // }
  return true;
}

} // namespace orteaf::internal::kernel::key_resolver
