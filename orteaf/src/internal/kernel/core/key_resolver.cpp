#include "orteaf/internal/kernel/core/key_resolver.h"

#include <orteaf/internal/architecture/architecture.h>

namespace orteaf::internal::kernel::key_resolver {

namespace arch = ::orteaf::internal::architecture;

ResolveContext buildContext(const KeyRequest &request) {
  ResolveContext context;

  // Extract fixed components from request
  context.fixed.op = request.op;
  context.fixed.dtype = request.dtype;

  // Build fallback chain using parent hierarchy
  // Order: requested arch → parent → parent's parent → ... → Generic
  arch::forEachFallback(
      request.architecture, [&](arch::Architecture architecture) {
        context.rules.pushBack({
            {architecture, static_cast<Layout>(0), static_cast<Variant>(0)},
            nullptr,
        });
      });

  return context;
}

bool defaultVerify(const VariableKeyComponents & /*components*/,
                   const KernelArgs & /*args*/) {
  // TODO: Implement actual verification logic based on Layout/Variant
  // For now, always return true (accept all candidates)
  return true;
}

} // namespace orteaf::internal::kernel::key_resolver
