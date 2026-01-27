#include "orteaf/internal/kernel/core/key_resolver.h"

#include <orteaf/internal/architecture/architecture.h>

namespace orteaf::internal::kernel::key_resolver {

namespace arch = ::orteaf::internal::architecture;

ResolveContext buildContext(const KeyRequest &request) {
  ResolveContext context;

  // Extract fixed components from request
  context.fixed.op = request.op;
  context.fixed.dtype = request.dtype;

  // Get execution from architecture
  auto execution = arch::executionOf(request.architecture);

  // Get all architectures for this execution backend
  auto architectures = arch::architecturesOf(execution);

  // Add rules: requested architecture first, then fallback to less specific
  // Order: requested arch → other specific archs → generic (index 0)

  // Start with the requested architecture
  context.rules.pushBack({
      {request.architecture, static_cast<Layout>(0), static_cast<Variant>(0)},
      nullptr,
  });

  // Add other specific architectures (indices > 0, excluding requested)
  for (std::size_t i = architectures.size(); i > 1; --i) {
    auto architecture = architectures[i - 1];
    if (architecture != request.architecture) {
      context.rules.pushBack({
          {architecture, static_cast<Layout>(0), static_cast<Variant>(0)},
          nullptr,
      });
    }
  }

  // Add generic architecture last (index 0) if not already the requested one
  if (!architectures.empty() && architectures[0] != request.architecture) {
    context.rules.pushBack({
        {architectures[0], static_cast<Layout>(0), static_cast<Variant>(0)},
        nullptr,
    });
  }

  return context;
}

bool defaultVerify(const VariableKeyComponents & /*components*/,
                   const KernelArgs & /*args*/) {
  // TODO: Implement actual verification logic based on Layout/Variant
  // For now, always return true (accept all candidates)
  return true;
}

} // namespace orteaf::internal::kernel::key_resolver
