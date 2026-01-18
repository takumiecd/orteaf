#pragma once

#if ORTEAF_ENABLE_MPS

#include "orteaf/internal/execution_context/mps/context.h"

namespace orteaf::internal::execution_context::mps {

class CurrentContext {
public:
  Context current{};
};

const CurrentContext &current();
void setCurrent(CurrentContext state);
void reset();

const Context &currentContext();
Context::DeviceLease currentDevice();
Context::CommandQueueLease currentCommandQueue();

} // namespace orteaf::internal::execution_context::mps

#endif // ORTEAF_ENABLE_MPS
