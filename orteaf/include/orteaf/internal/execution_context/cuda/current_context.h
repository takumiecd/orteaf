#pragma once

#if ORTEAF_ENABLE_CUDA

#include "orteaf/internal/execution_context/cuda/context.h"

namespace orteaf::internal::execution_context::cuda {

class CurrentContext {
public:
  Context current{};
};

const CurrentContext &current();
void setCurrent(CurrentContext state);
void setCurrentContext(Context context);
void reset();

const Context &currentContext();
Context::DeviceLease currentDevice();
Context::Architecture currentArchitecture();
Context::ContextLease currentCudaContext();
Context::StreamLease currentStream();

} // namespace orteaf::internal::execution_context::cuda

#endif // ORTEAF_ENABLE_CUDA
