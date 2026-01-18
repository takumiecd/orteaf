#pragma once

#include "orteaf/internal/execution_context/cpu/context.h"

namespace orteaf::internal::execution_context::cpu {

class CurrentContext {
public:
  Context current{};
};

const CurrentContext &current();
void setCurrent(CurrentContext state);
void reset();

const Context &currentContext();
Context::DeviceLease currentDevice();

} // namespace orteaf::internal::execution_context::cpu
