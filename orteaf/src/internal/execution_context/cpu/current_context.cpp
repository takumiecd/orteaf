#include "orteaf/internal/execution_context/cpu/current_context.h"

#include "orteaf/internal/execution/cpu/api/cpu_execution_api.h"

namespace orteaf::internal::execution_context::cpu {
namespace {

CurrentContext &currentStateStorage() {
  static CurrentContext state{};
  return state;
}

void ensureDefaultContext(CurrentContext &state) {
  if (state.current.device) {
    return;
  }
  state.current.device =
      ::orteaf::internal::execution::cpu::api::CpuExecutionApi::acquireDevice(
          ::orteaf::internal::execution::cpu::CpuDeviceHandle{0});
}

} // namespace

const CurrentContext &current() {
  auto &state = currentStateStorage();
  ensureDefaultContext(state);
  return state;
}

void setCurrent(CurrentContext state) {
  currentStateStorage() = std::move(state);
}

void reset() { currentStateStorage() = CurrentContext{}; }

const Context &currentContext() {
  auto &state = currentStateStorage();
  ensureDefaultContext(state);
  return state.current;
}

Context::DeviceLease currentDevice() {
  auto &state = currentStateStorage();
  ensureDefaultContext(state);
  return state.current.device;
}

} // namespace orteaf::internal::execution_context::cpu
