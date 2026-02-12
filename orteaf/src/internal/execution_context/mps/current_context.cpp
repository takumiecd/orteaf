#include "orteaf/internal/execution_context/mps/current_context.h"

#if ORTEAF_ENABLE_MPS

#include "orteaf/internal/execution/mps/api/mps_execution_api.h"

namespace orteaf::internal::execution_context::mps {
namespace {

CurrentContext &currentStateStorage() {
  static CurrentContext state{};
  return state;
}

void ensureDefaultDevice(CurrentContext &state) {
  if (state.current.device) {
    return;
  }
  state.current.device =
      ::orteaf::internal::execution::mps::api::MpsExecutionApi::acquireDevice(
          ::orteaf::internal::execution::mps::MpsDeviceHandle{0});
}

void ensureDefaultCommandQueue(CurrentContext &state) {
  if (state.current.command_queue) {
    return;
  }
  ensureDefaultDevice(state);
  auto *resource = state.current.device.operator->();
  if (resource == nullptr) {
    return;
  }
  state.current.command_queue = resource->commandQueueManager().acquire();
}

} // namespace

const CurrentContext &current() {
  auto &state = currentStateStorage();
  ensureDefaultCommandQueue(state);
  return state;
}

void setCurrent(CurrentContext state) {
  currentStateStorage() = std::move(state);
}

void setCurrentContext(Context context) {
  currentStateStorage().current = std::move(context);
}

void reset() { currentStateStorage() = CurrentContext{}; }

const Context &currentContext() {
  auto &state = currentStateStorage();
  ensureDefaultCommandQueue(state);
  return state.current;
}

Context::DeviceLease currentDevice() {
  auto &state = currentStateStorage();
  ensureDefaultDevice(state);
  return state.current.device;
}

Context::Architecture currentArchitecture() {
  auto &state = currentStateStorage();
  ensureDefaultDevice(state);
  return state.current.architecture();
}

Context::CommandQueueLease currentCommandQueue() {
  auto &state = currentStateStorage();
  ensureDefaultCommandQueue(state);
  return state.current.command_queue;
}

} // namespace orteaf::internal::execution_context::mps

#endif // ORTEAF_ENABLE_MPS
