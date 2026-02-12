#include "orteaf/internal/execution_context/cuda/current_context.h"

#if ORTEAF_ENABLE_CUDA

#include "orteaf/internal/execution/cuda/api/cuda_execution_api.h"

namespace orteaf::internal::execution_context::cuda {
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
      ::orteaf::internal::execution::cuda::api::CudaExecutionApi::acquireDevice(
          ::orteaf::internal::execution::cuda::CudaDeviceHandle{0});
}

void ensureDefaultContext(CurrentContext &state) {
  if (state.current.context) {
    return;
  }
  ensureDefaultDevice(state);
  auto *device_resource = state.current.device.operator->();
  if (device_resource == nullptr) {
    return;
  }
  state.current.context = device_resource->context_manager.acquirePrimary();
}

void ensureDefaultStream(CurrentContext &state) {
  if (state.current.stream) {
    return;
  }
  ensureDefaultContext(state);
  auto *context_resource = state.current.context.operator->();
  if (context_resource == nullptr) {
    return;
  }
  state.current.stream = context_resource->stream_manager.acquire();
}

} // namespace

const CurrentContext &current() {
  auto &state = currentStateStorage();
  ensureDefaultStream(state);
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
  ensureDefaultStream(state);
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

Context::ContextLease currentCudaContext() {
  auto &state = currentStateStorage();
  ensureDefaultContext(state);
  return state.current.context;
}

Context::StreamLease currentStream() {
  auto &state = currentStateStorage();
  ensureDefaultStream(state);
  return state.current.stream;
}

} // namespace orteaf::internal::execution_context::cuda

#endif // ORTEAF_ENABLE_CUDA
