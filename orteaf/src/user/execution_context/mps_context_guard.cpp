#include "orteaf/user/execution_context/mps_context_guard.h"

#if ORTEAF_ENABLE_MPS

#include "orteaf/internal/execution/mps/api/mps_execution_api.h"

#include <utility>

namespace orteaf::user::execution_context {
namespace {

namespace mps_context = ::orteaf::internal::execution_context::mps;
namespace mps_api = ::orteaf::internal::execution::mps::api;

mps_context::Context makeContext(
    ::orteaf::internal::execution::mps::MpsDeviceHandle device) {
  mps_context::Context context{};
  context.device = mps_api::MpsExecutionApi::acquireDevice(device);
  if (auto *resource = context.device.operator->()) {
    context.command_queue = resource->command_queue_manager.acquire();
  }
  return context;
}

mps_context::Context makeContext(
    ::orteaf::internal::execution::mps::MpsDeviceHandle device,
    ::orteaf::internal::execution::mps::MpsCommandQueueHandle command_queue) {
  mps_context::Context context{};
  context.device = mps_api::MpsExecutionApi::acquireDevice(device);
  if (auto *resource = context.device.operator->()) {
    context.command_queue =
        resource->command_queue_manager.acquire(command_queue);
  }
  return context;
}

} // namespace

MpsExecutionContextGuard::MpsExecutionContextGuard() {
  activate(makeContext(::orteaf::internal::execution::mps::MpsDeviceHandle{0}));
}

MpsExecutionContextGuard::MpsExecutionContextGuard(
    ::orteaf::internal::execution::mps::MpsDeviceHandle device) {
  activate(makeContext(device));
}

MpsExecutionContextGuard::MpsExecutionContextGuard(
    ::orteaf::internal::execution::mps::MpsDeviceHandle device,
    ::orteaf::internal::execution::mps::MpsCommandQueueHandle command_queue) {
  activate(makeContext(device, command_queue));
}

MpsExecutionContextGuard::MpsExecutionContextGuard(
    MpsExecutionContextGuard &&other) noexcept
    : previous_(std::move(other.previous_)), active_(other.active_) {
  other.active_ = false;
}

MpsExecutionContextGuard &MpsExecutionContextGuard::operator=(
    MpsExecutionContextGuard &&other) noexcept {
  if (this != &other) {
    release();
    previous_ = std::move(other.previous_);
    active_ = other.active_;
    other.active_ = false;
  }
  return *this;
}

MpsExecutionContextGuard::~MpsExecutionContextGuard() { release(); }

void MpsExecutionContextGuard::activate(
    ::orteaf::internal::execution_context::mps::Context context) {
  previous_.current = mps_context::currentContext();
  mps_context::setCurrentContext(std::move(context));
  active_ = true;
}

void MpsExecutionContextGuard::release() noexcept {
  if (!active_) {
    return;
  }
  mps_context::setCurrent(std::move(previous_));
  active_ = false;
}

} // namespace orteaf::user::execution_context

#endif // ORTEAF_ENABLE_MPS
