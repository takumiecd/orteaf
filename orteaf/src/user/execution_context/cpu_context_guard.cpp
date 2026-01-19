#include "orteaf/user/execution_context/cpu_context_guard.h"

#include "orteaf/internal/execution/cpu/api/cpu_execution_api.h"

#include <utility>

namespace orteaf::user::execution_context {
namespace {

namespace cpu_context = ::orteaf::internal::execution_context::cpu;
namespace cpu_api = ::orteaf::internal::execution::cpu::api;

cpu_context::Context makeContext(
    ::orteaf::internal::execution::cpu::CpuDeviceHandle device) {
  cpu_context::Context context{};
  context.device = cpu_api::CpuExecutionApi::acquireDevice(device);
  return context;
}

} // namespace

CpuExecutionContextGuard::CpuExecutionContextGuard() {
  activate(makeContext(::orteaf::internal::execution::cpu::CpuDeviceHandle{0}));
}

CpuExecutionContextGuard::CpuExecutionContextGuard(
    ::orteaf::internal::execution::cpu::CpuDeviceHandle device) {
  activate(makeContext(device));
}

CpuExecutionContextGuard::CpuExecutionContextGuard(
    CpuExecutionContextGuard &&other) noexcept
    : previous_(std::move(other.previous_)), active_(other.active_) {
  other.active_ = false;
}

CpuExecutionContextGuard &CpuExecutionContextGuard::operator=(
    CpuExecutionContextGuard &&other) noexcept {
  if (this != &other) {
    release();
    previous_ = std::move(other.previous_);
    active_ = other.active_;
    other.active_ = false;
  }
  return *this;
}

CpuExecutionContextGuard::~CpuExecutionContextGuard() { release(); }

void CpuExecutionContextGuard::activate(
    ::orteaf::internal::execution_context::cpu::Context context) {
  previous_.current = cpu_context::currentContext();
  cpu_context::setCurrentContext(std::move(context));
  active_ = true;
}

void CpuExecutionContextGuard::release() noexcept {
  if (!active_) {
    return;
  }
  cpu_context::setCurrent(std::move(previous_));
  active_ = false;
}

} // namespace orteaf::user::execution_context
