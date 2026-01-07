#include "orteaf/internal/execution/cpu/manager/cpu_device_manager.h"

#include "orteaf/internal/diagnostics/error/error.h"

namespace orteaf::internal::execution::cpu::manager {

// =============================================================================
// CpuDeviceResource Implementation
// =============================================================================

CpuDeviceResource::CpuDeviceResource(CpuDeviceResource &&other) noexcept {
  moveFrom(std::move(other));
}

CpuDeviceResource &
CpuDeviceResource::operator=(CpuDeviceResource &&other) noexcept {
  if (this != &other) {
    reset(nullptr);
    moveFrom(std::move(other));
  }
  return *this;
}

CpuDeviceResource::~CpuDeviceResource() { reset(nullptr); }

void CpuDeviceResource::reset([[maybe_unused]] SlowOps *slow_ops) noexcept {
  // Shutdown sub-managers here when added
  arch = ::orteaf::internal::architecture::Architecture::CpuGeneric;
  is_alive = false;
}

void CpuDeviceResource::moveFrom(CpuDeviceResource &&other) noexcept {
  arch = other.arch;
  is_alive = other.is_alive;
  other.arch = ::orteaf::internal::architecture::Architecture::CpuGeneric;
  other.is_alive = false;
}

// =============================================================================
// DevicePayloadPoolTraits Implementation
// =============================================================================

bool DevicePayloadPoolTraits::create(Payload &payload, const Request &request,
                                     const Context &context) {
  if (context.ops == nullptr || !request.handle.isValid()) {
    return false;
  }

  // CPU has only one device with index 0
  if (request.handle.index != 0) {
    return false;
  }

  payload.arch = context.ops->detectArchitecture(request.handle);
  payload.is_alive = true;

  // Initialize sub-managers here when added
  return true;
}

void DevicePayloadPoolTraits::destroy(Payload &payload, const Request &,
                                      const Context &context) {
  payload.reset(context.ops);
}

// =============================================================================
// CpuDeviceManager Implementation
// =============================================================================

void CpuDeviceManager::configure(const Config &config) {
  shutdown();

  if (config.ops == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "CPU device manager requires valid ops");
  }

  ops_ = config.ops;

  // Setup pool configuration
  auto pool_config = config.pool;
  // CPU has exactly 1 device
  pool_config.payload_capacity = 1;
  pool_config.payload_block_size = 1;
  if (pool_config.control_block_capacity == 0) {
    pool_config.control_block_capacity = 4;
  }
  if (pool_config.control_block_block_size == 0) {
    pool_config.control_block_block_size = 4;
  }

  DevicePayloadPoolTraits::Request request{};
  request.handle = DeviceHandle{0};

  DevicePayloadPoolTraits::Context context{};
  context.ops = ops_;

  core_.configure(pool_config, request, context);
  core_.createAllPayloads(request, context);
}

void CpuDeviceManager::shutdown() {
  lifetime_.clear();

  DevicePayloadPoolTraits::Request request{};
  DevicePayloadPoolTraits::Context context{};
  context.ops = ops_;

  core_.shutdown(request, context);
  ops_ = nullptr;
}

CpuDeviceManager::DeviceLease CpuDeviceManager::acquire(DeviceHandle handle) {
  core_.ensureConfigured();

  if (!handle.isValid() || handle.index != 0) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "invalid CPU device handle");
  }

  // Check if we already have an active lease
  if (lifetime_.has(handle)) {
    return lifetime_.get(handle);
  }

  // Create new lease
  auto lease = core_.acquireStrongLease(handle);
  if (lease) {
    lifetime_.set(DeviceLease{lease});
  }
  return lease;
}

void CpuDeviceManager::release(DeviceLease &lease) noexcept { lease.release(); }

::orteaf::internal::architecture::Architecture
CpuDeviceManager::getArch(DeviceHandle handle) {
  core_.ensureConfigured();

  if (!handle.isValid() || handle.index != 0) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "invalid CPU device handle");
  }

  // Get lease from lifetime registry
  if (!lifetime_.has(handle)) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "CPU device not acquired");
  }

  auto lease = lifetime_.get(handle);
  if (!lease) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "CPU device lease is invalid");
  }

  return lease.payloadPtr()->arch;
}

std::size_t CpuDeviceManager::getDeviceCount() const noexcept {
  return core_.isConfigured() ? 1u : 0u;
}

bool CpuDeviceManager::isAlive(DeviceHandle handle) const noexcept {
  if (!core_.isConfigured() || !handle.isValid() || handle.index != 0) {
    return false;
  }
  return core_.isAlive(handle);
}

#if ORTEAF_ENABLE_TEST
std::size_t CpuDeviceManager::getDeviceCountForTest() const noexcept {
  return core_.payloadPoolSizeForTest();
}

bool CpuDeviceManager::isConfiguredForTest() const noexcept {
  return core_.isConfigured();
}

std::size_t CpuDeviceManager::payloadPoolSizeForTest() const noexcept {
  return core_.payloadPoolSizeForTest();
}

std::size_t CpuDeviceManager::payloadPoolCapacityForTest() const noexcept {
  return core_.payloadPoolCapacityForTest();
}

bool CpuDeviceManager::isAliveForTest(DeviceHandle handle) const noexcept {
  return core_.isAlive(handle);
}

std::size_t
CpuDeviceManager::controlBlockPoolAvailableForTest() const noexcept {
  return core_.controlBlockPoolAvailableForTest();
}
#endif

} // namespace orteaf::internal::execution::cpu::manager
