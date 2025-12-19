#include "orteaf/internal/runtime/mps/manager/mps_device_manager.h"

#if ORTEAF_ENABLE_MPS

namespace orteaf::internal::runtime::mps::manager {

void MpsDeviceManager::initialize(SlowOps *slow_ops) {
  shutdown();

  if (slow_ops == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS device manager requires valid ops");
  }
  ops_ = slow_ops;

  const int device_count = ops_->getDeviceCount();
  if (device_count <= 0) {
    Base::setupPoolEmpty();
    return;
  }

  // Setup pool with device count
  Base::setupPool(static_cast<std::size_t>(device_count));

  // Initialize devices in a separate loop since create() only gets Payload
  for (int i = 0; i < device_count; ++i) {
    const DeviceHandle handle{static_cast<std::uint32_t>(i)};
    auto &cb = Base::getControlBlock(handle);

    cb.acquire([this, i, handle](MpsDeviceResource &resource) {
      const auto device = ops_->getDevice(
          static_cast<
              ::orteaf::internal::runtime::mps::platform::wrapper::MPSInt_t>(
              i));
      resource.device = device;

      if (device == nullptr) {
        resource.arch =
            ::orteaf::internal::architecture::Architecture::MpsGeneric;
        return false;
      }

      resource.arch = ops_->detectArchitecture(handle);
      resource.command_queue_manager.initialize(
          device, ops_, command_queue_initial_capacity_);
      resource.library_manager.initialize(device, ops_,
                                          library_initial_capacity_);
      resource.heap_manager.initialize(device, handle,
                                       &resource.library_manager, ops_,
                                       heap_initial_capacity_);
      resource.graph_manager.initialize(device, ops_, graph_initial_capacity_);
      resource.event_pool.initialize(device, ops_, 0);
      resource.fence_pool.initialize(device, ops_, 0);
      return true;
    });
  }
    }

void MpsDeviceManager::shutdown() {
  Base::teardownPool([this](MpsDeviceResource &payload) {
    payload.reset(ops_);
  });
      ops_ = nullptr;
}

MpsDeviceManager::DeviceLease MpsDeviceManager::acquire(DeviceHandle handle) {
      auto &cb = Base::acquireExisting(handle);
      Base::acquireWeakRef(handle);
      return DeviceLease{this, handle, cb.payload().device};
}

void MpsDeviceManager::release(DeviceLease &lease) noexcept {
  Base::releaseWeakRef(lease.handle());
  lease.invalidate();
}

::orteaf::internal::architecture::Architecture
MpsDeviceManager::getArch(DeviceHandle handle) const {
      return ensureValidControlBlockConst(handle).payload().arch;
}

MpsCommandQueueManager *
MpsDeviceManager::commandQueueManager(DeviceHandle handle) {
      return &ensureValidControlBlock(handle).payload().command_queue_manager;
}

MpsHeapManager *MpsDeviceManager::heapManager(DeviceHandle handle) {
      return &ensureValidControlBlock(handle).payload().heap_manager;
}

MpsLibraryManager *MpsDeviceManager::libraryManager(DeviceHandle handle) {
      return &ensureValidControlBlock(handle).payload().library_manager;
}

MpsGraphManager *MpsDeviceManager::graphManager(DeviceHandle handle) {
      return &ensureValidControlBlock(handle).payload().graph_manager;
}

MpsEventManager *MpsDeviceManager::eventPool(DeviceHandle handle) {
      return &ensureValidControlBlock(handle).payload().event_pool;
}

MpsFenceManager *MpsDeviceManager::fencePool(DeviceHandle handle) {
      return &ensureValidControlBlock(handle).payload().fence_pool;
}

MpsDeviceManager::ControlBlock &
MpsDeviceManager::ensureValidControlBlock(DeviceHandle handle) {
      Base::ensureInitialized();
      if (!Base::isValidHandle(handle)) {
        ::orteaf::internal::diagnostics::error::throwError(
            ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
            "MPS device handle is invalid");
      }
      ControlBlock &cb = Base::getControlBlock(handle);
      if (cb.isCreated()) {
        ::orteaf::internal::diagnostics::error::throwError(
            ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
            "MPS device is unavailable");
      }
      return cb;
}

const MpsDeviceManager::ControlBlock &
MpsDeviceManager::ensureValidControlBlockConst(DeviceHandle handle) const {
      if (!Base::isInitialized()) {
        ::orteaf::internal::diagnostics::error::throwError(
            ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
            "MPS devices not initialized");
      }
      if (!Base::isValidHandle(handle)) {
        ::orteaf::internal::diagnostics::error::throwError(
            ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
            "MPS device handle is invalid");
      }
      const ControlBlock &cb = Base::getControlBlock(handle);
      if (cb.isCreated()) {
        ::orteaf::internal::diagnostics::error::throwError(
            ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
            "MPS device is unavailable");
      }
      return cb;
}

  } // namespace orteaf::internal::runtime::mps::manager

#endif // ORTEAF_ENABLE_MPS
