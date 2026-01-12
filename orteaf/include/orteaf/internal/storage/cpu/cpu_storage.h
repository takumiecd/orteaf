#pragma once

#include <utility>

#include <orteaf/internal/diagnostics/error/error.h>
#include <orteaf/internal/execution/cpu/api/cpu_execution_api.h>
#include <orteaf/internal/execution/cpu/manager/cpu_device_manager.h>
#include <orteaf/internal/storage/cpu/cpu_storage_layout.h>

namespace orteaf::internal::storage::cpu {

class CpuStorage {
public:
  using BufferManager =
      ::orteaf::internal::execution::cpu::manager::CpuBufferManager;
  using DeviceManager =
      ::orteaf::internal::execution::cpu::manager::CpuDeviceManager;
  using DeviceHandle = DeviceManager::DeviceHandle;
  using DeviceLease = DeviceManager::DeviceLease;
  using BufferLease = BufferManager::BufferLease;
  using Layout = ::orteaf::internal::storage::cpu::CpuStorageLayout;

  /**
   * @brief Builder for constructing CpuStorage instances.
   *
   * Provides a fluent interface for configuring and creating CpuStorage.
   * The builder acquires a buffer lease from the manager upon build().
   *
   * @par Example
   * @code
   * auto storage = CpuStorage::builder()
   *     .withDeviceLease(device_lease)
   *     .withSize(1024)
   *     .withAlignment(16)
   *     .withLayout(layout)
   *     .build();
   * @endcode
   */
  class Builder {
  public:
    Builder() = default;

    Builder &withDeviceLease(const DeviceLease &lease) {
      device_lease_ = lease;
      return *this;
    }

    Builder &withDeviceHandle(DeviceHandle handle) {
      device_lease_ = ::orteaf::internal::execution::cpu::api::
          CpuExecutionApi::acquireDevice(handle);
      return *this;
    }

    Builder &withSize(std::size_t size) {
      size_ = size;
      return *this;
    }

    Builder &withAlignment(std::size_t alignment) {
      alignment_ = alignment;
      return *this;
    }

    Builder &withLayout(Layout layout) {
      layout_ = std::move(layout);
      return *this;
    }

    /**
     * @brief Build the CpuStorage instance.
     *
     * Acquires a buffer lease from the configured manager and
     * constructs the CpuStorage.
     *
     * @return Constructed CpuStorage instance.
     * @throws If buffer_manager is null or acquisition fails.
     */
    CpuStorage build() {
      if (!device_lease_) {
        ::orteaf::internal::diagnostics::error::throwError(
            ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
            "CpuStorage requires a valid device lease");
      }
      BufferLease lease =
          device_lease_->buffer_manager.acquire(size_, alignment_);
      return CpuStorage(std::move(lease), std::move(layout_));
    }

  private:
    DeviceLease device_lease_{};
    std::size_t size_{0};
    std::size_t alignment_{0};
    Layout layout_{};
  };

  /**
   * @brief Create a new Builder instance.
   * @return A new Builder for constructing CpuStorage.
   */
  static Builder builder() { return Builder{}; }

  CpuStorage() = default;

  CpuStorage(const CpuStorage &) = default;
  CpuStorage &operator=(const CpuStorage &) = default;
  CpuStorage(CpuStorage &&) = default;
  CpuStorage &operator=(CpuStorage &&) = default;
  ~CpuStorage() = default;

private:
  CpuStorage(BufferLease buffer_lease, Layout layout)
      : buffer_lease_(std::move(buffer_lease)), layout_(std::move(layout)) {}

  BufferLease buffer_lease_;
  Layout layout_;
};

} // namespace orteaf::internal::storage::cpu
