#pragma once

#include <utility>

#include <orteaf/internal/diagnostics/error/error.h>
#include <orteaf/internal/dtype/dtype.h>
#include <orteaf/internal/execution/cpu/api/cpu_execution_api.h>
#include <orteaf/internal/execution/cpu/manager/cpu_device_manager.h>
#include <orteaf/internal/execution/execution.h>
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
  using DType = ::orteaf::internal::DType;
  using Execution = ::orteaf::internal::execution::Execution;

  /// @brief The execution backend for this storage type.
  static constexpr Execution kExecution = Execution::Cpu;

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
   *     .withDType(DType::F32)
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
      device_lease_ = ::orteaf::internal::execution::cpu::api::CpuExecutionApi::
          acquireDevice(handle);
      return *this;
    }

    Builder &withDType(DType dtype) {
      dtype_ = dtype;
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
      return CpuStorage(std::move(lease), std::move(layout_), dtype_, size_);
    }

  private:
    DeviceLease device_lease_{};
    DType dtype_{DType::F32};
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

  /// @brief Return the execution backend for this storage.
  constexpr Execution execution() const { return kExecution; }

  /// @brief Return the data type of elements in this storage.
  DType dtype() const { return dtype_; }

  /// @brief Return the size of the storage in bytes.
  std::size_t sizeInBytes() const { return size_; }

private:
  CpuStorage(BufferLease buffer_lease, Layout layout, DType dtype,
             std::size_t size)
      : buffer_lease_(std::move(buffer_lease)), layout_(std::move(layout)),
        dtype_(dtype), size_(size) {}

  BufferLease buffer_lease_;
  Layout layout_;
  DType dtype_{DType::F32};
  std::size_t size_{0};
};

} // namespace orteaf::internal::storage::cpu
