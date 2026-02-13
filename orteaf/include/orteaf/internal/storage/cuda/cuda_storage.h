#pragma once

#if ORTEAF_ENABLE_CUDA

#include <utility>

#include <orteaf/internal/diagnostics/error/error.h>
#include <orteaf/internal/dtype/dtype.h>
#include <orteaf/internal/execution/cuda/api/cuda_execution_api.h>
#include <orteaf/internal/execution/cuda/manager/cuda_context_manager.h>
#include <orteaf/internal/execution/cuda/manager/cuda_device_manager.h>
#include <orteaf/internal/execution/cuda/resource/cuda_buffer_view.h>
#include <orteaf/internal/execution/execution.h>
#include <orteaf/internal/storage/cuda/cuda_storage_layout.h>

namespace orteaf::internal::storage::cuda {

class CudaStorage {
public:
  using BufferManager =
      ::orteaf::internal::execution::cuda::manager::CudaBufferManager;
  using DeviceManager =
      ::orteaf::internal::execution::cuda::manager::CudaDeviceManager;
  using ContextManager =
      ::orteaf::internal::execution::cuda::manager::CudaContextManager;
  using DeviceHandle = DeviceManager::DeviceHandle;
  using DeviceLease = DeviceManager::DeviceLease;
  using ContextLease = ContextManager::ContextLease;
  using BufferLease = BufferManager::BufferLease;
  using BufferView =
      ::orteaf::internal::execution::cuda::resource::CudaBufferView;
  using Layout = ::orteaf::internal::storage::cuda::CudaStorageLayout;
  using DType = ::orteaf::internal::DType;
  using Execution = ::orteaf::internal::execution::Execution;

  static constexpr Execution kExecution = Execution::Cuda;

  class Builder {
  public:
    Builder() = default;

    Builder &withDeviceLease(const DeviceLease &lease) {
      device_lease_ = lease;
      return *this;
    }

    Builder &withDeviceHandle(DeviceHandle handle) {
      device_lease_ =
          ::orteaf::internal::execution::cuda::api::CudaExecutionApi::
              acquireDevice(handle);
      return *this;
    }

    Builder &withContextLease(const ContextLease &lease) {
      context_lease_ = lease;
      return *this;
    }

    Builder &withDType(DType dtype) {
      dtype_ = dtype;
      return *this;
    }

    Builder &withNumElements(std::size_t numel) {
      numel_ = numel;
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

    CudaStorage build() {
      if (!context_lease_) {
        if (!device_lease_) {
          ::orteaf::internal::diagnostics::error::throwError(
              ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
              "CudaStorage requires a valid device or context lease");
        }
        auto *device_resource = device_lease_.operator->();
        if (device_resource == nullptr) {
          ::orteaf::internal::diagnostics::error::throwError(
              ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
              "CudaStorage device lease has no payload");
        }
        context_lease_ = device_resource->context_manager.acquirePrimary();
      }

      if (!context_lease_) {
        ::orteaf::internal::diagnostics::error::throwError(
            ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
            "CudaStorage requires a valid context lease");
      }

      auto *context_resource = context_lease_.operator->();
      if (context_resource == nullptr) {
        ::orteaf::internal::diagnostics::error::throwError(
            ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
            "CudaStorage context lease has no payload");
      }

      const std::size_t size_in_bytes =
          numel_ * ::orteaf::internal::sizeOf(dtype_);
      BufferLease lease =
          context_resource->buffer_manager.acquire(size_in_bytes, alignment_);
      return CudaStorage(std::move(lease), std::move(layout_), dtype_, numel_);
    }

  private:
    DeviceLease device_lease_{};
    ContextLease context_lease_{};
    DType dtype_{DType::F32};
    std::size_t numel_{0};
    std::size_t alignment_{0};
    Layout layout_{};
  };

  static Builder builder() { return Builder{}; }

  CudaStorage() = default;

  CudaStorage(const CudaStorage &) = default;
  CudaStorage &operator=(const CudaStorage &) = default;
  CudaStorage(CudaStorage &&) = default;
  CudaStorage &operator=(CudaStorage &&) = default;
  ~CudaStorage() = default;

  constexpr Execution execution() const { return kExecution; }

  DType dtype() const { return dtype_; }

  std::size_t numel() const { return numel_; }

  std::size_t sizeInBytes() const {
    return numel_ * ::orteaf::internal::sizeOf(dtype_);
  }

  const BufferLease &bufferLease() const { return buffer_lease_; }

  BufferLease &bufferLease() { return buffer_lease_; }

  BufferView bufferView() const {
    if (!buffer_lease_) {
      return BufferView{};
    }
    auto *buffer_payload = buffer_lease_.operator->();
    if (buffer_payload == nullptr || !buffer_payload->valid()) {
      return BufferView{};
    }
    return buffer_payload->view;
  }

  typename BufferView::pointer buffer() const {
    auto view = bufferView();
    return view ? view.raw() : 0;
  }

  std::size_t bufferOffset() const {
    auto view = bufferView();
    return view ? view.offset() : 0;
  }

private:
  CudaStorage(BufferLease buffer_lease, Layout layout, DType dtype,
              std::size_t numel)
      : buffer_lease_(std::move(buffer_lease)), layout_(std::move(layout)),
        dtype_(dtype), numel_(numel) {}

  BufferLease buffer_lease_{};
  Layout layout_{};
  DType dtype_{DType::F32};
  std::size_t numel_{0};
};

} // namespace orteaf::internal::storage::cuda

#endif // ORTEAF_ENABLE_CUDA
