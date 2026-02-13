#pragma once

/**
 * @file dense_tensor_impl.h
 * @brief Dense tensor implementation holding layout and storage.
 *
 * DenseTensorImpl combines a DenseTensorLayout (shape, strides, offset)
 * with a StorageLease (type-erased backend storage lease) to represent
 * dense tensor data.
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>

#include <orteaf/extension/tensor/layout/dense_tensor_layout.h>
#include <orteaf/internal/diagnostics/error/error.h>
#include <orteaf/internal/dtype/dtype.h>
#include <orteaf/internal/execution/execution.h>
#include <orteaf/internal/execution_context/cpu/current_context.h>
#include <orteaf/internal/kernel/core/kernel_arg_slots.h>
#include <orteaf/internal/kernel/storage/operand_id.h>
#include <orteaf/internal/storage/registry/storage_types.h>
#include <orteaf/internal/storage/storage_lease.h>
#include <orteaf/internal/tensor/traits/tensor_impl_traits.h>

#if ORTEAF_ENABLE_MPS
#include <orteaf/internal/execution/mps/platform/wrapper/mps_buffer.h>
#include <orteaf/internal/execution_context/mps/current_context.h>
#endif

#if ORTEAF_ENABLE_CUDA
#include <orteaf/internal/execution_context/cuda/current_context.h>
#endif

namespace orteaf::user::tensor {
class Tensor;
}

namespace orteaf::extension::tensor {

/**
 * @brief Dense tensor implementation.
 *
 * Holds a DenseTensorLayout describing the logical view (shape, strides,
 * offset) and a StorageLease providing access to the underlying data buffer.
 *
 * Multiple DenseTensorImpl instances can share the same storage (for views).
 * The layout's numel() represents the logical element count, while
 * storage's numel() represents the physical buffer capacity.
 *
 * Invariant: layout_.numel() <= storage_.lease().numel() (for valid views)
 */
class DenseTensorImpl {
public:
  using Layout = DenseTensorLayout;
  using Dims = Layout::Dims;
  using Dim = Layout::Dim;
  using StorageLease = ::orteaf::internal::storage::StorageLease;
  using StorageSlot = ::orteaf::internal::kernel::StorageSlot<
      ::orteaf::internal::kernel::Role::Data>;
  using DType = ::orteaf::internal::DType;
  using Execution = ::orteaf::internal::execution::Execution;

  struct CreateRequest {
    enum class PlacementPolicy : std::uint8_t {
      CurrentContext,
    };

    Dims shape{};
    DType dtype{DType::F32};
    Execution execution{Execution::Cpu};
    std::size_t alignment{0};
    PlacementPolicy placement_policy{PlacementPolicy::CurrentContext};
  };

  class Builder {
  public:
    Builder() = default;

    Builder &withShape(std::span<const Dim> shape);
    Builder &withDType(DType dtype) noexcept;
    Builder &withExecution(Execution execution) noexcept;
    Builder &withAlignment(std::size_t alignment) noexcept;
    ::orteaf::user::tensor::Tensor build() const;

  private:
    CreateRequest request_{};
    bool shape_set_{false};
    bool execution_set_{false};
  };

  /**
   * @brief Default constructor. Creates an uninitialized impl.
   */
  DenseTensorImpl() = default;

  static Builder builder();

  /**
   * @brief Construct from layout and storage lease.
   *
   * @param layout The tensor layout (shape, strides, offset).
   * @param storage The storage lease holding the data buffer.
   */
  DenseTensorImpl(Layout layout, StorageLease storage)
      : layout_(std::move(layout)), storage_(StorageSlot(std::move(storage))) {}

  DenseTensorImpl(Layout layout, StorageSlot storage)
      : layout_(std::move(layout)), storage_(std::move(storage)) {}

  DenseTensorImpl(const DenseTensorImpl &) = default;
  DenseTensorImpl &operator=(const DenseTensorImpl &) = default;
  DenseTensorImpl(DenseTensorImpl &&) = default;
  DenseTensorImpl &operator=(DenseTensorImpl &&) = default;
  ~DenseTensorImpl() = default;

  // ===== Accessors =====

  /// @brief Return the tensor layout.
  const Layout &layout() const noexcept { return layout_; }

  /// @brief Return the storage lease.
  const StorageLease &storageLease() const noexcept { return storage_.lease(); }

  /// @brief Return the storage slot.
  const StorageSlot &storageSlot() const noexcept { return storage_; }
  StorageSlot &storageSlot() noexcept { return storage_; }

  /// @brief Check if this impl is valid (has storage).
  bool valid() const noexcept { return static_cast<bool>(storage_.lease()); }

  // ===== Forwarding from StorageLease =====

  /// @brief Return the data type.
  DType dtype() const { return storage_.lease().dtype(); }

  /// @brief Return the execution backend.
  Execution execution() const { return storage_.lease().execution(); }

  /// @brief Return the storage size in bytes.
  std::size_t storageSizeInBytes() const { return storage_.lease().sizeInBytes(); }

  // ===== Forwarding from Layout =====

  /// @brief Return the tensor shape.
  const Dims &shape() const noexcept { return layout_.shape(); }

  /// @brief Return the tensor strides.
  const Dims &strides() const noexcept { return layout_.strides(); }

  /// @brief Return the element offset.
  Dim offset() const noexcept { return layout_.offset(); }

  /// @brief Return the number of elements (logical, based on shape).
  Dim numel() const noexcept { return layout_.numel(); }

  /// @brief Return the rank (number of dimensions).
  std::size_t rank() const noexcept { return layout_.rank(); }

  /// @brief Check if the layout is contiguous.
  bool isContiguous() const noexcept { return layout_.isContiguous(); }

  void bindAllArgs(::orteaf::internal::kernel::KernelArgs &args,
                   ::orteaf::internal::kernel::OperandId operand_id) const {
    storage_.bind(args, operand_id);
    layout_.bindParams(args, operand_id);
  }

private:
  Layout layout_{};
  StorageSlot storage_{};
};

} // namespace orteaf::extension::tensor

namespace orteaf::internal::tensor::registry {

namespace detail {

inline std::int64_t denseNumelOrThrow(std::span<const std::int64_t> shape) {
  std::int64_t numel = 1;
  for (const auto dim : shape) {
    if (dim < 0) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
          "Tensor create request contains a negative shape dimension");
    }
    if (dim == 0) {
      return 0;
    }

    constexpr auto kMaxNumel = std::numeric_limits<std::int64_t>::max();
    if (numel > kMaxNumel / dim) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
          "Tensor create request shape overflowed numel");
    }
    numel *= dim;
  }
  return numel;
}

} // namespace detail

template <>
struct TensorImplTraits<::orteaf::extension::tensor::DenseTensorImpl> {
  using Impl = ::orteaf::extension::tensor::DenseTensorImpl;
  using CreateRequest = typename Impl::CreateRequest;
  using Execution = ::orteaf::internal::execution::Execution;
  using StorageLease = ::orteaf::internal::storage::StorageLease;
  using StorageRegistry = ::orteaf::internal::storage::RegisteredStorages;

  static constexpr const char *name = "dense";

  static void validateCreateRequest(const CreateRequest &request) {
    if (request.placement_policy !=
        CreateRequest::PlacementPolicy::CurrentContext) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
          "Tensor create request placement policy is unsupported");
    }

    switch (request.execution) {
    case Execution::Cpu:
      break;
#if ORTEAF_ENABLE_MPS
    case Execution::Mps:
      break;
#endif
#if ORTEAF_ENABLE_CUDA
    case Execution::Cuda:
      break;
#endif
    default:
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::
              ExecutionUnavailable,
          "Tensor create request execution is unavailable");
    }
  }

  template <typename Context>
  static bool createPayload(Impl &payload, const CreateRequest &request,
                            const Context &context) {
    validateCreateRequest(request);

    if (context.storage_registry == nullptr) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
          "Tensor storage registry is not configured");
    }

    const auto numel = detail::denseNumelOrThrow(
        std::span<const std::int64_t>(request.shape.data(),
                                      request.shape.size()));
    auto storage_lease =
        createStorageLease(request, context.storage_registry, numel);
    if (!storage_lease.valid()) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
          "Tensor create request failed to acquire storage");
    }

    auto layout = Impl::Layout::contiguous(request.shape);
    payload = Impl(std::move(layout), std::move(storage_lease));
    return true;
  }

  template <typename Context>
  static void destroyPayload(Impl &payload, const Context &) {
    payload = Impl{};
  }

private:
  static StorageLease createStorageLease(const CreateRequest &request,
                                         StorageRegistry *storage_registry,
                                         std::int64_t numel) {
    switch (request.execution) {
    case Execution::Cpu:
      return createCpuStorage(request, storage_registry, numel);
#if ORTEAF_ENABLE_MPS
    case Execution::Mps:
      return createMpsStorage(request, storage_registry, numel);
#endif
#if ORTEAF_ENABLE_CUDA
    case Execution::Cuda:
      return createCudaStorage(request, storage_registry, numel);
#endif
    default:
      return {};
    }
  }

  static StorageLease createCpuStorage(const CreateRequest &request,
                                       StorageRegistry *storage_registry,
                                       std::int64_t numel) {
    using CpuStorage = ::orteaf::internal::storage::cpu::CpuStorage;
    using CpuStorageManager = ::orteaf::internal::storage::CpuStorageManager;

    auto device_lease = ::orteaf::internal::execution_context::cpu::currentDevice();
    if (!device_lease) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
          "CPU current context has no active device");
    }

    typename CpuStorageManager::Request storage_request{};
    storage_request.device = device_lease.payloadHandle();
    storage_request.dtype = request.dtype;
    storage_request.numel = static_cast<std::size_t>(numel);
    storage_request.alignment = request.alignment;
    storage_request.layout = typename CpuStorage::Layout{};

    auto lease =
        storage_registry->template get<CpuStorage>().acquire(storage_request);
    return StorageLease::erase(std::move(lease));
  }

#if ORTEAF_ENABLE_MPS
  static StorageLease createMpsStorage(const CreateRequest &request,
                                       StorageRegistry *storage_registry,
                                       std::int64_t numel) {
    using MpsStorage = ::orteaf::internal::storage::mps::MpsStorage;
    using MpsStorageManager = ::orteaf::internal::storage::MpsStorageManager;

    auto device_lease = ::orteaf::internal::execution_context::mps::currentDevice();
    if (!device_lease) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
          "MPS current context has no active device");
    }

    typename MpsStorageManager::Request storage_request{};
    storage_request.device = device_lease.payloadHandle();
    const auto numel_size =
        static_cast<std::size_t>(numel) * ::orteaf::internal::sizeOf(request.dtype);
    storage_request.heap_key = MpsStorage::HeapDescriptorKey::Sized(numel_size);
    storage_request.heap_key.storage_mode =
        ::orteaf::internal::execution::mps::platform::wrapper::
            kMPSStorageModePrivate;
    storage_request.dtype = request.dtype;
    storage_request.numel = static_cast<std::size_t>(numel);
    storage_request.alignment = request.alignment;
    storage_request.layout = typename MpsStorage::Layout{};

    auto lease =
        storage_registry->template get<MpsStorage>().acquire(storage_request);
    return StorageLease::erase(std::move(lease));
  }
#endif

#if ORTEAF_ENABLE_CUDA
  static StorageLease createCudaStorage(const CreateRequest &request,
                                        StorageRegistry *storage_registry,
                                        std::int64_t numel) {
    using CudaStorage = ::orteaf::internal::storage::cuda::CudaStorage;
    using CudaStorageManager = ::orteaf::internal::storage::CudaStorageManager;

    auto device_lease = ::orteaf::internal::execution_context::cuda::currentDevice();
    if (!device_lease) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
          "CUDA current context has no active device");
    }

    typename CudaStorageManager::Request storage_request{};
    storage_request.device = device_lease.payloadHandle();
    storage_request.dtype = request.dtype;
    storage_request.numel = static_cast<std::size_t>(numel);
    storage_request.alignment = request.alignment;
    storage_request.layout = typename CudaStorage::Layout{};

    auto lease =
        storage_registry->template get<CudaStorage>().acquire(storage_request);
    return StorageLease::erase(std::move(lease));
  }
#endif
};

} // namespace orteaf::internal::tensor::registry
