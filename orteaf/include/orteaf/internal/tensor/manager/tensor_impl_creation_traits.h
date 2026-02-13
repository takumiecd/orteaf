#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>

#include <orteaf/extension/tensor/dense_tensor_impl.h>
#include <orteaf/internal/diagnostics/error/error.h>
#include <orteaf/internal/execution/execution.h>
#include <orteaf/internal/execution_context/cpu/current_context.h>
#include <orteaf/internal/storage/registry/storage_types.h>
#include <orteaf/internal/storage/storage_lease.h>

#if ORTEAF_ENABLE_MPS
#include <orteaf/internal/execution/mps/platform/wrapper/mps_buffer.h>
#include <orteaf/internal/execution_context/mps/current_context.h>
#endif

#if ORTEAF_ENABLE_CUDA
#include <orteaf/internal/execution_context/cuda/current_context.h>
#endif

namespace orteaf::internal::tensor {

namespace detail {

template <typename>
struct DependentFalse : std::false_type {};

inline std::int64_t computeNumelOrThrow(std::span<const std::int64_t> shape) {
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

template <typename Impl> struct TensorImplCreationTraits {
  using CreateRequest = typename Impl::CreateRequest;

  static void validateCreateRequest(const CreateRequest &) {
    static_assert(detail::DependentFalse<Impl>::value,
                  "TensorImplCreationTraits specialization is required");
  }

  template <typename Context>
  static bool createPayload(Impl &, const CreateRequest &, const Context &) {
    static_assert(detail::DependentFalse<Impl>::value,
                  "TensorImplCreationTraits specialization is required");
    return false;
  }

  template <typename Context>
  static void destroyPayload(Impl &, const Context &) {
    static_assert(detail::DependentFalse<Impl>::value,
                  "TensorImplCreationTraits specialization is required");
  }
};

template <>
struct TensorImplCreationTraits<::orteaf::extension::tensor::DenseTensorImpl> {
  using Impl = ::orteaf::extension::tensor::DenseTensorImpl;
  using CreateRequest = typename Impl::CreateRequest;
  using Execution = ::orteaf::internal::execution::Execution;
  using StorageLease = ::orteaf::internal::storage::StorageLease;
  using StorageRegistry = ::orteaf::internal::storage::RegisteredStorages;

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

    const auto numel = detail::computeNumelOrThrow(std::span<const std::int64_t>(
        request.shape.data(), request.shape.size()));
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

} // namespace orteaf::internal::tensor
