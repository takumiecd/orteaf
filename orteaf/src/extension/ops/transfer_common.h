#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>

#include <orteaf/internal/base/checked_int.h>
#include <orteaf/internal/diagnostics/error/error.h>
#include <orteaf/internal/dtype/dtype.h>
#include <orteaf/internal/execution/execution.h>
#include <orteaf/internal/storage/registry/storage_types.h>
#include <orteaf/internal/tensor/api/tensor_api.h>

#if ORTEAF_ENABLE_MPS
#include <orteaf/internal/execution/mps/platform/wrapper/mps_buffer.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_command_buffer.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_heap.h>
#include <orteaf/internal/execution_context/mps/current_context.h>
#endif

#include "op_common.h"

namespace orteaf::extension::ops::detail::transfer {

namespace error = ::orteaf::internal::diagnostics::error;
using DenseTensorImpl = ::orteaf::extension::tensor::DenseTensorImpl;
using DType = ::orteaf::internal::DType;
using Execution = ::orteaf::internal::execution::Execution;

inline constexpr std::uint8_t kTransferShapeInlineCapacity = 8;

struct LayoutStats {
  std::size_t numel{};
  bool has_zero{};
  bool contiguous{};
  std::int64_t min_index{};
  std::int64_t max_index{};
};

inline LayoutStats analyzeLayout(std::span<const std::int64_t> shape,
                                 std::span<const std::int64_t> strides,
                                 std::int64_t offset,
                                 const char *op_name,
                                 const char *tensor_name) {
  if (shape.size() != strides.size()) {
    error::throwError(
        error::OrteafErrc::InvalidParameter,
        std::string(op_name) + ": " + tensor_name +
            " has mismatched shape/strides");
  }

  LayoutStats stats{};
  stats.numel = 1;
  stats.has_zero = false;
  stats.contiguous = true;
  stats.min_index = offset;
  stats.max_index = offset;

  if (shape.empty()) {
    return stats;
  }

  std::size_t expected_stride = 1;
  for (std::size_t i = shape.size(); i-- > 0;) {
    const auto dim = shape[i];
    if (dim < 0) {
      error::throwError(
          error::OrteafErrc::InvalidParameter,
          std::string(op_name) + ": " + tensor_name +
              " has negative shape dimension");
    }
    if (dim == 0) {
      stats.numel = 0;
      stats.has_zero = true;
      return stats;
    }

    const auto dim_size = static_cast<std::size_t>(dim);
    if (stats.numel > std::numeric_limits<std::size_t>::max() / dim_size) {
      error::throwError(
          error::OrteafErrc::InvalidParameter,
          std::string(op_name) + ": " + tensor_name + " shape is too large");
    }
    stats.numel *= dim_size;

    if (strides[i] != static_cast<std::int64_t>(expected_stride)) {
      stats.contiguous = false;
    }
    if (expected_stride > std::numeric_limits<std::size_t>::max() / dim_size) {
      error::throwError(
          error::OrteafErrc::InvalidParameter,
          std::string(op_name) + ": " + tensor_name + " shape is too large");
    }
    expected_stride *= dim_size;
  }

  std::int64_t min_index = offset;
  std::int64_t max_index = offset;
  for (std::size_t i = 0; i < shape.size(); ++i) {
    const auto dim = shape[i];
    if (dim <= 0) {
      continue;
    }
    const auto stride = strides[i];
    std::int64_t span = 0;
    if (::orteaf::internal::base::mulOverflowI64(stride, dim - 1, span)) {
      error::throwError(error::OrteafErrc::InvalidParameter,
                        std::string(op_name) + ": " + tensor_name +
                            " index range overflow");
    }
    if (stride >= 0) {
      if (::orteaf::internal::base::addOverflowI64(max_index, span,
                                                   max_index)) {
        error::throwError(error::OrteafErrc::InvalidParameter,
                          std::string(op_name) + ": " + tensor_name +
                              " index range overflow");
      }
    } else {
      if (::orteaf::internal::base::addOverflowI64(min_index, span,
                                                   min_index)) {
        error::throwError(error::OrteafErrc::InvalidParameter,
                          std::string(op_name) + ": " + tensor_name +
                              " index range overflow");
      }
    }
  }

  stats.min_index = min_index;
  stats.max_index = max_index;
  return stats;
}

inline LayoutStats validateViewBounds(const DenseTensorImpl *impl,
                                      const char *op_name,
                                      const char *tensor_name) {
  const auto storage_numel_raw = impl->storageLease().numel();
  if (storage_numel_raw >
      static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      std::string(op_name) + ": " + tensor_name +
                          " storage is too large");
  }
  const auto storage_numel = static_cast<std::int64_t>(storage_numel_raw);
  const auto offset = impl->offset();
  if (offset < 0) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      std::string(op_name) + ": " + tensor_name +
                          " has negative offset");
  }

  const auto &shape = impl->shape();
  const auto &strides = impl->strides();
  const auto stats = analyzeLayout(
      std::span<const std::int64_t>(shape.data(), shape.size()),
      std::span<const std::int64_t>(strides.data(), strides.size()), offset,
      op_name, tensor_name);

  if (!stats.has_zero &&
      (stats.min_index < 0 || stats.max_index < 0 ||
       stats.max_index >= storage_numel)) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      std::string(op_name) + ": " + tensor_name +
                          " view exceeds storage bounds");
  }

  return stats;
}

inline void requireExecution(const DenseTensorImpl *impl, Execution expected,
                             const char *op_name, const char *tensor_name) {
  if (impl->execution() != expected) {
    error::throwError(error::OrteafErrc::ExecutionUnavailable,
                      std::string(op_name) + ": " + tensor_name +
                          " has unsupported execution");
  }
}

inline void requireMatchingShapeAndDType(const DenseTensorImpl *output,
                                         const DenseTensorImpl *input,
                                         const char *op_name) {
  if (output->dtype() != input->dtype()) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      std::string(op_name) +
                          ": input/output dtype must match");
  }
  if (output->rank() != input->rank()) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      std::string(op_name) +
                          ": input/output rank must match");
  }
  const auto &out_shape = output->shape();
  const auto &in_shape = input->shape();
  for (std::size_t i = 0; i < out_shape.size(); ++i) {
    if (out_shape[i] != in_shape[i]) {
      error::throwError(error::OrteafErrc::InvalidParameter,
                        std::string(op_name) +
                            ": input/output shape must match");
    }
  }
}

inline void ensureRankSupported(const DenseTensorImpl *impl,
                                const char *op_name) {
  if (impl->rank() > kTransferShapeInlineCapacity) {
    error::throwError(error::OrteafErrc::Unsupported,
                      std::string(op_name) + ": rank > 8 is unsupported on MPS");
  }
}

inline const std::byte *requireCpuConstData(const DenseTensorImpl *impl,
                                            const char *op_name,
                                            const char *tensor_name) {
  auto *cpu_lease = impl->storageLease().tryAs<::orteaf::internal::storage::CpuStorageLease>();
  if (!cpu_lease || !(*cpu_lease)) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      std::string(op_name) + ": " + tensor_name +
                          " requires CPU storage");
  }
  auto *cpu_storage = cpu_lease->operator->();
  if (cpu_storage == nullptr || cpu_storage->buffer() == nullptr) {
    error::throwError(error::OrteafErrc::InvalidState,
                      std::string(op_name) + ": " + tensor_name +
                          " CPU buffer is unavailable");
  }
  return static_cast<const std::byte *>(cpu_storage->buffer());
}

inline std::byte *requireCpuMutableData(const DenseTensorImpl *impl,
                                        const char *op_name,
                                        const char *tensor_name) {
  return const_cast<std::byte *>(
      requireCpuConstData(impl, op_name, tensor_name));
}

inline std::int64_t physicalIndexForLinear(
    std::uint64_t linear, std::span<const std::int64_t> shape,
    std::span<const std::int64_t> strides, std::int64_t offset) {
  std::int64_t physical = offset;
  std::uint64_t remaining = linear;
  for (std::size_t i = shape.size(); i-- > 0;) {
    const auto dim = static_cast<std::uint64_t>(shape[i]);
    const auto coord = dim == 0 ? 0 : (remaining % dim);
    remaining = dim == 0 ? 0 : (remaining / dim);
    physical += static_cast<std::int64_t>(coord) * strides[i];
  }
  return physical;
}

inline void packStridedToContiguous(std::byte *dst_contiguous,
                                    const std::byte *src_base,
                                    std::size_t elem_size,
                                    std::span<const std::int64_t> shape,
                                    std::span<const std::int64_t> strides,
                                    std::int64_t offset,
                                    std::size_t numel) {
  for (std::size_t linear = 0; linear < numel; ++linear) {
    const auto src_index = physicalIndexForLinear(
        static_cast<std::uint64_t>(linear), shape, strides, offset);
    const auto src_byte_index =
        static_cast<std::size_t>(src_index) * elem_size;
    std::copy_n(src_base + src_byte_index, elem_size,
                dst_contiguous + linear * elem_size);
  }
}

inline void unpackContiguousToStrided(std::byte *dst_base,
                                      const std::byte *src_contiguous,
                                      std::size_t elem_size,
                                      std::span<const std::int64_t> shape,
                                      std::span<const std::int64_t> strides,
                                      std::int64_t offset,
                                      std::size_t numel) {
  for (std::size_t linear = 0; linear < numel; ++linear) {
    const auto dst_index = physicalIndexForLinear(
        static_cast<std::uint64_t>(linear), shape, strides, offset);
    const auto dst_byte_index =
        static_cast<std::size_t>(dst_index) * elem_size;
    std::copy_n(src_contiguous + linear * elem_size, elem_size,
                dst_base + dst_byte_index);
  }
}

#if ORTEAF_ENABLE_MPS

inline ::orteaf::internal::storage::MpsStorageLease
acquireSharedMpsStaging(DType dtype, std::size_t numel, const char *op_name) {
  using MpsStorage = ::orteaf::internal::storage::mps::MpsStorage;
  using MpsStorageManager = ::orteaf::internal::storage::MpsStorageManager;

  const auto elem_size = ::orteaf::internal::sizeOf(dtype);
  if (numel > std::numeric_limits<std::size_t>::max() / elem_size) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      std::string(op_name) + ": staging size overflow");
  }
  const auto bytes = numel * elem_size;

  typename MpsStorageManager::Request request{};
  request.device = ::orteaf::internal::execution::mps::MpsDeviceHandle{0};
  request.heap_key = MpsStorage::HeapDescriptorKey::Sized(bytes);
  request.heap_key.storage_mode =
      ::orteaf::internal::execution::mps::platform::wrapper::
          kMPSStorageModeShared;
  request.dtype = dtype;
  request.numel = numel;
  request.alignment = 0;
  request.layout = typename MpsStorage::Layout{};

  auto lease = ::orteaf::internal::tensor::api::TensorApi::storage()
                   .template get<MpsStorage>()
                   .acquire(request);
  if (!lease || lease->buffer() == nullptr) {
    error::throwError(error::OrteafErrc::InvalidState,
                      std::string(op_name) +
                          ": failed to acquire shared MPS staging");
  }
  return lease;
}

inline std::byte *requireSharedMpsMutableData(
    const ::orteaf::internal::storage::MpsStorageLease &lease,
    const char *op_name) {
  auto *storage = lease.operator->();
  if (storage == nullptr || storage->buffer() == nullptr) {
    error::throwError(error::OrteafErrc::InvalidState,
                      std::string(op_name) +
                          ": staging MPS buffer is unavailable");
  }
  auto *ptr = ::orteaf::internal::execution::mps::platform::wrapper::
      getBufferContents(storage->buffer());
  if (ptr == nullptr) {
    error::throwError(error::OrteafErrc::InvalidState,
                      std::string(op_name) +
                          ": shared MPS staging has no CPU mapping");
  }
  return static_cast<std::byte *>(ptr);
}

inline const std::byte *requireSharedMpsConstData(
    const ::orteaf::internal::storage::MpsStorageLease &lease,
    const char *op_name) {
  auto *storage = lease.operator->();
  if (storage == nullptr || storage->buffer() == nullptr) {
    error::throwError(error::OrteafErrc::InvalidState,
                      std::string(op_name) +
                          ": staging MPS buffer is unavailable");
  }
  auto *ptr = ::orteaf::internal::execution::mps::platform::wrapper::
      getBufferContentsConst(storage->buffer());
  if (ptr == nullptr) {
    error::throwError(error::OrteafErrc::InvalidState,
                      std::string(op_name) +
                          ": shared MPS staging has no CPU mapping");
  }
  return static_cast<const std::byte *>(ptr);
}

inline void syncCurrentMpsQueue(const char *op_name) {
  namespace mps_context = ::orteaf::internal::execution_context::mps;
  namespace mps_wrapper = ::orteaf::internal::execution::mps::platform::wrapper;

  auto queue_lease = mps_context::currentCommandQueue();
  auto *resource = queue_lease.operator->();
  if (resource == nullptr || resource->queue() == nullptr) {
    error::throwError(error::OrteafErrc::InvalidState,
                      std::string(op_name) + ": MPS command queue unavailable");
  }

  auto command_buffer = mps_wrapper::createCommandBuffer(resource->queue());
  if (command_buffer == nullptr) {
    error::throwError(error::OrteafErrc::InvalidState,
                      std::string(op_name) + ": failed to create command buffer");
  }
  mps_wrapper::commit(command_buffer);
  mps_wrapper::waitUntilCompleted(command_buffer);
  mps_wrapper::destroyCommandBuffer(command_buffer);
}

#endif // ORTEAF_ENABLE_MPS

} // namespace orteaf::extension::ops::detail::transfer

