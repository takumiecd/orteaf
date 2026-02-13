#pragma once

#include <cstddef>
#include <limits>
#include <string>

#include <orteaf/internal/diagnostics/error/error.h>
#include <orteaf/internal/dtype/dtype.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_heap.h>
#include <orteaf/internal/storage/registry/storage_types.h>
#include <orteaf/internal/tensor/api/tensor_api.h>

namespace orteaf::extension::kernel::mps::copy_detail {

namespace error = ::orteaf::internal::diagnostics::error;

inline ::orteaf::internal::storage::MpsStorageLease
acquireSharedMpsStaging(::orteaf::internal::DType dtype, std::size_t numel,
                        const char *op_name) {
  using MpsStorage = ::orteaf::internal::storage::mps::MpsStorage;
  using MpsStorageManager = ::orteaf::internal::storage::MpsStorageManager;

  const auto elem_size = ::orteaf::internal::sizeOf(dtype);
  if (numel > std::numeric_limits<std::size_t>::max() / elem_size) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      std::string(op_name) + " staging size overflow");
  }

  typename MpsStorageManager::Request request{};
  request.device = {};
  request.heap_key = MpsStorage::HeapDescriptorKey::Sized(numel * elem_size);
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
                          " failed to acquire shared staging storage");
  }
  return lease;
}

} // namespace orteaf::extension::kernel::mps::copy_detail
