#pragma once

#include <cstddef>
#include <variant>

#include <orteaf/internal/storage/cpu/cpu_storage.h>
#if ORTEAF_ENABLE_MPS
#include <orteaf/internal/storage/mps/mps_storage.h>
#endif // ORTEAF_ENABLE_MPS

namespace orteaf::internal::storage::manager {

struct CpuStorageRequest {
  using DeviceHandle = ::orteaf::internal::storage::cpu::CpuStorage::DeviceHandle;
  using Layout = ::orteaf::internal::storage::cpu::CpuStorage::Layout;

  DeviceHandle device{DeviceHandle::invalid()};
  std::size_t size{0};
  std::size_t alignment{0};
  Layout layout{};
};

#if ORTEAF_ENABLE_MPS
struct MpsStorageRequest {
  using DeviceHandle = ::orteaf::internal::storage::mps::MpsStorage::DeviceHandle;
  using HeapDescriptorKey =
      ::orteaf::internal::storage::mps::MpsStorage::HeapDescriptorKey;
  using Layout = ::orteaf::internal::storage::mps::MpsStorage::Layout;

  DeviceHandle device{DeviceHandle::invalid()};
  HeapDescriptorKey heap_key{};
  std::size_t size{0};
  std::size_t alignment{0};
  Layout layout{};
};
#endif // ORTEAF_ENABLE_MPS

using StorageRequest = std::variant<CpuStorageRequest
#if ORTEAF_ENABLE_MPS
                                   , MpsStorageRequest
#endif // ORTEAF_ENABLE_MPS
                                   >;

} // namespace orteaf::internal::storage::manager
