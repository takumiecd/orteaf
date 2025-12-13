#pragma once

#if ORTEAF_ENABLE_MPS

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>

#include <orteaf/internal/base/handle.h>
#include <orteaf/internal/base/lease.h>
#include <orteaf/internal/runtime/allocator/resource/mps/mps_resource.h>
#include <orteaf/internal/runtime/base/shared_cache_manager.h>
#include <orteaf/internal/runtime/mps/manager/mps_buffer_manager.h>
#include <orteaf/internal/runtime/mps/platform/mps_slow_ops.h>
#include <orteaf/internal/runtime/mps/platform/wrapper/mps_heap.h>

namespace orteaf::internal::runtime::mps::manager {

struct HeapDescriptorKey {
  std::size_t size_bytes{0};
  ::orteaf::internal::runtime::mps::platform::wrapper::MPSResourceOptions_t
      resource_options{::orteaf::internal::runtime::mps::platform::wrapper::
                           kMPSDefaultResourceOptions};
  ::orteaf::internal::runtime::mps::platform::wrapper::MPSStorageMode_t
      storage_mode{::orteaf::internal::runtime::mps::platform::wrapper::
                       kMPSStorageModeShared};
  ::orteaf::internal::runtime::mps::platform::wrapper::MPSCPUCacheMode_t
      cpu_cache_mode{::orteaf::internal::runtime::mps::platform::wrapper::
                         kMPSCPUCacheModeDefaultCache};
  ::orteaf::internal::runtime::mps::platform::wrapper::MPSHazardTrackingMode_t
      hazard_tracking_mode{::orteaf::internal::runtime::mps::platform::wrapper::
                               kMPSHazardTrackingModeDefault};
  ::orteaf::internal::runtime::mps::platform::wrapper::MPSHeapType_t heap_type{
      ::orteaf::internal::runtime::mps::platform::wrapper::
          kMPSHeapTypeAutomatic};

  static HeapDescriptorKey Sized(std::size_t size) {
    HeapDescriptorKey key{};
    key.size_bytes = size;
    return key;
  }

  friend bool operator==(const HeapDescriptorKey &lhs,
                         const HeapDescriptorKey &rhs) noexcept = default;
};

struct HeapDescriptorKeyHasher {
  std::size_t operator()(const HeapDescriptorKey &key) const noexcept {
    std::size_t seed = key.size_bytes;
    constexpr std::size_t kMagic = 0x9e3779b97f4a7c15ull;
    auto mix = [&](std::size_t value) {
      seed ^= value + kMagic + (seed << 6) + (seed >> 2);
    };
    mix(static_cast<std::size_t>(key.resource_options));
    mix(static_cast<std::size_t>(key.storage_mode));
    mix(static_cast<std::size_t>(key.cpu_cache_mode));
    mix(static_cast<std::size_t>(key.hazard_tracking_mode));
    mix(static_cast<std::size_t>(key.heap_type));
    return seed;
  }
};

// Resource struct: holds heap + buffer_manager (heap-allocated for pointer
// stability)
struct MpsHeapResource {
  ::orteaf::internal::runtime::mps::platform::wrapper::MPSHeap_t heap{nullptr};
  std::unique_ptr<::orteaf::internal::runtime::mps::manager::MpsBufferManagerT<
      ::orteaf::internal::runtime::allocator::resource::mps::MpsResource>>
      buffer_manager;
};

// Use SharedCacheState template
using MpsHeapManagerState =
    ::orteaf::internal::runtime::base::SharedCacheState<MpsHeapResource>;

struct MpsHeapManagerTraits {
  using OpsType = ::orteaf::internal::runtime::mps::platform::MpsSlowOps;
  using StateType = MpsHeapManagerState;
  static constexpr const char *Name = "MPS heap manager";
};

class MpsHeapManager
    : public ::orteaf::internal::runtime::base::SharedCacheManager<
          MpsHeapManager, MpsHeapManagerTraits> {
public:
  using Base = ::orteaf::internal::runtime::base::SharedCacheManager<
      MpsHeapManager, MpsHeapManagerTraits>;
  using SlowOps = ::orteaf::internal::runtime::mps::platform::MpsSlowOps;
  using DeviceType =
      ::orteaf::internal::runtime::mps::platform::wrapper::MPSDevice_t;
  using HeapHandle = ::orteaf::internal::base::HeapHandle;
  using HeapLease = ::orteaf::internal::base::Lease<
      HeapHandle,
      ::orteaf::internal::runtime::mps::platform::wrapper::MPSHeap_t,
      MpsHeapManager>;
  using BufferManager =
      ::orteaf::internal::runtime::mps::manager::MpsBufferManagerT<
          ::orteaf::internal::runtime::allocator::resource::mps::MpsResource>;

  MpsHeapManager() = default;
  MpsHeapManager(const MpsHeapManager &) = delete;
  MpsHeapManager &operator=(const MpsHeapManager &) = delete;
  MpsHeapManager(MpsHeapManager &&) = default;
  MpsHeapManager &operator=(MpsHeapManager &&) = default;
  ~MpsHeapManager() = default;

  void initialize(DeviceType device,
                  ::orteaf::internal::base::DeviceHandle device_handle,
                  MpsLibraryManager *library_manager, SlowOps *ops,
                  std::size_t capacity);
  void shutdown();

  HeapLease acquire(const HeapDescriptorKey &key);
  void release(HeapLease &lease) noexcept;

private:
  void validateKey(const HeapDescriptorKey &key) const;

  ::orteaf::internal::runtime::mps::platform::wrapper::MPSHeap_t
  createHeap(const HeapDescriptorKey &key);

  std::unordered_map<HeapDescriptorKey, std::size_t, HeapDescriptorKeyHasher>
      key_to_index_{};
  DeviceType device_{nullptr};
  ::orteaf::internal::base::DeviceHandle device_handle_{};
  MpsLibraryManager *library_manager_{nullptr};
};

} // namespace orteaf::internal::runtime::mps::manager

#endif // ORTEAF_ENABLE_MPS
