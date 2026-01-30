#pragma once

/**
 * @file typed_storage_manager.h
 * @brief Generic template for Storage management.
 *
 * This template provides automatic pool management for any Storage type
 * that satisfies the StorageConcept. Similar to TensorImplManager,
 * this allows auto-generation of managers for different storage backends.
 */

#include <cstddef>
#include <type_traits>
#include <utility>

#include <orteaf/internal/base/handle.h>
#include <orteaf/internal/base/lease/control_block/strong.h>
#include <orteaf/internal/base/manager/pool_manager.h>
#include <orteaf/internal/base/pool/slot_pool.h>
#include <orteaf/internal/diagnostics/error/error_macros.h>
#include <orteaf/internal/dtype/dtype.h>
#include <orteaf/internal/execution/execution.h>
#include <orteaf/internal/storage/concepts/storage_concepts.h>
#if ORTEAF_ENABLE_MPS
#include <orteaf/internal/storage/mps/mps_storage.h>
#endif

namespace orteaf::internal::storage::manager {

// =============================================================================
// Handle for Storage
// =============================================================================

template <typename Storage> struct StorageTag {};

template <typename Storage>
using StorageHandle =
    ::orteaf::internal::base::Handle<StorageTag<Storage>, uint32_t, uint32_t>;

// =============================================================================
// Pool Traits for Storage
// =============================================================================

namespace detail {

/// @brief Request for creating a new Storage
template <typename Storage> struct TypedStorageRequest {
  using DeviceHandle = typename Storage::DeviceHandle;
  using Layout = typename Storage::Layout;
  using DType = typename Storage::DType;

  DeviceHandle device{DeviceHandle::invalid()};
  DType dtype{DType::F32};
  std::size_t numel{0};
  std::size_t alignment{0};
  Layout layout{};
};

#if ORTEAF_ENABLE_MPS
template <>
struct TypedStorageRequest<::orteaf::internal::storage::mps::MpsStorage> {
  using DeviceHandle =
      ::orteaf::internal::storage::mps::MpsStorage::DeviceHandle;
  using HeapHandle =
      ::orteaf::internal::storage::mps::MpsStorage::HeapHandle;
  using HeapDescriptorKey =
      ::orteaf::internal::storage::mps::MpsStorage::HeapDescriptorKey;
  using Layout = ::orteaf::internal::storage::mps::MpsStorage::Layout;
  using DType = ::orteaf::internal::storage::mps::MpsStorage::DType;

  DeviceHandle device{DeviceHandle::invalid()};
  HeapHandle heap_handle{HeapHandle::invalid()};
  HeapDescriptorKey heap_key{};
  DType dtype{DType::F32};
  std::size_t numel{0};
  std::size_t alignment{0};
  Layout layout{};
};
#endif

/// @brief Context for pool operations
template <typename Storage> struct TypedStorageContext {};

/// @brief Pool traits for generic Storage
template <typename Storage> struct TypedStoragePoolTraits {
  using Payload = Storage;
  using Handle = StorageHandle<Storage>;
  using Request = TypedStorageRequest<Storage>;
  using Context = TypedStorageContext<Storage>;

  static constexpr bool destroy_on_release = true;
  static constexpr const char *ManagerName = "TypedStorage manager";

  static void validateRequestOrThrow(const Request &request) {
  #if ORTEAF_ENABLE_MPS
    if constexpr (std::is_same_v<
                      Storage, ::orteaf::internal::storage::mps::MpsStorage>) {
      if (!request.heap_handle.isValid() &&
          request.heap_key.size_bytes == 0) {
        ORTEAF_THROW(
            InvalidArgument,
            "MpsStorage request requires a valid heap handle or heap key");
      }
    } else
  #endif
    if constexpr (requires { request.device.isValid(); }) {
      if (!request.device.isValid()) {
        ORTEAF_THROW(InvalidArgument,
                     "Storage request requires a valid device handle");
      }
    }
    if (request.numel == 0) {
      ORTEAF_THROW(InvalidArgument, "Storage request requires non-zero numel");
    }
  }

  static bool create(Payload &payload, const Request &request,
                     const Context & /*context*/) {
    if constexpr (requires { Payload::builder(); }) {
      auto builder = Payload::builder();
      if constexpr (requires { builder.withHeapHandle(request.heap_handle); }) {
        if (request.heap_handle.isValid()) {
          builder.withHeapHandle(request.heap_handle);
        }
      }
      if constexpr (requires { builder.withHeapKey(request.heap_key); }) {
        if (request.heap_key.size_bytes != 0) {
          builder.withHeapKey(request.heap_key);
        }
      }
      if constexpr (requires { builder.withDeviceHandle(request.device); }) {
        if (request.device.isValid()) {
          builder.withDeviceHandle(request.device);
        }
      }
      if constexpr (requires { builder.withDType(request.dtype); }) {
        builder.withDType(request.dtype);
      }
      if constexpr (requires { builder.withNumElements(request.numel); }) {
        builder.withNumElements(request.numel);
      }
      if constexpr (requires { builder.withAlignment(request.alignment); }) {
        builder.withAlignment(request.alignment);
      }
      if constexpr (requires { builder.withLayout(request.layout); }) {
        builder.withLayout(request.layout);
      }
      payload = builder.build();
      return true;
    }
    return false;
  }

  static void destroy(Payload &payload, const Request & /*request*/,
                      const Context & /*context*/) {
    payload = Payload{};
  }
};

} // namespace detail

// =============================================================================
// Generic TypedStorageManager
// =============================================================================

/**
 * @brief Generic manager for Storage types.
 *
 * Provides automatic pool management for any Storage type.
 *
 * @tparam Storage The Storage type (must satisfy StorageConcept)
 */
template <typename Storage>
  requires concepts::StorageConcept<Storage>
class TypedStorageManager {
public:
  using PayloadPool = ::orteaf::internal::base::pool::SlotPool<
      detail::TypedStoragePoolTraits<Storage>>;
  using ControlBlock =
      ::orteaf::internal::base::StrongControlBlock<StorageHandle<Storage>,
                                                   Storage, PayloadPool>;

  struct Traits {
    using PayloadPool = TypedStorageManager::PayloadPool;
    using ControlBlock = TypedStorageManager::ControlBlock;
    struct ControlBlockTag {};
    using PayloadHandle = StorageHandle<Storage>;
    static constexpr const char *Name =
        detail::TypedStoragePoolTraits<Storage>::ManagerName;
  };

  using Core = ::orteaf::internal::base::PoolManager<Traits>;
  using StorageLease = typename Core::StrongLeaseType;
  using Layout = typename Storage::Layout;
  using DType = typename Storage::DType;
  using Request = detail::TypedStorageRequest<Storage>;
  using Execution = ::orteaf::internal::execution::Execution;

  static constexpr Execution kExecution = Storage::kExecution;

  struct Config {
    std::size_t control_block_capacity{64};
    std::size_t control_block_block_size{16};
    std::size_t control_block_growth_chunk_size{1};
    std::size_t payload_capacity{64};
    std::size_t payload_block_size{16};
    std::size_t payload_growth_chunk_size{1};
  };

  TypedStorageManager() = default;
  TypedStorageManager(const TypedStorageManager &) = delete;
  TypedStorageManager &operator=(const TypedStorageManager &) = delete;
  TypedStorageManager(TypedStorageManager &&) = default;
  TypedStorageManager &operator=(TypedStorageManager &&) = default;
  ~TypedStorageManager() = default;

  void configure(const Config &config);
  StorageLease acquire(const Request &request);
  void shutdown();
  bool isConfigured() const noexcept;

private:
  Core core_{};
};

} // namespace orteaf::internal::storage::manager
