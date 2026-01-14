#pragma once

#include <cstddef>
#include <orteaf/internal/base/lease/control_block/strong.h>
#include <orteaf/internal/base/manager/pool_manager.h>
#include <orteaf/internal/base/pool/slot_pool.h>
#include <orteaf/internal/storage/manager/storage_request.h>
#include <orteaf/internal/storage/storage.h>
#include <orteaf/internal/storage/storage_handles.h>

namespace orteaf::internal::storage::manager {

namespace detail {

struct StoragePayloadPoolTraits {
  using Payload = ::orteaf::internal::storage::Storage;
  using Handle = ::orteaf::internal::storage::StorageHandle;
  using Request = StorageRequest;
  struct Context {};

  static constexpr bool destroy_on_release = true;
  static constexpr const char *ManagerName = "Storage manager";

  static void validateRequestOrThrow(const Request &request);
  static bool create(Payload &payload, const Request &request,
                     const Context &context);
  static void destroy(Payload &payload, const Request &request,
                      const Context &context);
};

} // namespace detail

class StorageManager {
public:
  using PayloadPool = ::orteaf::internal::base::pool::SlotPool<
      detail::StoragePayloadPoolTraits>;
  using ControlBlock = ::orteaf::internal::base::StrongControlBlock<
      ::orteaf::internal::storage::StorageHandle,
      ::orteaf::internal::storage::Storage, PayloadPool>;

  struct Traits {
    using PayloadPool = StorageManager::PayloadPool;
    using ControlBlock = StorageManager::ControlBlock;
    struct ControlBlockTag {};
    using PayloadHandle = ::orteaf::internal::storage::StorageHandle;
    static constexpr const char *Name =
        detail::StoragePayloadPoolTraits::ManagerName;
  };

  using Core = ::orteaf::internal::base::PoolManager<Traits>;
  using StorageLease = Core::StrongLeaseType;
  using Request = detail::StoragePayloadPoolTraits::Request;
  using Context = detail::StoragePayloadPoolTraits::Context;

  struct Config {
    std::size_t control_block_capacity{0};
    std::size_t control_block_block_size{0};
    std::size_t control_block_growth_chunk_size{1};
    std::size_t payload_capacity{0};
    std::size_t payload_block_size{0};
    std::size_t payload_growth_chunk_size{1};
  };

  StorageManager() = default;
  StorageManager(const StorageManager &) = delete;
  StorageManager &operator=(const StorageManager &) = delete;
  StorageManager(StorageManager &&) = default;
  StorageManager &operator=(StorageManager &&) = default;
  ~StorageManager() = default;

  void configure(const Config &config);
  StorageLease acquire(const Request &request);
  void shutdown();
  bool isConfigured() const noexcept;

private:
  Core core_{};
};

} // namespace orteaf::internal::storage::manager
