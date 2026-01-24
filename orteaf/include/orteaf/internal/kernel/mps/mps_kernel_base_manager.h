#pragma once

#if ORTEAF_ENABLE_MPS

#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>

#include "orteaf/internal/base/handle.h"
#include "orteaf/internal/base/lease/control_block/strong.h"
#include "orteaf/internal/base/manager/lease_lifetime_registry.h"
#include "orteaf/internal/base/manager/pool_manager.h"
#include "orteaf/internal/base/pool/fixed_slot_store.h"
#include "orteaf/internal/execution/mps/manager/mps_library_manager.h"
#include "orteaf/internal/execution/mps/mps_handles.h"
#include "orteaf/internal/kernel/mps/mps_kernel_base.h"

namespace orteaf::internal::kernel::mps {

// Handle type for KernelBase
struct MpsKernelBaseTag {};
using MpsKernelBaseHandle =
    ::orteaf::internal::base::Handle<MpsKernelBaseTag, uint32_t, void>;

// Key for identifying a kernel base (can be extended later)
struct KernelBaseKey {
  std::string identifier{};

  static KernelBaseKey Named(std::string name) {
    KernelBaseKey key{};
    key.identifier = std::move(name);
    return key;
  }

  friend bool operator==(const KernelBaseKey &lhs,
                         const KernelBaseKey &rhs) noexcept = default;
};

struct KernelBaseKeyHasher {
  std::size_t operator()(const KernelBaseKey &key) const noexcept {
    return std::hash<std::string>{}(key.identifier);
  }
};

/**
 * @brief Manager for MpsKernelBase instances using PoolManager pattern.
 *
 * This manager follows the same pattern as MpsLibraryManager - it uses
 * PoolManager to manage KernelBase resources with leasing and lifetime
 * tracking.
 *
 * @tparam N Maximum number of kernels per KernelBase
 */
template <std::size_t N> class MpsKernelBaseManager {
public:
  using KernelBase = MpsKernelBase<N>;

  // Payload pool traits
  struct PayloadPoolTraits {
    using Payload = KernelBase;
    using Handle = MpsKernelBaseHandle;

    struct Request {
      KernelBaseKey key{};
      typename KernelBase::KeyLiteral *keys{nullptr};
      std::size_t key_count{0};
    };

    struct Context {
      // Empty for now - KernelBase doesn't need external resources to create
      // (it only stores keys initially)
    };

    static bool create(Payload &payload, const Request &request,
                       const Context &) {
      if (request.keys == nullptr || request.key_count == 0) {
        return false;
      }
      // Create KernelBase with the provided keys
      std::initializer_list<typename KernelBase::KeyLiteral> key_list(
          request.keys, request.keys + request.key_count);
      payload = KernelBase(key_list);
      return true;
    }

    static void destroy(Payload &payload, const Request &, const Context &) {
      // KernelBase destructor handles cleanup automatically
      payload = KernelBase{};
    }
  };

  using PayloadPool =
      ::orteaf::internal::base::pool::FixedSlotStore<PayloadPoolTraits>;

  using ControlBlock =
      ::orteaf::internal::base::StrongControlBlock<MpsKernelBaseHandle,
                                                   KernelBase, PayloadPool>;

  struct ManagerTraits {
    using PayloadPool = PayloadPool;
    using ControlBlock = ControlBlock;
    struct ControlBlockTag {};
    using PayloadHandle = MpsKernelBaseHandle;
    static constexpr const char *Name = "MpsKernelBaseManager";
  };

  using Core = ::orteaf::internal::base::PoolManager<ManagerTraits>;
  using KernelBaseHandle = MpsKernelBaseHandle;
  using ControlBlockHandle = typename Core::ControlBlockHandle;
  using ControlBlockPool = typename Core::ControlBlockPool;
  using KernelBaseLease = typename Core::StrongLeaseType;
  using LifetimeRegistry =
      ::orteaf::internal::base::manager::LeaseLifetimeRegistry<
          KernelBaseHandle, KernelBaseLease>;

  MpsKernelBaseManager() = default;
  MpsKernelBaseManager(const MpsKernelBaseManager &) = delete;
  MpsKernelBaseManager &operator=(const MpsKernelBaseManager &) = delete;
  MpsKernelBaseManager(MpsKernelBaseManager &&) = default;
  MpsKernelBaseManager &operator=(MpsKernelBaseManager &&) = default;
  ~MpsKernelBaseManager() = default;

  struct Config {
    std::size_t control_block_capacity{0};
    std::size_t control_block_block_size{0};
    std::size_t control_block_growth_chunk_size{1};
    std::size_t payload_capacity{0};
    std::size_t payload_block_size{0};
    std::size_t payload_growth_chunk_size{1};
  };

  void configure(const Config &config) {
    typename Core::Config core_config{};
    core_config.control_block_capacity = config.control_block_capacity;
    core_config.control_block_block_size = config.control_block_block_size;
    core_config.control_block_growth_chunk_size =
        config.control_block_growth_chunk_size;
    core_config.payload_capacity = config.payload_capacity;
    core_config.payload_block_size = config.payload_block_size;
    core_config.payload_growth_chunk_size = config.payload_growth_chunk_size;
    core_.configure(core_config);
    configured_ = true;
  }

  void shutdown() {
    lifetime_.clear();
    key_to_index_.clear();
    core_.shutdown();
    configured_ = false;
  }

  /**
   * @brief Acquire a KernelBase by key.
   *
   * If the key already exists, returns cached lease. Otherwise creates
   * a new KernelBase with the provided kernel function keys.
   */
  KernelBaseLease acquire(const KernelBaseKey &key,
                          typename KernelBase::KeyLiteral *keys,
                          std::size_t key_count) {
    if (!configured_) {
      return KernelBaseLease{};
    }

    // Check cache
    if (auto it = key_to_index_.find(key); it != key_to_index_.end()) {
      auto handle = KernelBaseHandle{it->second};
      auto cached = lifetime_.get(handle);
      if (cached) {
        return cached;
      }
      // Cached lease expired, acquire again
      auto lease = core_.acquireStrongLease(handle);
      lifetime_.set(lease);
      return lease;
    }

    // Create new
    typename PayloadPoolTraits::Request request{};
    request.key = key;
    request.keys = keys;
    request.key_count = key_count;

    typename PayloadPoolTraits::Context context{};

    auto handle = core_.reserveUncreatedPayloadOrGrow();
    if (!core_.emplacePayload(handle, request, context)) {
      core_.releasePayload(handle);
      return KernelBaseLease{};
    }

    key_to_index_.emplace(key, handle.index);
    auto lease = core_.acquireStrongLease(handle);
    lifetime_.set(lease);
    return lease;
  }

  /**
   * @brief Acquire an existing KernelBase by handle.
   */
  KernelBaseLease acquire(KernelBaseHandle handle) {
    if (!configured_ || !core_.isAlive(handle)) {
      return KernelBaseLease{};
    }

    auto cached = lifetime_.get(handle);
    if (cached) {
      return cached;
    }

    auto lease = core_.acquireStrongLease(handle);
    lifetime_.set(lease);
    return lease;
  }

  void release(KernelBaseLease &lease) noexcept { lease.release(); }

#if ORTEAF_ENABLE_TEST
  bool isConfiguredForTest() const noexcept { return configured_; }
  std::size_t payloadPoolSizeForTest() const noexcept {
    return core_.payloadPoolSizeForTest();
  }
  std::size_t payloadPoolCapacityForTest() const noexcept {
    return core_.payloadPoolCapacityForTest();
  }
  std::size_t controlBlockPoolSizeForTest() const noexcept {
    return core_.controlBlockPoolSizeForTest();
  }
  std::size_t controlBlockPoolCapacityForTest() const noexcept {
    return core_.controlBlockPoolCapacityForTest();
  }
  bool isAliveForTest(KernelBaseHandle handle) const noexcept {
    return core_.isAlive(handle);
  }
#endif

private:
  Core core_{};
  LifetimeRegistry lifetime_{};
  std::unordered_map<KernelBaseKey, std::size_t, KernelBaseKeyHasher>
      key_to_index_{};
  bool configured_{false};
};

} // namespace orteaf::internal::kernel::mps

#endif // ORTEAF_ENABLE_MPS
