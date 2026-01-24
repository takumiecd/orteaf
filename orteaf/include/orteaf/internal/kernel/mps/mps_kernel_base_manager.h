#pragma once

#if ORTEAF_ENABLE_MPS

#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "orteaf/internal/base/handle.h"
#include "orteaf/internal/base/lease/control_block/strong.h"
#include "orteaf/internal/base/manager/lease_lifetime_registry.h"
#include "orteaf/internal/base/manager/pool_manager.h"
#include "orteaf/internal/base/pool/fixed_slot_store.h"
#include "orteaf/internal/execution/mps/mps_handles.h"
#include "orteaf/internal/kernel/mps/mps_kernel_base.h"

namespace orteaf::internal::kernel::mps {

// Handle type for KernelBase
struct MpsKernelBaseTag {};
using MpsKernelBaseHandle =
    ::orteaf::internal::base::Handle<MpsKernelBaseTag, uint32_t, void>;

// Key for identifying a kernel base using library/function names
struct KernelBaseKey {
  std::vector<std::pair<std::string, std::string>> kernels{};

  static KernelBaseKey Create(
      std::initializer_list<std::pair<const char *, const char *>> kernel_pairs) {
    KernelBaseKey key{};
    for (const auto &[lib, func] : kernel_pairs) {
      key.kernels.emplace_back(lib, func);
    }
    return key;
  }

  friend bool operator==(const KernelBaseKey &lhs,
                         const KernelBaseKey &rhs) noexcept = default;
};

struct KernelBaseKeyHasher {
  std::size_t operator()(const KernelBaseKey &key) const noexcept {
    std::size_t hash = 0;
    for (const auto &[lib, func] : key.kernels) {
      hash ^= std::hash<std::string>{}(lib) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
      hash ^= std::hash<std::string>{}(func) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
  }
};

/**
 * @brief Manager for MpsKernelBase instances using PoolManager pattern.
 *
 * This manager follows the same pattern as MpsLibraryManager - it uses
 * PoolManager to manage KernelBase resources with leasing and lifetime
 * tracking.
 */
class MpsKernelBaseManager {
public:
  using KernelBase = MpsKernelBase;

  // Payload pool traits
  struct PayloadPoolTraits {
    using Payload = KernelBase;
    using Handle = MpsKernelBaseHandle;

    struct Request {
      KernelBaseKey key{};
    };

    struct Context {
      // Empty for now - KernelBase doesn't need external resources to create
      // (it only stores keys initially)
    };

    static bool create(Payload &payload, const Request &request,
                       const Context &context);

    static void destroy(Payload &payload, const Request &request,
                        const Context &context);
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

  void configure(const Config &config);

  void shutdown();

  /**
   * @brief Acquire a KernelBase by key.
   *
   * If the key already exists, returns cached lease. Otherwise creates
   * a new KernelBase with the kernel functions specified in the key.
   */
  KernelBaseLease acquire(const KernelBaseKey &key);

  /**
   * @brief Acquire an existing KernelBase by handle.
   */
  KernelBaseLease acquire(KernelBaseHandle handle);

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
