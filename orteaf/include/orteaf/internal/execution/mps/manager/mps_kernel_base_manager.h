#pragma once

#if ORTEAF_ENABLE_MPS

#include <cstddef>
#include <utility>

#include "orteaf/internal/base/heap_vector.h"
#include "orteaf/internal/base/lease/control_block/strong.h"
#include "orteaf/internal/base/manager/pool_manager.h"
#include "orteaf/internal/base/pool/slot_pool.h"
#include "orteaf/internal/execution/mps/manager/mps_device_manager.h"
#include "orteaf/internal/execution/mps/mps_handles.h"
#include "orteaf/internal/execution/mps/resource/mps_kernel_base.h"
#include "orteaf/internal/execution/mps/platform/mps_slow_ops.h"

namespace orteaf::internal::execution::mps::manager {

// Forward declaration
class MpsKernelBaseManager;
class MpsExecutionManager;

// =============================================================================
// Payload (Kernel Base Resource)
// =============================================================================

// =============================================================================
// Payload Pool Traits
// =============================================================================

struct KernelBasePayloadPoolTraits {
  using Payload = ::orteaf::internal::execution::mps::resource::MpsKernelBase;
  using Handle = ::orteaf::internal::execution::mps::MpsKernelBaseHandle;
  using DeviceLease = ::orteaf::internal::execution::mps::manager::
      MpsDeviceManager::DeviceLease;
  using LibraryKey = ::orteaf::internal::execution::mps::manager::LibraryKey;
  using FunctionKey = ::orteaf::internal::execution::mps::manager::FunctionKey;
  using Key = std::pair<LibraryKey, FunctionKey>;

  struct Request {
    ::orteaf::internal::base::HeapVector<Key> keys;
  };

  struct Context {};

  static constexpr bool destroy_on_release = true;

  static bool create(Payload &payload, const Request &request,
                     const Context &context);

  static void destroy(Payload &payload, const Request &request,
                      const Context &context);
};

using KernelBasePayloadPool =
    ::orteaf::internal::base::pool::SlotPool<KernelBasePayloadPoolTraits>;

// =============================================================================
// Control Block
// =============================================================================

struct KernelBaseControlBlockTag {};

using KernelBaseControlBlock = ::orteaf::internal::base::StrongControlBlock<
    ::orteaf::internal::execution::mps::MpsKernelBaseHandle,
    ::orteaf::internal::execution::mps::resource::MpsKernelBase,
    KernelBasePayloadPool>;

// =============================================================================
// Manager Traits for PoolManager
// =============================================================================

struct MpsKernelBaseManagerTraits {
  using PayloadPool = KernelBasePayloadPool;
  using ControlBlock = KernelBaseControlBlock;
  struct ControlBlockTag {};
  using PayloadHandle = ::orteaf::internal::execution::mps::MpsKernelBaseHandle;
  static constexpr const char *Name = "MpsKernelBaseManager";
};

// =============================================================================
// MpsKernelBaseManager
// =============================================================================

/**
 * @brief Manager for MPS kernel base resources.
 *
 * Manages kernel base instances via PoolManager pattern.
 * Each kernel base holds pipeline state leases for kernel execution.
 *
 * Design pattern: Same as MpsEventManager, MpsDeviceManager, etc.
 * - PoolManager for lifecycle management
 * - StrongLease for reference counting
 * - Payload holds actual resources (pipeline leases)
 */
class MpsKernelBaseManager {
  using Core = ::orteaf::internal::base::PoolManager<MpsKernelBaseManagerTraits>;

public:
  using SlowOps = ::orteaf::internal::execution::mps::platform::MpsSlowOps;
  using DeviceLease = ::orteaf::internal::execution::mps::manager::
      MpsDeviceManager::DeviceLease;
  using KernelBaseHandle =
      ::orteaf::internal::execution::mps::MpsKernelBaseHandle;
  using LibraryKey = ::orteaf::internal::execution::mps::manager::LibraryKey;
  using FunctionKey = ::orteaf::internal::execution::mps::manager::FunctionKey;
  using Key = std::pair<LibraryKey, FunctionKey>;

  using ControlBlock = Core::ControlBlock;
  using ControlBlockHandle = Core::ControlBlockHandle;
  using ControlBlockPool = Core::ControlBlockPool;

  /// Strong lease type for kernel base resources
  using KernelBaseLease = Core::StrongLeaseType;

public:
  struct Config {
    // PoolManager settings
    std::size_t control_block_capacity{0};
    std::size_t control_block_block_size{1};
    std::size_t control_block_growth_chunk_size{1};
    std::size_t payload_capacity{0};
    std::size_t payload_block_size{1};
    std::size_t payload_growth_chunk_size{1};
  };

  MpsKernelBaseManager() = default;
  MpsKernelBaseManager(const MpsKernelBaseManager &) = delete;
  MpsKernelBaseManager &operator=(const MpsKernelBaseManager &) = delete;
  MpsKernelBaseManager(MpsKernelBaseManager &&) = default;
  MpsKernelBaseManager &operator=(MpsKernelBaseManager &&) = default;
  ~MpsKernelBaseManager() = default;

private:
  struct InternalConfig {
    Config public_config{};
  };

  void configure(const InternalConfig &config);

  friend class MpsExecutionManager;

public:
  void shutdown();

  /**
   * @brief Acquire a kernel base lease.
   *
   * Creates a new kernel base with the specified library/function keys.
   * The returned lease holds the key list but does not configure pipelines.
   *
   * @param keys Library/function key pairs for kernel functions
   * @return Strong lease to kernel base resource
   */
  KernelBaseLease acquire(const ::orteaf::internal::base::HeapVector<Key> &keys);

#if ORTEAF_ENABLE_TEST
  void configureForTest(const Config &config) {
    InternalConfig internal{};
    internal.public_config = config;
    configure(internal);
  }

  bool isConfiguredForTest() const noexcept { return core_.isConfigured(); }

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
  std::size_t payloadGrowthChunkSizeForTest() const noexcept {
    return core_.payloadGrowthChunkSize();
  }
  std::size_t controlBlockGrowthChunkSizeForTest() const noexcept {
    return core_.controlBlockGrowthChunkSize();
  }
#endif

private:
  Core core_{};
};

} // namespace orteaf::internal::execution::mps::manager

#endif // ORTEAF_ENABLE_MPS
