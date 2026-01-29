#pragma once

#if ORTEAF_ENABLE_MPS

#include <cstddef>
#include <utility>

#include "orteaf/internal/base/heap_vector.h"
#include "orteaf/internal/base/lease/control_block/strong.h"
#include "orteaf/internal/base/manager/pool_manager.h"
#include "orteaf/internal/base/pool/slot_pool.h"
#include "orteaf/internal/execution/mps/manager/mps_compute_pipeline_state_manager.h"
#include "orteaf/internal/execution/mps/manager/mps_library_manager.h"
#include "orteaf/internal/execution/mps/mps_handles.h"
#include "orteaf/internal/execution/mps/platform/mps_slow_ops.h"

namespace orteaf::internal::kernel::mps {

// Forward declaration
class MpsKernelBaseManager;

// =============================================================================
// Payload: holds pipeline state leases for kernel execution
// =============================================================================

/**
 * @brief Payload for MPS kernel base resource.
 *
 * Contains pipeline state leases for one or more kernel functions.
 * Each kernel can have multiple library/function pairs.
 */
struct MpsKernelBasePayload {
  using PipelineLease = ::orteaf::internal::execution::mps::manager::
      MpsComputePipelineStateManager::PipelineLease;

  /// Pipeline state leases for each kernel function
  ::orteaf::internal::base::HeapVector<PipelineLease> pipelines;

  /// Get the number of kernel functions
  [[nodiscard]] std::size_t kernelCount() const noexcept {
    return pipelines.size();
  }

  /// Get a pipeline lease by index
  [[nodiscard]] PipelineLease *getPipeline(std::size_t index) noexcept {
    if (index >= pipelines.size()) {
      return nullptr;
    }
    return &pipelines[index];
  }

  /// Get a const pipeline lease by index
  [[nodiscard]] const PipelineLease *
  getPipeline(std::size_t index) const noexcept {
    if (index >= pipelines.size()) {
      return nullptr;
    }
    return &pipelines[index];
  }
};

// =============================================================================
// Payload Pool Traits
// =============================================================================

struct KernelBasePayloadPoolTraits {
  using Payload = MpsKernelBasePayload;
  using Handle = ::orteaf::internal::execution::mps::MpsKernelBaseHandle;
  using LibraryKey = ::orteaf::internal::execution::mps::manager::LibraryKey;
  using FunctionKey = ::orteaf::internal::execution::mps::manager::FunctionKey;
  using Key = std::pair<LibraryKey, FunctionKey>;

  struct Request {
    ::orteaf::internal::base::HeapVector<Key> keys;
  };

  struct Context {
    ::orteaf::internal::execution::mps::manager::MpsLibraryManager
        *library_manager{nullptr};
    ::orteaf::internal::execution::mps::platform::MpsSlowOps *ops{nullptr};
  };

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
    MpsKernelBasePayload, KernelBasePayloadPool>;

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
  using LibraryManager =
      ::orteaf::internal::execution::mps::manager::MpsLibraryManager;
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

private:
  friend KernelBaseLease;

public:
  struct Config {
    // PoolManager settings
    std::size_t control_block_capacity{0};
    std::size_t control_block_block_size{0};
    std::size_t control_block_growth_chunk_size{1};
    std::size_t payload_capacity{0};
    std::size_t payload_block_size{0};
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
    LibraryManager *library_manager{nullptr};
    SlowOps *ops{nullptr};
  };

  void configure(const InternalConfig &config);

public:
  void shutdown();

  /**
   * @brief Acquire a kernel base lease.
   *
   * Creates a new kernel base with the specified library/function keys.
   * The returned lease holds strong references to pipeline states.
   *
   * @param keys Library/function key pairs for kernel functions
   * @return Strong lease to kernel base resource
   */
  KernelBaseLease acquire(const ::orteaf::internal::base::HeapVector<Key> &keys);

  /**
   * @brief Release a kernel base lease.
   *
   * Decrements reference count and returns resource to pool when count reaches
   * zero.
   *
   * @param lease Lease to release
   */
  void release(KernelBaseLease &lease) noexcept { lease.release(); }

#if ORTEAF_ENABLE_TEST
  void configureForTest(const Config &config, LibraryManager *library_manager,
                        SlowOps *ops) {
    InternalConfig internal{};
    internal.public_config = config;
    internal.library_manager = library_manager;
    internal.ops = ops;
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
  KernelBasePayloadPoolTraits::Context makePayloadContext() const noexcept;

  LibraryManager *library_manager_{nullptr};
  SlowOps *ops_{nullptr};
  Core core_{};
};

} // namespace orteaf::internal::kernel::mps

#endif // ORTEAF_ENABLE_MPS
