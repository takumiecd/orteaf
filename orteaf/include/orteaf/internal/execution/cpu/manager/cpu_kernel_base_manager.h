#pragma once

#include <cstddef>

#include "orteaf/internal/base/lease/control_block/strong.h"
#include "orteaf/internal/base/manager/pool_manager.h"
#include "orteaf/internal/base/pool/slot_pool.h"
#include "orteaf/internal/execution/cpu/cpu_handles.h"
#include "orteaf/internal/execution/cpu/resource/cpu_kernel_base.h"

namespace orteaf::internal::execution::cpu::manager {

// Forward declaration
class CpuKernelBaseManager;
class CpuExecutionManager;

//  =============================================================================
// Payload Pool Traits
// =============================================================================

struct KernelBasePayloadPoolTraits {
  using Payload = ::orteaf::internal::execution::cpu::resource::CpuKernelBase;
  using Handle = ::orteaf::internal::execution::cpu::CpuKernelBaseHandle;
  using ExecuteFunc = Payload::ExecuteFunc;

  struct Request {
    ExecuteFunc execute{nullptr};
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
    ::orteaf::internal::execution::cpu::CpuKernelBaseHandle,
    ::orteaf::internal::execution::cpu::resource::CpuKernelBase,
    KernelBasePayloadPool>;

// =============================================================================
// Manager Traits for PoolManager
// =============================================================================

struct CpuKernelBaseManagerTraits {
  using PayloadPool = KernelBasePayloadPool;
  using ControlBlock = KernelBaseControlBlock;
  struct ControlBlockTag {};
  using PayloadHandle = ::orteaf::internal::execution::cpu::CpuKernelBaseHandle;
  static constexpr const char *Name = "CpuKernelBaseManager";
};

// =============================================================================
// CpuKernelBaseManager
// =============================================================================

/**
 * @brief Manager for CPU kernel base resources.
 *
 * Manages kernel base instances via PoolManager pattern.
 * Each kernel base holds an ExecuteFunc for kernel execution.
 *
 * Design pattern: Same as MpsKernelBaseManager, CpuBufferManager, etc.
 * - PoolManager for lifecycle management
 * - StrongLease for reference counting
 * - Payload holds actual resources (ExecuteFunc)
 */
class CpuKernelBaseManager {
  using Core =
      ::orteaf::internal::base::PoolManager<CpuKernelBaseManagerTraits>;

public:
  using KernelBaseHandle =
      ::orteaf::internal::execution::cpu::CpuKernelBaseHandle;
  using ExecuteFunc =
      ::orteaf::internal::execution::cpu::resource::CpuKernelBase::ExecuteFunc;

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

  CpuKernelBaseManager() = default;
  CpuKernelBaseManager(const CpuKernelBaseManager &) = delete;
  CpuKernelBaseManager &operator=(const CpuKernelBaseManager &) = delete;
  CpuKernelBaseManager(CpuKernelBaseManager &&) = default;
  CpuKernelBaseManager &operator=(CpuKernelBaseManager &&) = default;
  ~CpuKernelBaseManager() = default;

private:
  struct InternalConfig {
    Config public_config{};
  };

  void configure(const InternalConfig &config);

  friend class CpuExecutionManager;

public:
  void shutdown();

  /**
   * @brief Acquire a kernel base lease.
   *
   * Creates a new kernel base with the specified ExecuteFunc.
   *
   * @param execute ExecuteFunc for kernel execution
   * @return Strong lease to kernel base resource
   */
  KernelBaseLease acquire(ExecuteFunc execute);

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

} // namespace orteaf::internal::execution::cpu::manager
