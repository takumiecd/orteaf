#pragma once

#include <cstddef>

#include "orteaf/internal/base/lease/control_block/strong.h"
#include "orteaf/internal/base/manager/pool_manager.h"
#include "orteaf/internal/base/pool/fixed_slot_store.h"
#include "orteaf/internal/execution/cpu/cpu_handles.h"
#include "orteaf/internal/execution/cpu/resource/cpu_kernel_metadata.h"

namespace orteaf::internal::execution::cpu::manager {

// Forward declaration
class CpuKernelMetadataManager;
class CpuExecutionManager;

// =============================================================================
// Payload Pool Traits
// =============================================================================

struct KernelMetadataPayloadPoolTraits {
  using Payload =
      ::orteaf::internal::execution::cpu::resource::CpuKernelMetadata;
  using Handle = ::orteaf::internal::execution::cpu::CpuKernelMetadataHandle;
  using ExecuteFunc = Payload::ExecuteFunc;

  struct Request {
    ExecuteFunc execute;
  };

  struct Context {};

  static bool create(Payload &payload, const Request &request, const Context &);

  static void destroy(Payload &payload, const Request &, const Context &);
};

using KernelMetadataPayloadPool =
    ::orteaf::internal::base::pool::FixedSlotStore<
        KernelMetadataPayloadPoolTraits>;

// =============================================================================
// Control Block
// =============================================================================

struct KernelMetadataControlBlockTag {};

using KernelMetadataControlBlock = ::orteaf::internal::base::StrongControlBlock<
    ::orteaf::internal::execution::cpu::CpuKernelMetadataHandle,
    ::orteaf::internal::execution::cpu::resource::CpuKernelMetadata,
    KernelMetadataPayloadPool>;

// =============================================================================
// Manager Traits for PoolManager
// =============================================================================

struct CpuKernelMetadataManagerTraits {
  using PayloadPool = KernelMetadataPayloadPool;
  using ControlBlock = KernelMetadataControlBlock;
  struct ControlBlockTag {};
  using PayloadHandle =
      ::orteaf::internal::execution::cpu::CpuKernelMetadataHandle;
  static constexpr const char *Name = "CpuKernelMetadataManager";
};

// =============================================================================
// CpuKernelMetadataManager
// =============================================================================

/**
 * @brief Manager for CPU kernel metadata resources.
 */
class CpuKernelMetadataManager {
  using Core =
      ::orteaf::internal::base::PoolManager<CpuKernelMetadataManagerTraits>;

public:
  using KernelMetadataHandle =
      ::orteaf::internal::execution::cpu::CpuKernelMetadataHandle;
  using ExecuteFunc = ::orteaf::internal::execution::cpu::resource::
      CpuKernelMetadata::ExecuteFunc;

  using ControlBlock = Core::ControlBlock;
  using ControlBlockHandle = Core::ControlBlockHandle;
  using ControlBlockPool = Core::ControlBlockPool;

  using CpuKernelMetadataLease = Core::StrongLeaseType;

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

  CpuKernelMetadataManager() = default;
  CpuKernelMetadataManager(const CpuKernelMetadataManager &) = delete;
  CpuKernelMetadataManager &
  operator=(const CpuKernelMetadataManager &) = delete;
  CpuKernelMetadataManager(CpuKernelMetadataManager &&) = default;
  CpuKernelMetadataManager &operator=(CpuKernelMetadataManager &&) = default;
  ~CpuKernelMetadataManager() = default;

private:
  struct InternalConfig {
    Config public_config{};
  };

  void configure(const InternalConfig &config);

  friend class CpuExecutionManager;

public:
  void shutdown();

  CpuKernelMetadataLease acquire(ExecuteFunc execute);

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
  bool isAliveForTest(KernelMetadataHandle handle) const noexcept {
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
