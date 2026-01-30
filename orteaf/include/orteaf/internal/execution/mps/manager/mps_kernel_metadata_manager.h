#pragma once

#if ORTEAF_ENABLE_MPS

#include <cstddef>
#include <utility>

#include "orteaf/internal/base/heap_vector.h"
#include "orteaf/internal/base/lease/control_block/strong.h"
#include "orteaf/internal/base/manager/pool_manager.h"
#include "orteaf/internal/base/pool/fixed_slot_store.h"
#include "orteaf/internal/execution/mps/mps_handles.h"
#include "orteaf/internal/execution/mps/resource/mps_kernel_metadata.h"

namespace orteaf::internal::execution::mps::manager {

// Forward declaration
class MpsKernelMetadataManager;
class MpsExecutionManager;

// =============================================================================
// Payload Pool Traits
// =============================================================================

struct KernelMetadataPayloadPoolTraits {
  using Payload =
      ::orteaf::internal::execution::mps::resource::MpsKernelMetadata;
  using Handle = ::orteaf::internal::execution::mps::MpsKernelMetadataHandle;
  using LibraryKey = ::orteaf::internal::execution::mps::manager::LibraryKey;
  using FunctionKey = ::orteaf::internal::execution::mps::manager::FunctionKey;
  using Key = std::pair<LibraryKey, FunctionKey>;
  using ExecuteFunc = Payload::ExecuteFunc;

  struct Request {
    ::orteaf::internal::base::HeapVector<Key> keys;
    ExecuteFunc execute{nullptr};
  };

  struct Context {};

  static bool create(Payload &payload, const Request &request,
                     const Context &);

  static void destroy(Payload &payload, const Request &, const Context &);
};

using KernelMetadataPayloadPool =
    ::orteaf::internal::base::pool::FixedSlotStore<KernelMetadataPayloadPoolTraits>;

// =============================================================================
// Control Block
// =============================================================================

struct KernelMetadataControlBlockTag {};

using KernelMetadataControlBlock = ::orteaf::internal::base::StrongControlBlock<
    ::orteaf::internal::execution::mps::MpsKernelMetadataHandle,
    ::orteaf::internal::execution::mps::resource::MpsKernelMetadata,
    KernelMetadataPayloadPool>;

// =============================================================================
// Manager Traits for PoolManager
// =============================================================================

struct MpsKernelMetadataManagerTraits {
  using PayloadPool = KernelMetadataPayloadPool;
  using ControlBlock = KernelMetadataControlBlock;
  struct ControlBlockTag {};
  using PayloadHandle =
      ::orteaf::internal::execution::mps::MpsKernelMetadataHandle;
  static constexpr const char *Name = "MpsKernelMetadataManager";
};

// =============================================================================
// MpsKernelMetadataManager
// =============================================================================

/**
 * @brief Manager for MPS kernel metadata resources.
 */
class MpsKernelMetadataManager {
  using Core =
      ::orteaf::internal::base::PoolManager<MpsKernelMetadataManagerTraits>;

public:
  using KernelMetadataHandle =
      ::orteaf::internal::execution::mps::MpsKernelMetadataHandle;
  using LibraryKey = ::orteaf::internal::execution::mps::manager::LibraryKey;
  using FunctionKey = ::orteaf::internal::execution::mps::manager::FunctionKey;
  using Key = std::pair<LibraryKey, FunctionKey>;
  using ExecuteFunc =
      ::orteaf::internal::execution::mps::resource::MpsKernelMetadata::
          ExecuteFunc;

  using ControlBlock = Core::ControlBlock;
  using ControlBlockHandle = Core::ControlBlockHandle;
  using ControlBlockPool = Core::ControlBlockPool;

  using KernelMetadataLease = Core::StrongLeaseType;

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

  MpsKernelMetadataManager() = default;
  MpsKernelMetadataManager(const MpsKernelMetadataManager &) = delete;
  MpsKernelMetadataManager &
  operator=(const MpsKernelMetadataManager &) = delete;
  MpsKernelMetadataManager(MpsKernelMetadataManager &&) = default;
  MpsKernelMetadataManager &operator=(MpsKernelMetadataManager &&) = default;
  ~MpsKernelMetadataManager() = default;

private:
  struct InternalConfig {
    Config public_config{};
  };

  void configure(const InternalConfig &config);

  friend class MpsExecutionManager;

public:
  void shutdown();

  KernelMetadataLease acquire(
      const ::orteaf::internal::base::HeapVector<Key> &keys,
      ExecuteFunc execute);

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

} // namespace orteaf::internal::execution::mps::manager

#endif // ORTEAF_ENABLE_MPS
