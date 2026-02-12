#pragma once

#if ORTEAF_ENABLE_CUDA

#include <cstddef>

#include "orteaf/internal/base/heap_vector.h"
#include "orteaf/internal/base/lease/control_block/strong.h"
#include "orteaf/internal/base/manager/pool_manager.h"
#include "orteaf/internal/base/pool/slot_pool.h"
#include "orteaf/internal/execution/cuda/cuda_handles.h"
#include "orteaf/internal/execution/cuda/resource/cuda_kernel_metadata.h"

namespace orteaf::internal::execution::cuda::manager {

class CudaKernelMetadataManager;
class CudaExecutionManager;

struct KernelMetadataPayloadPoolTraits {
  using Payload =
      ::orteaf::internal::execution::cuda::resource::CudaKernelMetadata;
  using Handle = ::orteaf::internal::execution::cuda::CudaKernelMetadataHandle;
  using Key = ::orteaf::internal::execution::cuda::resource::
      CudaKernelMetadata::Key;
  using ExecuteFunc =
      ::orteaf::internal::execution::cuda::resource::CudaKernelMetadata::ExecuteFunc;

  struct Request {
    ::orteaf::internal::base::HeapVector<Key> keys;
    ExecuteFunc execute{nullptr};
  };

  struct Context {};

  static constexpr bool destroy_on_release = true;

  static bool create(Payload &payload, const Request &request, const Context &);
  static void destroy(Payload &payload, const Request &, const Context &);
};

using KernelMetadataPayloadPool = ::orteaf::internal::base::pool::SlotPool<
    KernelMetadataPayloadPoolTraits>;

struct KernelMetadataControlBlockTag {};

using KernelMetadataControlBlock = ::orteaf::internal::base::StrongControlBlock<
    ::orteaf::internal::execution::cuda::CudaKernelMetadataHandle,
    ::orteaf::internal::execution::cuda::resource::CudaKernelMetadata,
    KernelMetadataPayloadPool>;

struct CudaKernelMetadataManagerTraits {
  using PayloadPool = KernelMetadataPayloadPool;
  using ControlBlock = KernelMetadataControlBlock;
  struct ControlBlockTag {};
  using PayloadHandle =
      ::orteaf::internal::execution::cuda::CudaKernelMetadataHandle;
  static constexpr const char *Name = "CudaKernelMetadataManager";
};

class CudaKernelMetadataManager {
  using Core =
      ::orteaf::internal::base::PoolManager<CudaKernelMetadataManagerTraits>;

public:
  using KernelMetadataHandle =
      ::orteaf::internal::execution::cuda::CudaKernelMetadataHandle;
  using Key = ::orteaf::internal::execution::cuda::resource::
      CudaKernelMetadata::Key;
  using ExecuteFunc =
      ::orteaf::internal::execution::cuda::resource::CudaKernelMetadata::ExecuteFunc;

  using ControlBlock = Core::ControlBlock;
  using ControlBlockHandle = Core::ControlBlockHandle;
  using ControlBlockPool = Core::ControlBlockPool;
  using CudaKernelMetadataLease = Core::StrongLeaseType;

public:
  struct Config {
    std::size_t control_block_capacity{0};
    std::size_t control_block_block_size{1};
    std::size_t control_block_growth_chunk_size{1};
    std::size_t payload_capacity{0};
    std::size_t payload_block_size{1};
    std::size_t payload_growth_chunk_size{1};
  };

  CudaKernelMetadataManager() = default;
  CudaKernelMetadataManager(const CudaKernelMetadataManager &) = delete;
  CudaKernelMetadataManager &
  operator=(const CudaKernelMetadataManager &) = delete;
  CudaKernelMetadataManager(CudaKernelMetadataManager &&) = default;
  CudaKernelMetadataManager &operator=(CudaKernelMetadataManager &&) = default;
  ~CudaKernelMetadataManager() = default;

private:
  struct InternalConfig {
    Config public_config{};
  };

  void configure(const InternalConfig &config);

  friend class CudaExecutionManager;

public:
  void shutdown();

  CudaKernelMetadataLease
  acquire(const ::orteaf::internal::base::HeapVector<Key> &keys);

  CudaKernelMetadataLease acquire(
      const ::orteaf::internal::execution::cuda::resource::CudaKernelMetadata
          &metadata);

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

} // namespace orteaf::internal::execution::cuda::manager

#endif // ORTEAF_ENABLE_CUDA
