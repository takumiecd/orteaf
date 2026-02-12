#pragma once

#if ORTEAF_ENABLE_CUDA

#include <cstddef>

#include "orteaf/internal/base/heap_vector.h"
#include "orteaf/internal/base/lease/control_block/strong.h"
#include "orteaf/internal/base/manager/pool_manager.h"
#include "orteaf/internal/base/pool/slot_pool.h"
#include "orteaf/internal/execution/cuda/cuda_handles.h"
#include "orteaf/internal/execution/cuda/resource/cuda_kernel_base.h"
#include "orteaf/internal/execution/cuda/resource/cuda_kernel_metadata.h"

namespace orteaf::internal::execution::cuda::manager {

class CudaKernelBaseManager;
class CudaExecutionManager;

struct KernelBasePayloadPoolTraits {
  using Payload = ::orteaf::internal::execution::cuda::resource::CudaKernelBase;
  using Handle = ::orteaf::internal::execution::cuda::CudaKernelBaseHandle;
  using Key = ::orteaf::internal::execution::cuda::resource::CudaKernelBase::Key;
  using ExecuteFunc =
      ::orteaf::internal::execution::cuda::resource::CudaKernelBase::ExecuteFunc;

  struct Request {
    ::orteaf::internal::base::HeapVector<Key> keys;
    ExecuteFunc execute{nullptr};
  };

  struct Context {};

  static constexpr bool destroy_on_release = true;

  static bool create(Payload &payload, const Request &request, const Context &);
  static void destroy(Payload &payload, const Request &, const Context &);
};

using KernelBasePayloadPool =
    ::orteaf::internal::base::pool::SlotPool<KernelBasePayloadPoolTraits>;

struct KernelBaseControlBlockTag {};

using KernelBaseControlBlock = ::orteaf::internal::base::StrongControlBlock<
    ::orteaf::internal::execution::cuda::CudaKernelBaseHandle,
    ::orteaf::internal::execution::cuda::resource::CudaKernelBase,
    KernelBasePayloadPool>;

struct CudaKernelBaseManagerTraits {
  using PayloadPool = KernelBasePayloadPool;
  using ControlBlock = KernelBaseControlBlock;
  struct ControlBlockTag {};
  using PayloadHandle =
      ::orteaf::internal::execution::cuda::CudaKernelBaseHandle;
  static constexpr const char *Name = "CudaKernelBaseManager";
};

class CudaKernelBaseManager {
  using Core =
      ::orteaf::internal::base::PoolManager<CudaKernelBaseManagerTraits>;

public:
  using KernelBaseHandle =
      ::orteaf::internal::execution::cuda::CudaKernelBaseHandle;
  using Key = ::orteaf::internal::execution::cuda::resource::CudaKernelBase::Key;
  using ExecuteFunc =
      ::orteaf::internal::execution::cuda::resource::CudaKernelBase::ExecuteFunc;

  using ControlBlock = Core::ControlBlock;
  using ControlBlockHandle = Core::ControlBlockHandle;
  using ControlBlockPool = Core::ControlBlockPool;
  using KernelBaseLease = Core::StrongLeaseType;

public:
  struct Config {
    std::size_t control_block_capacity{0};
    std::size_t control_block_block_size{1};
    std::size_t control_block_growth_chunk_size{1};
    std::size_t payload_capacity{0};
    std::size_t payload_block_size{1};
    std::size_t payload_growth_chunk_size{1};
  };

  CudaKernelBaseManager() = default;
  CudaKernelBaseManager(const CudaKernelBaseManager &) = delete;
  CudaKernelBaseManager &operator=(const CudaKernelBaseManager &) = delete;
  CudaKernelBaseManager(CudaKernelBaseManager &&) = default;
  CudaKernelBaseManager &operator=(CudaKernelBaseManager &&) = default;
  ~CudaKernelBaseManager() = default;

private:
  struct InternalConfig {
    Config public_config{};
  };

  void configure(const InternalConfig &config);

  friend class CudaExecutionManager;

public:
  void shutdown();

  KernelBaseLease acquire(
      const ::orteaf::internal::base::HeapVector<Key> &keys);

  KernelBaseLease acquire(
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

} // namespace orteaf::internal::execution::cuda::manager

#endif // ORTEAF_ENABLE_CUDA
