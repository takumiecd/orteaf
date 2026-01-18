#pragma once

#if ORTEAF_ENABLE_CUDA

#include <cstddef>
#include <cstdint>
#include <utility>

#include "orteaf/internal/architecture/architecture.h"
#include "orteaf/internal/architecture/cuda_detect.h"
#include "orteaf/internal/base/lease/control_block/strong.h"
#include "orteaf/internal/base/manager/lease_lifetime_registry.h"
#include "orteaf/internal/base/manager/pool_manager.h"
#include "orteaf/internal/base/pool/fixed_slot_store.h"
#include "orteaf/internal/execution/cuda/cuda_handles.h"
#include "orteaf/internal/execution/cuda/platform/cuda_slow_ops.h"
#include "orteaf/internal/runtime/cuda/manager/cuda_context_manager.h"

namespace orteaf::internal::runtime::cuda::manager {

class CudaExecutionManager;

// =============================================================================
// Device Resource
// =============================================================================

struct CudaDeviceResource {
  using SlowOps = ::orteaf::internal::execution::cuda::platform::CudaSlowOps;
  using DeviceType =
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaDevice_t;

  DeviceType device{};
  ::orteaf::internal::architecture::Architecture arch{
      ::orteaf::internal::architecture::Architecture::CudaGeneric};
  CudaContextManager context_manager{};

  CudaDeviceResource() = default;
  CudaDeviceResource(const CudaDeviceResource &) = delete;
  CudaDeviceResource &operator=(const CudaDeviceResource &) = delete;

  CudaDeviceResource(CudaDeviceResource &&other) noexcept {
    moveFrom(std::move(other));
  }

  CudaDeviceResource &operator=(CudaDeviceResource &&other) noexcept {
    if (this != &other) {
      reset(nullptr);
      moveFrom(std::move(other));
    }
    return *this;
  }

  ~CudaDeviceResource() { reset(nullptr); }

  void reset([[maybe_unused]] SlowOps *slow_ops) noexcept {
    context_manager.shutdown();
    device = DeviceType{};
    arch = ::orteaf::internal::architecture::Architecture::CudaGeneric;
  }

private:
  void moveFrom(CudaDeviceResource &&other) noexcept {
    device = other.device;
    arch = other.arch;
    context_manager = std::move(other.context_manager);
    other.device = DeviceType{};
    other.arch = ::orteaf::internal::architecture::Architecture::CudaGeneric;
  }
};

// =============================================================================
// Payload Pool Traits
// =============================================================================

struct DevicePayloadPoolTraits {
  using Payload = CudaDeviceResource;
  using Handle = ::orteaf::internal::execution::cuda::CudaDeviceHandle;
  using SlowOps = ::orteaf::internal::execution::cuda::platform::CudaSlowOps;

  struct Request {
    Handle handle{Handle::invalid()};
  };

  struct Context {
    SlowOps *ops{nullptr};
    CudaContextManager::Config context_config{};
  };

  static bool create(Payload &payload, const Request &request,
                     const Context &context);
  static void destroy(Payload &payload, const Request &request,
                      const Context &context);
};

// =============================================================================
// Payload Pool
// =============================================================================

using DevicePayloadPool =
    ::orteaf::internal::base::pool::FixedSlotStore<DevicePayloadPoolTraits>;

struct DeviceManagerCBTag {};

// =============================================================================
// ControlBlock
// =============================================================================

using DeviceControlBlock = ::orteaf::internal::base::StrongControlBlock<
    ::orteaf::internal::execution::cuda::CudaDeviceHandle, CudaDeviceResource,
    DevicePayloadPool>;

// =============================================================================
// Manager Traits
// =============================================================================

struct CudaDeviceManagerTraits {
  using PayloadPool = DevicePayloadPool;
  using ControlBlock = DeviceControlBlock;
  using ControlBlockTag = DeviceManagerCBTag;
  using PayloadHandle = ::orteaf::internal::execution::cuda::CudaDeviceHandle;
  static constexpr const char *Name = "CUDA device manager";
};

// =============================================================================
// CudaDeviceManager
// =============================================================================

class CudaDeviceManager {
public:
  using SlowOps = ::orteaf::internal::execution::cuda::platform::CudaSlowOps;
  using DeviceHandle = ::orteaf::internal::execution::cuda::CudaDeviceHandle;

  using Core = ::orteaf::internal::base::PoolManager<CudaDeviceManagerTraits>;
  using ControlBlock = Core::ControlBlock;
  using ControlBlockHandle = Core::ControlBlockHandle;
  using ControlBlockPool = Core::ControlBlockPool;

  using DeviceLease = Core::StrongLeaseType;
  using LifetimeRegistry =
      ::orteaf::internal::base::manager::LeaseLifetimeRegistry<DeviceHandle,
                                                               DeviceLease>;

  struct Config {
    std::size_t control_block_capacity{0};
    std::size_t control_block_block_size{0};
    std::size_t control_block_growth_chunk_size{1};
    std::size_t payload_capacity{0};
    std::size_t payload_block_size{0};
    std::size_t payload_growth_chunk_size{1};
    CudaContextManager::Config context_config{};
  };

  CudaDeviceManager() = default;
  CudaDeviceManager(const CudaDeviceManager &) = delete;
  CudaDeviceManager &operator=(const CudaDeviceManager &) = delete;
  CudaDeviceManager(CudaDeviceManager &&) = default;
  CudaDeviceManager &operator=(CudaDeviceManager &&) = default;
  ~CudaDeviceManager() = default;

private:
  struct InternalConfig {
    Config public_config{};
    SlowOps *ops{nullptr};
  };

  void configure(const InternalConfig &config);

  friend class CudaExecutionManager;

public:
  void shutdown();

  DeviceLease acquire(DeviceHandle handle);

#if ORTEAF_ENABLE_TEST
  void configureForTest(const Config &config, SlowOps *ops) {
    InternalConfig internal{};
    internal.public_config = config;
    internal.ops = ops;
    configure(internal);
  }

  std::size_t getDeviceCountForTest() const noexcept {
    return core_.payloadPoolSizeForTest();
  }
  bool isConfiguredForTest() const noexcept { return core_.isConfigured(); }
  std::size_t payloadPoolSizeForTest() const noexcept {
    return core_.payloadPoolSizeForTest();
  }
  std::size_t payloadPoolCapacityForTest() const noexcept {
    return core_.payloadPoolCapacityForTest();
  }
  bool isAliveForTest(DeviceHandle handle) const noexcept {
    return core_.isAlive(handle);
  }
  std::size_t controlBlockPoolAvailableForTest() const noexcept {
    return core_.controlBlockPoolAvailableForTest();
  }
#endif

private:
  SlowOps *ops_{nullptr};
  Core core_{};
  LifetimeRegistry lifetime_{};
};

} // namespace orteaf::internal::runtime::cuda::manager

#endif // ORTEAF_ENABLE_CUDA
