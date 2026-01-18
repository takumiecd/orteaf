#pragma once

#if ORTEAF_ENABLE_CUDA

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>

#include "orteaf/internal/base/lease/control_block/strong.h"
#include "orteaf/internal/base/lease/strong_lease.h"
#include "orteaf/internal/base/manager/lease_lifetime_registry.h"
#include "orteaf/internal/base/manager/pool_manager.h"
#include "orteaf/internal/base/pool/slot_pool.h"
#include "orteaf/internal/execution/cuda/cuda_handles.h"
#include "orteaf/internal/execution/cuda/platform/cuda_slow_ops.h"
#include "orteaf/internal/execution/cuda/platform/wrapper/cuda_module.h"

namespace orteaf::internal::execution::cuda::manager {

struct FunctionPayloadPoolTraits {
  using Payload =
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaFunction_t;
  using Handle = ::orteaf::internal::execution::cuda::CudaFunctionHandle;
  using ContextType =
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaContext_t;
  using ModuleType =
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaModule_t;
  using SlowOps = ::orteaf::internal::execution::cuda::platform::CudaSlowOps;

  struct Request {
    std::string name{};
  };

  struct Context {
    ContextType context{nullptr};
    ModuleType module{nullptr};
    SlowOps *ops{nullptr};
  };

  static bool create(Payload &payload, const Request &request,
                     const Context &context) {
    if (context.ops == nullptr || context.context == nullptr ||
        context.module == nullptr || request.name.empty()) {
      return false;
    }
    context.ops->setContext(context.context);
    payload = context.ops->getFunction(context.module, request.name.c_str());
    return payload != nullptr;
  }

  static void destroy(Payload &payload, const Request &,
                      const Context &) {
    payload = nullptr;
  }
};

using FunctionPayloadPool =
    ::orteaf::internal::base::pool::SlotPool<FunctionPayloadPoolTraits>;

struct FunctionControlBlockTag {};

using FunctionControlBlock = ::orteaf::internal::base::StrongControlBlock<
    ::orteaf::internal::execution::cuda::CudaFunctionHandle,
    ::orteaf::internal::execution::cuda::platform::wrapper::CudaFunction_t,
    FunctionPayloadPool>;

struct CudaFunctionManagerTraits {
  using PayloadPool = FunctionPayloadPool;
  using ControlBlock = FunctionControlBlock;
  struct ControlBlockTag {};
  using PayloadHandle = ::orteaf::internal::execution::cuda::CudaFunctionHandle;
  static constexpr const char *Name = "CUDA function manager";
};

class CudaFunctionManager {
public:
  using Core = ::orteaf::internal::base::PoolManager<CudaFunctionManagerTraits>;
  using SlowOps = ::orteaf::internal::execution::cuda::platform::CudaSlowOps;
  using ContextType =
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaContext_t;
  using ModuleType =
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaModule_t;
  using FunctionHandle = ::orteaf::internal::execution::cuda::CudaFunctionHandle;
  using FunctionType =
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaFunction_t;
  using FunctionLease = Core::StrongLeaseType;
  using LifetimeRegistry =
      ::orteaf::internal::base::manager::LeaseLifetimeRegistry<FunctionHandle,
                                                               FunctionLease>;

  struct Config {
    std::size_t control_block_capacity{0};
    std::size_t control_block_block_size{0};
    std::size_t control_block_growth_chunk_size{1};
    std::size_t payload_capacity{0};
    std::size_t payload_block_size{0};
    std::size_t payload_growth_chunk_size{1};
  };

  CudaFunctionManager() = default;
  CudaFunctionManager(const CudaFunctionManager &) = delete;
  CudaFunctionManager &operator=(const CudaFunctionManager &) = delete;
  CudaFunctionManager(CudaFunctionManager &&) = default;
  CudaFunctionManager &operator=(CudaFunctionManager &&) = default;
  ~CudaFunctionManager() = default;

private:
  struct InternalConfig {
    Config public_config{};
    ContextType context{nullptr};
    ModuleType module{nullptr};
    SlowOps *ops{nullptr};
  };

  void configure(const InternalConfig &config);

  friend struct ModulePayloadPoolTraits;

public:
  void shutdown();

  FunctionLease acquire(std::string_view name);
  FunctionLease acquire(FunctionHandle handle);

  FunctionType getFunction(std::string_view name);

#if ORTEAF_ENABLE_TEST
  void configureForTest(const Config &config, ContextType context,
                        ModuleType module, SlowOps *ops) {
    InternalConfig internal{};
    internal.public_config = config;
    internal.context = context;
    internal.module = module;
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
  bool isAliveForTest(FunctionHandle handle) const noexcept {
    return core_.isAlive(handle);
  }
#endif

private:
  void validateName(std::string_view name) const;
  FunctionPayloadPoolTraits::Context makePayloadContext() const noexcept;

  ContextType context_{nullptr};
  ModuleType module_{nullptr};
  SlowOps *ops_{nullptr};
  Core core_{};
  LifetimeRegistry lifetime_{};
  std::unordered_map<std::string, std::size_t> name_to_index_{};
};

} // namespace orteaf::internal::execution::cuda::manager

#endif // ORTEAF_ENABLE_CUDA
