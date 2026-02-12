#pragma once

#if ORTEAF_ENABLE_CUDA

#include <cstddef>
#include <limits>
#include <string>
#include <utility>

#include "orteaf/internal/base/heap_vector.h"
#include "orteaf/internal/execution/cuda/cuda_handles.h"
#include "orteaf/internal/execution/cuda/manager/cuda_context_manager.h"
#include "orteaf/internal/execution/cuda/manager/cuda_module_manager.h"
#include "orteaf/internal/execution/cuda/platform/wrapper/cuda_types.h"

namespace orteaf::internal::kernel {
class KernelArgs;
}

namespace orteaf::internal::kernel::core {
class KernelEntry;
}

namespace orteaf::internal::execution::cuda::manager {
struct KernelBasePayloadPoolTraits;
}

namespace orteaf::internal::execution::cuda::resource {

struct CudaKernelMetadata;

struct CudaKernelBase {
  using ExecuteFunc = void (*)(CudaKernelBase &,
                               ::orteaf::internal::kernel::KernelArgs &);
  using MetadataType = CudaKernelMetadata;
  using ContextLease = ::orteaf::internal::execution::cuda::manager::
      CudaContextManager::ContextLease;
  using ModuleLease = ::orteaf::internal::execution::cuda::manager::
      CudaModuleManager::ModuleLease;
  using FunctionLease = ::orteaf::internal::execution::cuda::manager::
      CudaFunctionManager::FunctionLease;
  using ModuleKey = ::orteaf::internal::execution::cuda::manager::ModuleKey;
  using FunctionKey = std::string;
  using Key = std::pair<ModuleKey, FunctionKey>;
  using FunctionType =
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaFunction_t;

  CudaKernelBase() = default;

  CudaKernelBase(const CudaKernelBase &) = delete;
  CudaKernelBase &operator=(const CudaKernelBase &) = delete;
  CudaKernelBase(CudaKernelBase &&) = default;
  CudaKernelBase &operator=(CudaKernelBase &&) = default;
  ~CudaKernelBase() = default;

  bool configured(
      ::orteaf::internal::execution::cuda::CudaContextHandle context) const
      noexcept {
    const auto idx = findContextIndex(context);
    return idx != kInvalidIndex && context_functions_[idx].configured;
  }

  void configureFunctions(ContextLease &context_lease);

  bool ensureFunctions(ContextLease &context_lease);

  void reset() noexcept {
    context_functions_.clear();
    keys_.clear();
    execute_ = nullptr;
  }

  FunctionLease
  getFunctionLease(::orteaf::internal::execution::cuda::CudaContextHandle context,
                   std::size_t index) const noexcept {
    const auto idx = findContextIndex(context);
    if (idx == kInvalidIndex) {
      return FunctionLease{};
    }
    const auto &entry = context_functions_[idx];
    if (!entry.configured || index >= entry.functions.size()) {
      return FunctionLease{};
    }
    return entry.functions[index];
  }

  FunctionType
  getFunction(::orteaf::internal::execution::cuda::CudaContextHandle context,
              std::size_t index) const noexcept {
    auto lease = getFunctionLease(context, index);
    if (!lease) {
      return nullptr;
    }
    auto *payload = lease.operator->();
    if (payload == nullptr || *payload == nullptr) {
      return nullptr;
    }
    return *payload;
  }

  std::size_t kernelCount() const noexcept { return keys_.size(); }

  const ::orteaf::internal::base::HeapVector<Key> &keys() const noexcept {
    return keys_;
  }

  ExecuteFunc execute() const noexcept { return execute_; }

private:
  struct ContextFunctions {
    ::orteaf::internal::execution::cuda::CudaContextHandle context{};
    ::orteaf::internal::base::HeapVector<ModuleLease> modules{};
    ::orteaf::internal::base::HeapVector<FunctionLease> functions{};
    bool configured{false};
  };

  std::size_t
  findContextIndex(::orteaf::internal::execution::cuda::CudaContextHandle context)
      const noexcept {
    for (std::size_t i = 0; i < context_functions_.size(); ++i) {
      if (context_functions_[i].context == context) {
        return i;
      }
    }
    return kInvalidIndex;
  }

private:
  friend class ::orteaf::internal::kernel::core::KernelEntry;
  friend struct CudaKernelMetadata;
  friend struct ::orteaf::internal::execution::cuda::manager::
      KernelBasePayloadPoolTraits;

  void run(::orteaf::internal::kernel::KernelArgs &args);

  bool setKeys(const ::orteaf::internal::base::HeapVector<Key> &keys);

  void setExecute(ExecuteFunc execute) noexcept { execute_ = execute; }

  ::orteaf::internal::base::HeapVector<ContextFunctions> context_functions_{};
  ::orteaf::internal::base::HeapVector<Key> keys_{};
  ExecuteFunc execute_{nullptr};
  static constexpr std::size_t kInvalidIndex =
      std::numeric_limits<std::size_t>::max();
};

} // namespace orteaf::internal::execution::cuda::resource

#endif // ORTEAF_ENABLE_CUDA
