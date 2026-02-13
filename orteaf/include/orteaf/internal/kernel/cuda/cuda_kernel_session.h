#pragma once

#if ORTEAF_ENABLE_CUDA

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

#include "orteaf/internal/execution/cuda/resource/cuda_kernel_base.h"
#include "orteaf/internal/execution_context/cuda/context.h"
#include "orteaf/internal/kernel/core/kernel_args.h"
#include "orteaf/internal/kernel/cuda/cuda_kernel_session_ops.h"

namespace orteaf::internal::kernel::cuda {

using CudaKernelBase = ::orteaf::internal::execution::cuda::resource::CudaKernelBase;

class CudaKernelSession {
public:
  using Ops = CudaKernelSessionOps;
  using FunctionLease = Ops::FunctionLease;
  using FunctionType = Ops::FunctionType;
  using CudaDim3_t = Ops::CudaDim3_t;
  using CudaStream_t = Ops::CudaStream_t;
  using CudaContextHandle = ::orteaf::internal::execution::cuda::CudaContextHandle;

  static std::optional<CudaKernelSession>
  begin(CudaKernelBase &base, ::orteaf::internal::kernel::KernelArgs &args,
        std::size_t kernel_index = 0) {
    auto *context =
        args.context()
            .tryAs<::orteaf::internal::execution_context::cuda::Context>();
    if (context == nullptr || !context->context) {
      return std::nullopt;
    }

    auto stream = Ops::getStream(*context);
    if (stream == nullptr) {
      return std::nullopt;
    }

    const auto context_handle = context->context.payloadHandle();
    auto function_lease = base.getFunctionLease(context_handle, kernel_index);
    if (!function_lease) {
      return std::nullopt;
    }
    if (Ops::getFunction(function_lease) == nullptr) {
      return std::nullopt;
    }

    return CudaKernelSession(args, context_handle, std::move(function_lease),
                             stream);
  }

  CudaKernelSession(CudaKernelSession &&other) noexcept
      : args_(other.args_), context_handle_(other.context_handle_),
        function_lease_(std::move(other.function_lease_)),
        stream_(other.stream_) {
    other.args_ = nullptr;
    other.context_handle_ = CudaContextHandle{};
    other.stream_ = nullptr;
  }

  CudaKernelSession &operator=(CudaKernelSession &&other) noexcept {
    if (this != &other) {
      args_ = other.args_;
      context_handle_ = other.context_handle_;
      function_lease_ = std::move(other.function_lease_);
      stream_ = other.stream_;
      other.args_ = nullptr;
      other.context_handle_ = CudaContextHandle{};
      other.stream_ = nullptr;
    }
    return *this;
  }

  CudaKernelSession(const CudaKernelSession &) = delete;
  CudaKernelSession &operator=(const CudaKernelSession &) = delete;

  ~CudaKernelSession() = default;

  const FunctionLease &functionLease() const noexcept { return function_lease_; }
  FunctionType function() const noexcept { return Ops::getFunction(function_lease_); }
  CudaStream_t stream() const noexcept { return stream_; }
  CudaContextHandle contextHandle() const noexcept { return context_handle_; }

  void synchronize() const { Ops::synchronizeStream(stream_); }

  static CudaDim3_t makeBlock1D(std::uint32_t threads_per_block = 256) {
    return Ops::makeBlock1D(threads_per_block);
  }

  static CudaDim3_t makeGrid1D(std::size_t count,
                               std::uint32_t threads_per_block = 256) {
    return Ops::makeGrid1D(count, threads_per_block);
  }

private:
  CudaKernelSession(::orteaf::internal::kernel::KernelArgs &args,
                    CudaContextHandle context_handle,
                    FunctionLease function_lease, CudaStream_t stream)
      : args_(&args), context_handle_(context_handle),
        function_lease_(std::move(function_lease)), stream_(stream) {}

  ::orteaf::internal::kernel::KernelArgs *args_{nullptr};
  CudaContextHandle context_handle_{};
  FunctionLease function_lease_{};
  CudaStream_t stream_{nullptr};
};

} // namespace orteaf::internal::kernel::cuda

#endif // ORTEAF_ENABLE_CUDA
