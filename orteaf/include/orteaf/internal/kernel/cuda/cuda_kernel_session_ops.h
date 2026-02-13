#pragma once

#if ORTEAF_ENABLE_CUDA

#include <cstddef>
#include <cstdint>

#include "orteaf/internal/execution/cuda/platform/wrapper/cuda_dim.h"
#include "orteaf/internal/execution/cuda/platform/wrapper/cuda_stream.h"
#include "orteaf/internal/execution/cuda/resource/cuda_kernel_base.h"
#include "orteaf/internal/execution_context/cuda/context.h"

namespace orteaf::internal::kernel::cuda {

using CudaKernelBase = ::orteaf::internal::execution::cuda::resource::CudaKernelBase;

struct CudaKernelSessionOps {
  using FunctionLease = CudaKernelBase::FunctionLease;
  using FunctionType = CudaKernelBase::FunctionType;
  using CudaDim3_t =
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaDim3_t;
  using CudaStream_t =
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaStream_t;
  using Context = ::orteaf::internal::execution_context::cuda::Context;

  static CudaStream_t getStream(const Context &context) {
    if (!context.stream) {
      return nullptr;
    }
    auto *stream_payload = context.stream.operator->();
    if (stream_payload == nullptr || *stream_payload == nullptr) {
      return nullptr;
    }
    return *stream_payload;
  }

  static FunctionType getFunction(const FunctionLease &function_lease) {
    if (!function_lease) {
      return nullptr;
    }
    auto *payload = function_lease.operator->();
    if (payload == nullptr || *payload == nullptr) {
      return nullptr;
    }
    return *payload;
  }

  static void synchronizeStream(CudaStream_t stream) {
    if (stream == nullptr) {
      return;
    }
    ::orteaf::internal::execution::cuda::platform::wrapper::synchronizeStream(
        stream);
  }

  static CudaDim3_t makeDim3(std::uint32_t x, std::uint32_t y = 1,
                             std::uint32_t z = 1) {
    return ::orteaf::internal::execution::cuda::platform::wrapper::makeDim3(
        x, y, z);
  }

  static CudaDim3_t makeBlock1D(std::uint32_t threads_per_block = 256) {
    return makeDim3(threads_per_block, 1, 1);
  }

  static CudaDim3_t makeGrid1D(std::size_t count,
                               std::uint32_t threads_per_block = 256) {
    if (threads_per_block == 0) {
      return makeDim3(0, 1, 1);
    }
    const auto blocks =
        (count + static_cast<std::size_t>(threads_per_block) - 1) /
        static_cast<std::size_t>(threads_per_block);
    return makeDim3(static_cast<std::uint32_t>(blocks), 1, 1);
  }
};

} // namespace orteaf::internal::kernel::cuda

#endif // ORTEAF_ENABLE_CUDA
