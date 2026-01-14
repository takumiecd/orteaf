#pragma once

#include <orteaf/internal/execution/execution.h>

#include <orteaf/internal/execution/cpu/resource/cpu_buffer.h>
#if ORTEAF_ENABLE_CUDA
#include <orteaf/internal/execution/cuda/resource/cuda_buffer.h>
#endif // ORTEAF_ENABLE_CUDA
#if ORTEAF_ENABLE_MPS
#include <orteaf/internal/execution/mps/resource/mps_buffer.h>
#endif // ORTEAF_ENABLE_MPS

namespace orteaf::internal::execution::allocator {

template <execution::Execution B> struct ExecutionBufferType;

template <> struct ExecutionBufferType<execution::Execution::Cpu> {
  using Buffer = ::orteaf::internal::execution::cpu::resource::CpuBuffer;
  using Block = ::orteaf::internal::execution::cpu::resource::CpuBufferBlock;
};

#if ORTEAF_ENABLE_CUDA
template <> struct ExecutionBufferType<execution::Execution::Cuda> {
  using Buffer = ::orteaf::internal::execution::cuda::resource::CudaBuffer;
  using Block = ::orteaf::internal::execution::cuda::resource::CudaBufferBlock;
};
#endif // ORTEAF_ENABLE_CUDA

#if ORTEAF_ENABLE_MPS
template <> struct ExecutionBufferType<execution::Execution::Mps> {
  using Buffer = ::orteaf::internal::execution::mps::resource::MpsBuffer;
  using Block = ::orteaf::internal::execution::mps::resource::MpsBufferBlock;
};
#endif // ORTEAF_ENABLE_MPS

template <execution::Execution B>
using ExecutionBuffer = typename ExecutionBufferType<B>::Buffer;

template <execution::Execution B>
using ExecutionBufferBlock = typename ExecutionBufferType<B>::Block;

} // namespace orteaf::internal::execution::allocator
