#pragma once

#if ORTEAF_ENABLE_CUDA

namespace orteaf::internal::runtime::cuda::platform {

// Placeholder for CUDA fast-path operations. Methods will be added incrementally.
struct CudaFastOps {};

} // namespace orteaf::internal::runtime::cuda::platform

#endif // ORTEAF_ENABLE_CUDA
