#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <string>

#include <orteaf/internal/diagnostics/error/error.h>
#include <orteaf/internal/execution/cuda/platform/wrapper/cuda_alloc.h>

namespace orteaf::extension::kernel::cuda::copy_detail {

namespace error = ::orteaf::internal::diagnostics::error;
namespace cuda_wrapper =
    ::orteaf::internal::execution::cuda::platform::wrapper;

inline void throwCudaRuntimeError(const char *message, cudaError_t status) {
  error::throwError(error::OrteafErrc::OperationFailed,
                    std::string(message) + ": " + cudaGetErrorString(status));
}

struct DeviceBufferGuard {
  cuda_wrapper::CudaDevicePtr_t ptr{};
  std::size_t bytes{};

  DeviceBufferGuard() = default;
  DeviceBufferGuard(cuda_wrapper::CudaDevicePtr_t buffer_ptr,
                    std::size_t buffer_bytes) noexcept
      : ptr(buffer_ptr), bytes(buffer_bytes) {}
  DeviceBufferGuard(const DeviceBufferGuard &) = delete;
  DeviceBufferGuard &operator=(const DeviceBufferGuard &) = delete;
  DeviceBufferGuard(DeviceBufferGuard &&other) noexcept
      : ptr(other.ptr), bytes(other.bytes) {
    other.ptr = 0;
    other.bytes = 0;
  }
  DeviceBufferGuard &operator=(DeviceBufferGuard &&other) noexcept {
    if (this == &other) {
      return *this;
    }
    reset();
    ptr = other.ptr;
    bytes = other.bytes;
    other.ptr = 0;
    other.bytes = 0;
    return *this;
  }
  ~DeviceBufferGuard() noexcept { reset(); }

private:
  void reset() noexcept {
    if (ptr == 0) {
      return;
    }
    try {
      cuda_wrapper::free(ptr, bytes);
    } catch (...) {
    }
    ptr = 0;
    bytes = 0;
  }
};

} // namespace orteaf::extension::kernel::cuda::copy_detail
