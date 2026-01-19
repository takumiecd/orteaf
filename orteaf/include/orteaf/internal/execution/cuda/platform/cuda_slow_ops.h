#pragma once

#if ORTEAF_ENABLE_CUDA

#include <cstdint>
#include <string>

#include "orteaf/internal/execution/cuda/platform/wrapper/cuda_context.h"
#include "orteaf/internal/execution/cuda/platform/wrapper/cuda_device.h"
#include "orteaf/internal/execution/cuda/platform/wrapper/cuda_event.h"
#include "orteaf/internal/execution/cuda/platform/wrapper/cuda_module.h"
#include "orteaf/internal/execution/cuda/platform/wrapper/cuda_stream.h"

namespace orteaf::internal::execution::cuda::platform {

// CUDA slow-path operations interface. Virtual to allow mocking and late binding.
struct CudaSlowOps {
  virtual ~CudaSlowOps() = default;

  virtual int getDeviceCount() = 0;
  virtual ::orteaf::internal::execution::cuda::platform::wrapper::CudaDevice_t
  getDevice(std::uint32_t index) = 0;
  virtual ::orteaf::internal::execution::cuda::platform::wrapper::
      ComputeCapability
      getComputeCapability(
          ::orteaf::internal::execution::cuda::platform::wrapper::CudaDevice_t
              device) = 0;
  virtual std::string getDeviceName(
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaDevice_t
          device) = 0;
  virtual std::string getDeviceVendor(
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaDevice_t
          device) = 0;

  virtual ::orteaf::internal::execution::cuda::platform::wrapper::CudaContext_t
  getPrimaryContext(
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaDevice_t
          device) = 0;
  virtual ::orteaf::internal::execution::cuda::platform::wrapper::CudaContext_t
  createContext(
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaDevice_t
          device) = 0;
  virtual void setContext(
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaContext_t
          context) = 0;
  virtual void releasePrimaryContext(
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaDevice_t
          device) = 0;
  virtual void releaseContext(
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaContext_t
          context) = 0;

  virtual ::orteaf::internal::execution::cuda::platform::wrapper::CudaStream_t
  createStream() = 0;
  virtual void destroyStream(
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaStream_t
          stream) = 0;

  virtual ::orteaf::internal::execution::cuda::platform::wrapper::CudaEvent_t
  createEvent() = 0;
  virtual void destroyEvent(
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaEvent_t
          event) = 0;

  virtual ::orteaf::internal::execution::cuda::platform::wrapper::CudaModule_t
  loadModuleFromFile(const char *filepath) = 0;
  virtual ::orteaf::internal::execution::cuda::platform::wrapper::CudaModule_t
  loadModuleFromImage(const void *image) = 0;
  virtual ::orteaf::internal::execution::cuda::platform::wrapper::CudaFunction_t
  getFunction(::orteaf::internal::execution::cuda::platform::wrapper::CudaModule_t
                  cuda_module,
              const char *kernel_name) = 0;
  virtual void unloadModule(
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaModule_t
          cuda_module) = 0;
};

// Default implementation backed by wrapper functions.
struct CudaSlowOpsImpl final : public CudaSlowOps {
  int getDeviceCount() override;
  ::orteaf::internal::execution::cuda::platform::wrapper::CudaDevice_t
  getDevice(std::uint32_t index) override;
  ::orteaf::internal::execution::cuda::platform::wrapper::ComputeCapability
  getComputeCapability(
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaDevice_t
          device) override;
  std::string getDeviceName(
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaDevice_t
          device) override;
  std::string getDeviceVendor(
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaDevice_t
          device) override;

  ::orteaf::internal::execution::cuda::platform::wrapper::CudaContext_t
  getPrimaryContext(
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaDevice_t
          device) override;
  ::orteaf::internal::execution::cuda::platform::wrapper::CudaContext_t
  createContext(
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaDevice_t
          device) override;
  void setContext(
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaContext_t
          context) override;
  void releasePrimaryContext(
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaDevice_t
          device) override;
  void releaseContext(
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaContext_t
          context) override;

  ::orteaf::internal::execution::cuda::platform::wrapper::CudaStream_t
  createStream() override;
  void destroyStream(
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaStream_t
          stream) override;

  ::orteaf::internal::execution::cuda::platform::wrapper::CudaEvent_t
  createEvent() override;
  void destroyEvent(
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaEvent_t
          event) override;

  ::orteaf::internal::execution::cuda::platform::wrapper::CudaModule_t
  loadModuleFromFile(const char *filepath) override;
  ::orteaf::internal::execution::cuda::platform::wrapper::CudaModule_t
  loadModuleFromImage(const void *image) override;
  ::orteaf::internal::execution::cuda::platform::wrapper::CudaFunction_t
  getFunction(::orteaf::internal::execution::cuda::platform::wrapper::CudaModule_t
                  cuda_module,
              const char *kernel_name) override;
  void unloadModule(
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaModule_t
          cuda_module) override;
};

} // namespace orteaf::internal::execution::cuda::platform

#endif // ORTEAF_ENABLE_CUDA
