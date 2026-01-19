#include "orteaf/internal/execution/cuda/platform/cuda_slow_ops.h"

#if ORTEAF_ENABLE_CUDA

namespace orteaf::internal::execution::cuda::platform {

int CudaSlowOpsImpl::getDeviceCount() {
  return ::orteaf::internal::execution::cuda::platform::wrapper::getDeviceCount();
}

::orteaf::internal::execution::cuda::platform::wrapper::CudaDevice_t
CudaSlowOpsImpl::getDevice(std::uint32_t index) {
  return ::orteaf::internal::execution::cuda::platform::wrapper::getDevice(
      index);
}

::orteaf::internal::execution::cuda::platform::wrapper::ComputeCapability
CudaSlowOpsImpl::getComputeCapability(
    ::orteaf::internal::execution::cuda::platform::wrapper::CudaDevice_t
        device) {
  return ::orteaf::internal::execution::cuda::platform::wrapper::getComputeCapability(
      device);
}

std::string CudaSlowOpsImpl::getDeviceName(
    ::orteaf::internal::execution::cuda::platform::wrapper::CudaDevice_t
        device) {
  return ::orteaf::internal::execution::cuda::platform::wrapper::getDeviceName(
      device);
}

std::string CudaSlowOpsImpl::getDeviceVendor(
    ::orteaf::internal::execution::cuda::platform::wrapper::CudaDevice_t
        device) {
  return ::orteaf::internal::execution::cuda::platform::wrapper::getDeviceVendor(
      device);
}

::orteaf::internal::execution::cuda::platform::wrapper::CudaContext_t
CudaSlowOpsImpl::getPrimaryContext(
    ::orteaf::internal::execution::cuda::platform::wrapper::CudaDevice_t
        device) {
  return ::orteaf::internal::execution::cuda::platform::wrapper::getPrimaryContext(
      device);
}

::orteaf::internal::execution::cuda::platform::wrapper::CudaContext_t
CudaSlowOpsImpl::createContext(
    ::orteaf::internal::execution::cuda::platform::wrapper::CudaDevice_t
        device) {
  return ::orteaf::internal::execution::cuda::platform::wrapper::createContext(
      device);
}

void CudaSlowOpsImpl::setContext(
    ::orteaf::internal::execution::cuda::platform::wrapper::CudaContext_t
        context) {
  ::orteaf::internal::execution::cuda::platform::wrapper::setContext(context);
}

void CudaSlowOpsImpl::releasePrimaryContext(
    ::orteaf::internal::execution::cuda::platform::wrapper::CudaDevice_t
        device) {
  ::orteaf::internal::execution::cuda::platform::wrapper::releasePrimaryContext(
      device);
}

void CudaSlowOpsImpl::releaseContext(
    ::orteaf::internal::execution::cuda::platform::wrapper::CudaContext_t
        context) {
  ::orteaf::internal::execution::cuda::platform::wrapper::releaseContext(
      context);
}

::orteaf::internal::execution::cuda::platform::wrapper::CudaStream_t
CudaSlowOpsImpl::createStream() {
  return ::orteaf::internal::execution::cuda::platform::wrapper::getStream();
}

void CudaSlowOpsImpl::destroyStream(
    ::orteaf::internal::execution::cuda::platform::wrapper::CudaStream_t
        stream) {
  ::orteaf::internal::execution::cuda::platform::wrapper::releaseStream(stream);
}

::orteaf::internal::execution::cuda::platform::wrapper::CudaEvent_t
CudaSlowOpsImpl::createEvent() {
  return ::orteaf::internal::execution::cuda::platform::wrapper::createEvent();
}

void CudaSlowOpsImpl::destroyEvent(
    ::orteaf::internal::execution::cuda::platform::wrapper::CudaEvent_t
        event) {
  ::orteaf::internal::execution::cuda::platform::wrapper::destroyEvent(event);
}

::orteaf::internal::execution::cuda::platform::wrapper::CudaModule_t
CudaSlowOpsImpl::loadModuleFromFile(const char *filepath) {
  return ::orteaf::internal::execution::cuda::platform::wrapper::
      loadModuleFromFile(filepath);
}

::orteaf::internal::execution::cuda::platform::wrapper::CudaModule_t
CudaSlowOpsImpl::loadModuleFromImage(const void *image) {
  return ::orteaf::internal::execution::cuda::platform::wrapper::
      loadModuleFromImage(image);
}

::orteaf::internal::execution::cuda::platform::wrapper::CudaFunction_t
CudaSlowOpsImpl::getFunction(
    ::orteaf::internal::execution::cuda::platform::wrapper::CudaModule_t cuda_module,
    const char *kernel_name) {
  return ::orteaf::internal::execution::cuda::platform::wrapper::getFunction(
      cuda_module, kernel_name);
}

void CudaSlowOpsImpl::unloadModule(
    ::orteaf::internal::execution::cuda::platform::wrapper::CudaModule_t
        cuda_module) {
  ::orteaf::internal::execution::cuda::platform::wrapper::unloadModule(cuda_module);
}

} // namespace orteaf::internal::execution::cuda::platform

#endif // ORTEAF_ENABLE_CUDA
