#pragma once

#include <string_view>

#include "orteaf/internal/architecture/architecture.h"
#include "orteaf/internal/architecture/mps_detect.h"
#include "orteaf/internal/backend/mps/wrapper/mps_command_queue.h"
#include "orteaf/internal/backend/mps/wrapper/mps_compute_pipeline_state.h"
#include "orteaf/internal/backend/mps/wrapper/mps_device.h"
#include "orteaf/internal/backend/mps/wrapper/mps_event.h"
#include "orteaf/internal/backend/mps/wrapper/mps_fence.h"
#include "orteaf/internal/backend/mps/wrapper/mps_function.h"
#include "orteaf/internal/backend/mps/wrapper/mps_heap.h"
#include "orteaf/internal/backend/mps/wrapper/mps_library.h"
#include "orteaf/internal/backend/mps/wrapper/mps_string.h"
#include "orteaf/internal/base/strong_id.h"

namespace orteaf::internal::runtime::backend_ops::mps {

struct MpsSlowOps {
  virtual ~MpsSlowOps() = default;

  virtual int getDeviceCount() = 0;

  virtual ::orteaf::internal::backend::mps::MPSDevice_t
  getDevice(::orteaf::internal::backend::mps::MPSInt_t index) = 0;

  virtual void
  releaseDevice(::orteaf::internal::backend::mps::MPSDevice_t device) = 0;

  virtual ::orteaf::internal::architecture::Architecture
  detectArchitecture(::orteaf::internal::base::DeviceId device_id) = 0;

  virtual ::orteaf::internal::backend::mps::MPSCommandQueue_t
  createCommandQueue(::orteaf::internal::backend::mps::MPSDevice_t device) = 0;

  virtual void destroyCommandQueue(
      ::orteaf::internal::backend::mps::MPSCommandQueue_t queue) = 0;

  virtual ::orteaf::internal::backend::mps::MPSEvent_t
  createEvent(::orteaf::internal::backend::mps::MPSDevice_t device) = 0;

  virtual void
  destroyEvent(::orteaf::internal::backend::mps::MPSEvent_t event) = 0;

  virtual ::orteaf::internal::backend::mps::MPSFence_t
  createFence(::orteaf::internal::backend::mps::MPSDevice_t device) = 0;

  virtual void
  destroyFence(::orteaf::internal::backend::mps::MPSFence_t fence) = 0;

  virtual ::orteaf::internal::backend::mps::MPSLibrary_t
  createLibraryWithName(::orteaf::internal::backend::mps::MPSDevice_t device,
                        std::string_view name) = 0;

  virtual void
  destroyLibrary(::orteaf::internal::backend::mps::MPSLibrary_t library) = 0;

  virtual ::orteaf::internal::backend::mps::MPSFunction_t
  createFunction(::orteaf::internal::backend::mps::MPSLibrary_t library,
                 std::string_view name) = 0;

  virtual void
  destroyFunction(::orteaf::internal::backend::mps::MPSFunction_t function) = 0;

  virtual ::orteaf::internal::backend::mps::MPSComputePipelineState_t
  createComputePipelineState(
      ::orteaf::internal::backend::mps::MPSDevice_t device,
      ::orteaf::internal::backend::mps::MPSFunction_t function) = 0;

  virtual void destroyComputePipelineState(
      ::orteaf::internal::backend::mps::MPSComputePipelineState_t
          pipeline_state) = 0;

  virtual ::orteaf::internal::backend::mps::MPSHeapDescriptor_t
  createHeapDescriptor() = 0;

  virtual void destroyHeapDescriptor(
      ::orteaf::internal::backend::mps::MPSHeapDescriptor_t descriptor) = 0;

  virtual void setHeapDescriptorSize(
      ::orteaf::internal::backend::mps::MPSHeapDescriptor_t descriptor,
      std::size_t size) = 0;

  virtual void setHeapDescriptorResourceOptions(
      ::orteaf::internal::backend::mps::MPSHeapDescriptor_t descriptor,
      ::orteaf::internal::backend::mps::MPSResourceOptions_t options) = 0;

  virtual void setHeapDescriptorStorageMode(
      ::orteaf::internal::backend::mps::MPSHeapDescriptor_t descriptor,
      ::orteaf::internal::backend::mps::MPSStorageMode_t storage_mode) = 0;

  virtual void setHeapDescriptorCPUCacheMode(
      ::orteaf::internal::backend::mps::MPSHeapDescriptor_t descriptor,
      ::orteaf::internal::backend::mps::MPSCPUCacheMode_t cpu_cache_mode) = 0;

  virtual void setHeapDescriptorHazardTrackingMode(
      ::orteaf::internal::backend::mps::MPSHeapDescriptor_t descriptor,
      ::orteaf::internal::backend::mps::MPSHazardTrackingMode_t
          hazard_mode) = 0;

  virtual void setHeapDescriptorType(
      ::orteaf::internal::backend::mps::MPSHeapDescriptor_t descriptor,
      ::orteaf::internal::backend::mps::MPSHeapType_t type) = 0;

  virtual ::orteaf::internal::backend::mps::MPSHeap_t createHeap(
      ::orteaf::internal::backend::mps::MPSDevice_t device,
      ::orteaf::internal::backend::mps::MPSHeapDescriptor_t descriptor) = 0;

  virtual void
  destroyHeap(::orteaf::internal::backend::mps::MPSHeap_t heap) = 0;
};

struct MpsSlowOpsImpl final : public MpsSlowOps {
  int getDeviceCount() override {
    return ::orteaf::internal::backend::mps::getDeviceCount();
  }

  ::orteaf::internal::backend::mps::MPSDevice_t
  getDevice(::orteaf::internal::backend::mps::MPSInt_t index) override {
    return ::orteaf::internal::backend::mps::getDevice(index);
  }

  void
  releaseDevice(::orteaf::internal::backend::mps::MPSDevice_t device) override {
    ::orteaf::internal::backend::mps::deviceRelease(device);
  }

  ::orteaf::internal::architecture::Architecture
  detectArchitecture(::orteaf::internal::base::DeviceId device_id) override {
    return ::orteaf::internal::architecture::detectMpsArchitectureForDeviceId(
        device_id);
  }

  ::orteaf::internal::backend::mps::MPSCommandQueue_t createCommandQueue(
      ::orteaf::internal::backend::mps::MPSDevice_t device) override {
    return ::orteaf::internal::backend::mps::createCommandQueue(device);
  }

  void destroyCommandQueue(
      ::orteaf::internal::backend::mps::MPSCommandQueue_t queue) override {
    ::orteaf::internal::backend::mps::destroyCommandQueue(queue);
  }

  ::orteaf::internal::backend::mps::MPSEvent_t
  createEvent(::orteaf::internal::backend::mps::MPSDevice_t device) override {
    return ::orteaf::internal::backend::mps::createEvent(device);
  }

  void
  destroyEvent(::orteaf::internal::backend::mps::MPSEvent_t event) override {
    ::orteaf::internal::backend::mps::destroyEvent(event);
  }

  ::orteaf::internal::backend::mps::MPSFence_t
  createFence(::orteaf::internal::backend::mps::MPSDevice_t device) override {
    return ::orteaf::internal::backend::mps::createFence(device);
  }

  void
  destroyFence(::orteaf::internal::backend::mps::MPSFence_t fence) override {
    ::orteaf::internal::backend::mps::destroyFence(fence);
  }

  ::orteaf::internal::backend::mps::MPSLibrary_t
  createLibraryWithName(::orteaf::internal::backend::mps::MPSDevice_t device,
                        std::string_view name) override {
    const auto ns_name = ::orteaf::internal::backend::mps::toNsString(name);
    return ::orteaf::internal::backend::mps::createLibrary(device, ns_name);
  }

  void destroyLibrary(
      ::orteaf::internal::backend::mps::MPSLibrary_t library) override {
    ::orteaf::internal::backend::mps::destroyLibrary(library);
  }

  ::orteaf::internal::backend::mps::MPSFunction_t
  createFunction(::orteaf::internal::backend::mps::MPSLibrary_t library,
                 std::string_view name) override {
    return ::orteaf::internal::backend::mps::createFunction(library, name);
  }

  void destroyFunction(
      ::orteaf::internal::backend::mps::MPSFunction_t function) override {
    ::orteaf::internal::backend::mps::destroyFunction(function);
  }

  ::orteaf::internal::backend::mps::MPSComputePipelineState_t
  createComputePipelineState(
      ::orteaf::internal::backend::mps::MPSDevice_t device,
      ::orteaf::internal::backend::mps::MPSFunction_t function) override {
    return ::orteaf::internal::backend::mps::createComputePipelineState(
        device, function);
  }

  void destroyComputePipelineState(
      ::orteaf::internal::backend::mps::MPSComputePipelineState_t
          pipeline_state) override {
    ::orteaf::internal::backend::mps::destroyComputePipelineState(
        pipeline_state);
  }

  ::orteaf::internal::backend::mps::MPSHeapDescriptor_t
  createHeapDescriptor() override {
    return ::orteaf::internal::backend::mps::createHeapDescriptor();
  }

  void destroyHeapDescriptor(
      ::orteaf::internal::backend::mps::MPSHeapDescriptor_t descriptor)
      override {
    ::orteaf::internal::backend::mps::destroyHeapDescriptor(descriptor);
  }

  void setHeapDescriptorSize(
      ::orteaf::internal::backend::mps::MPSHeapDescriptor_t descriptor,
      std::size_t size) override {
    ::orteaf::internal::backend::mps::setHeapDescriptorSize(descriptor, size);
  }

  void setHeapDescriptorResourceOptions(
      ::orteaf::internal::backend::mps::MPSHeapDescriptor_t descriptor,
      ::orteaf::internal::backend::mps::MPSResourceOptions_t options) override {
    ::orteaf::internal::backend::mps::setHeapDescriptorResourceOptions(
        descriptor, options);
  }

  void setHeapDescriptorStorageMode(
      ::orteaf::internal::backend::mps::MPSHeapDescriptor_t descriptor,
      ::orteaf::internal::backend::mps::MPSStorageMode_t storage_mode)
      override {
    ::orteaf::internal::backend::mps::setHeapDescriptorStorageMode(
        descriptor, storage_mode);
  }

  void setHeapDescriptorCPUCacheMode(
      ::orteaf::internal::backend::mps::MPSHeapDescriptor_t descriptor,
      ::orteaf::internal::backend::mps::MPSCPUCacheMode_t cpu_cache_mode)
      override {
    ::orteaf::internal::backend::mps::setHeapDescriptorCPUCacheMode(
        descriptor, cpu_cache_mode);
  }

  void setHeapDescriptorHazardTrackingMode(
      ::orteaf::internal::backend::mps::MPSHeapDescriptor_t descriptor,
      ::orteaf::internal::backend::mps::MPSHazardTrackingMode_t hazard_mode)
      override {
    ::orteaf::internal::backend::mps::setHeapDescriptorHazardTrackingMode(
        descriptor, hazard_mode);
  }

  void setHeapDescriptorType(
      ::orteaf::internal::backend::mps::MPSHeapDescriptor_t descriptor,
      ::orteaf::internal::backend::mps::MPSHeapType_t type) override {
    ::orteaf::internal::backend::mps::setHeapDescriptorType(descriptor, type);
  }

  ::orteaf::internal::backend::mps::MPSHeap_t
  createHeap(::orteaf::internal::backend::mps::MPSDevice_t device,
             ::orteaf::internal::backend::mps::MPSHeapDescriptor_t descriptor)
      override {
    return ::orteaf::internal::backend::mps::createHeap(device, descriptor);
  }

  void destroyHeap(::orteaf::internal::backend::mps::MPSHeap_t heap) override {
    ::orteaf::internal::backend::mps::destroyHeap(heap);
  }
};

} // namespace orteaf::internal::runtime::backend_ops::mps
