#include "orteaf/internal/execution/mps/resource/mps_kernel_base.h"

#if ORTEAF_ENABLE_MPS

namespace orteaf::internal::execution::mps::resource {

void MpsKernelBase::configurePipelines(
    ::orteaf::internal::execution::mps::manager::MpsDeviceManager::DeviceLease
        &device_lease) {
  auto device = device_lease.payloadHandle();
  auto *device_resource = device_lease.operator->();
  if (!device_resource) {
    return; // Invalid device lease
  }

  auto entry_idx = findDeviceIndex(device);
  if (entry_idx == kInvalidIndex) {
    device_pipelines_.pushBack(DevicePipelines{device});
    entry_idx = device_pipelines_.size() - 1;
  }
  auto &entry = device_pipelines_[entry_idx];
  entry.pipelines.clear();
  entry.pipelines.reserve(keys_.size());
  for (std::size_t i = 0; i < keys_.size(); ++i) {
    const auto &key = keys_[i];
    auto library_lease = device_resource->libraryManager().acquire(key.first);
    auto *library_resource = library_lease.operator->();
    if (library_resource) {
      entry.pipelines.pushBack(
          library_resource->pipelineManager().acquire(key.second));
    } else {
      entry.pipelines.pushBack(PipelineLease{}); // Invalid lease
    }
  }
  entry.configured = true;
  for (const auto &pipeline : entry.pipelines) {
    if (!pipeline) {
      entry.configured = false;
      break;
    }
  }
}

bool MpsKernelBase::setKeys(
    const ::orteaf::internal::base::HeapVector<Key> &keys) {
  reset();
  keys_.reserve(keys.size());
  for (const auto &key : keys) {
    keys_.pushBack(key);
  }
  return true;
}

bool MpsKernelBase::ensurePipelines(
    ::orteaf::internal::execution::mps::manager::MpsDeviceManager::DeviceLease
        &device_lease) {
  if (!device_lease) {
    return false;
  }
  if (keys_.empty()) {
    return true;
  }
  const auto device = device_lease.payloadHandle();
  if (!configured(device)) {
    configurePipelines(device_lease);
  }
  const auto idx = findDeviceIndex(device);
  if (idx == kInvalidIndex) {
    return false;
  }
  const auto &entry = device_pipelines_[idx];
  if (!entry.configured || entry.pipelines.size() != keys_.size()) {
    return false;
  }
  for (const auto &pipeline : entry.pipelines) {
    if (!pipeline) {
      return false;
    }
  }
  return true;
}

} // namespace orteaf::internal::execution::mps::resource

#endif // ORTEAF_ENABLE_MPS
