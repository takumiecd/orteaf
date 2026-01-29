#include "orteaf/internal/execution/mps/resource/mps_kernel_base.h"

#if ORTEAF_ENABLE_MPS

namespace orteaf::internal::execution::mps::resource {

void MpsKernelBase::configure(
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
}

bool MpsKernelBase::initialize(
    const ::orteaf::internal::base::HeapVector<Key> &keys,
    ::orteaf::internal::execution::mps::manager::MpsDeviceManager::DeviceLease
        &device_lease) {
  reset();
  reserveKeys(keys.size());
  for (const auto &key : keys) {
    addKey(key.first.identifier.c_str(), key.second.identifier.c_str());
  }
  if (!device_lease) {
    reset();
    return false;
  }
  ::orteaf::internal::execution_context::mps::Context context{};
  configure(device_lease);

  if (keys.empty()) {
    return true;
  }

  const auto device = device_lease.payloadHandle();
  const auto idx = findDeviceIndex(device);
  if (idx == kInvalidIndex) {
    reset();
    return false;
  }
  const auto &entry = device_pipelines_[idx];
  if (!entry.configured || entry.pipelines.size() != keys.size()) {
    reset();
    return false;
  }
  for (const auto &pipeline : entry.pipelines) {
    if (!pipeline) {
      reset();
      return false;
    }
  }
  return true;
}

} // namespace orteaf::internal::execution::mps::resource

#endif // ORTEAF_ENABLE_MPS
