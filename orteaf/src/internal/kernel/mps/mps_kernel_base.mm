#include "orteaf/internal/kernel/mps/mps_kernel_base.h"

#if ORTEAF_ENABLE_MPS

namespace orteaf::internal::kernel::mps {

void MpsKernelBase::configure(
    ::orteaf::internal::execution_context::mps::Context &context) {
  auto device = context.device.payloadHandle();
  auto *device_resource = context.device.operator->();
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
    auto library_lease = device_resource->library_manager.acquire(key.first);
    auto *library_resource = library_lease.operator->();
    if (library_resource) {
      entry.pipelines.pushBack(
          library_resource->pipeline_manager.acquire(key.second));
    } else {
      entry.pipelines.pushBack(PipelineLease{}); // Invalid lease
    }
  }
  entry.configured = true;
}

} // namespace orteaf::internal::kernel::mps

#endif // ORTEAF_ENABLE_MPS
