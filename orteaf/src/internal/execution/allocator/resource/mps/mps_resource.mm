
#include "orteaf/internal/execution/allocator/resource/mps/mps_resource.h"
#include <orteaf/internal/execution/mps/platform/mps_fast_ops.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_buffer.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_heap.h>

#include "orteaf/internal/diagnostics/error/error_macros.h"
#include <limits>

namespace orteaf::internal::execution::allocator::resource::mps {

void MpsResource::initialize(const Config &config) {
  ORTEAF_THROW_IF_NULL(config.device, "MpsResource requires non-null device");
  ORTEAF_THROW_IF_NULL(config.heap, "MpsResource requires non-null heap");
  ORTEAF_THROW_IF(!config.device_handle.isValid(), InvalidParameter,
                  "MpsResource requires a valid MpsDeviceHandle");
  destroyFreelist();
  device_ = config.device;
  device_handle_ = config.device_handle;
  heap_ = config.heap;
  usage_ = config.usage;
  if (usage_ ==
      ::orteaf::internal::execution::mps::platform::wrapper::
          kMPSDefaultBufferUsage) {
    // MTLHeap newBufferWithLength:options: requires options whose storage mode
    // is compatible with the heap mode. Default 0 implies Shared, which
    // aborts on Private heaps.
    constexpr std::uint32_t kMtlResourceStorageModeShift = 4u;
    const auto storage_mode = static_cast<std::uint32_t>(
        ::orteaf::internal::execution::mps::platform::wrapper::heapStorageMode(
            heap_));
    usage_ = static_cast<decltype(usage_)>(
        static_cast<std::uint64_t>(storage_mode)
        << kMtlResourceStorageModeShift);
  }
  initialized_ = (device_ != nullptr && heap_ != nullptr);
}

MpsResource::BufferView MpsResource::allocate(std::size_t size,
                                              std::size_t /*alignment*/) {
  ORTEAF_THROW_IF(!initialized_, InvalidState,
                  "MpsResource::allocate called before initialize");
  ORTEAF_THROW_IF(size == 0, InvalidParameter,
                  "MpsResource::allocate requires size > 0");

  ::orteaf::internal::execution::mps::platform::wrapper::MpsBuffer_t buffer =
      ::orteaf::internal::execution::mps::platform::wrapper::createBuffer(
          heap_, size, usage_);
  if (!buffer) {
    return {};
  }

  return BufferView{buffer, 0, size};
}

void MpsResource::deallocate(BufferView view, std::size_t /*size*/,
                             std::size_t /*alignment*/) noexcept {
  if (!initialized_ || !view) {
    return;
  }
  ::orteaf::internal::execution::mps::platform::wrapper::destroyBuffer(
      view.raw());
}

bool MpsResource::isCompleted(FenceToken &token) {
  ORTEAF_THROW_IF(!initialized_, InvalidState,
                  "MpsResource::isCompleted called before initialize");
  bool all_completed = true;
  
  // Check write fence
  if (token.hasWriteFence()) {
    auto &lease = token.writeFence();
    auto *payload = lease.operator->();
    if (payload == nullptr) {
      lease.release();
    } else if (payload->isReady<
                   ::orteaf::internal::execution::mps::platform::MpsFastOps>()) {
      lease.release();
    } else {
      all_completed = false;
    }
  }
  
  // Check read fences
  if (all_completed) {
    for (std::size_t i = 0; i < token.readFenceCount(); ++i) {
      auto &lease = token.readFence(i);
      auto *payload = lease.operator->();
      if (payload == nullptr) {
        lease.release();
        continue;
      }
      if (payload->isReady<
              ::orteaf::internal::execution::mps::platform::MpsFastOps>()) {
        lease.release();
        continue;
      }
      all_completed = false;
      break;
    }
  }

  return all_completed;
}

bool MpsResource::isCompleted(ReuseToken &token) {
  ORTEAF_THROW_IF(!initialized_, InvalidState,
                  "MpsResource::isCompleted called before initialize");
  bool all_completed = true;
  for (auto &hazard : token) {
    if (!hazard.hasCommandBuffer()) {
      continue;
    }
    if (hazard.isCompleted<
            ::orteaf::internal::execution::mps::platform::MpsFastOps>()) {
      continue;
    } else {
      all_completed = false;
      break;
    }
  }

  return all_completed;
}

MpsResource::BufferView
MpsResource::makeView(BufferView base, std::size_t offset, std::size_t size) {
  return BufferView{base.raw(), offset, size};
}

void MpsResource::initializeChunkAsFreelist(std::size_t list_index,
                                            BufferView chunk,
                                            std::size_t chunk_size,
                                            std::size_t block_size,
                                            const LaunchParams &launch_params) {
  // this method is still unimplemented
}

MpsResource::BufferView
MpsResource::popFreelistNode(std::size_t list_index,
                             const LaunchParams &launch_params) {
  return {};
}

void MpsResource::pushFreelistNode(std::size_t list_index, BufferView view,
                                   const LaunchParams &launch_params) {
  // this method is still unimplemented
}

void MpsResource::destroyFreelist() {
  // this method is still unimplemented
}

void MpsResource::ensureList(std::size_t list_index) {
  // this method is still unimplemented
}

} // namespace orteaf::internal::execution::allocator::resource::mps
