
#include "orteaf/internal/runtime/allocator/resource/mps/mps_resource.h"
#include "orteaf/internal/runtime/mps/manager/mps_compute_pipeline_state_manager.h"
#include <orteaf/internal/runtime/allocator/resource/mps/mps_resource.h>
#include <orteaf/internal/runtime/mps/platform/wrapper/mps_buffer.h>
#include <orteaf/internal/runtime/mps/platform/wrapper/mps_command_buffer.h>

#include "orteaf/internal/diagnostics/error/error_macros.h"

namespace orteaf::internal::backend::mps {

namespace {
constexpr uint32_t kInvalidIndex = 0xffffffffu;
}

namespace {
class ResourcePrivateOps {
public:
  using DeviceHandle = ::orteaf::internal::base::DeviceHandle;
  using LibraryKey = ::orteaf::internal::runtime::mps::manager::LibraryKey;
  using FunctionKey = ::orteaf::internal::runtime::mps::manager::FunctionKey;
  using PipelineLease = ::orteaf::internal::runtime::mps::manager::
      MpsComputePipelineStateManager::PipelineLease;

  static void setManager(
      ::orteaf::internal::runtime::mps::manager::MpsLibraryManager *mgr) {
    manager_ = mgr;
  }

  static PipelineLease acquirePipeline(DeviceHandle /*device*/,
                                       const LibraryKey &library_key,
                                       const FunctionKey &function_key) {
    ORTEAF_THROW_IF_NULL(manager_,
                         "ResourcePrivateOps requires library manager");
    auto library = manager_->acquire(library_key);
    auto pipeline_mgr = manager_->acquirePipelineManager(library);
    return pipeline_mgr->acquire(function_key);
  }

private:
  static inline ::orteaf::internal::runtime::mps::manager::MpsLibraryManager
      *manager_{nullptr};
};
} // namespace

void MpsResource::initialize(const Config &config) {
  ORTEAF_THROW_IF_NULL(config.device, "MpsResource requires non-null device");
  ORTEAF_THROW_IF_NULL(config.heap, "MpsResource requires non-null heap");
  ORTEAF_THROW_IF_NULL(config.library_manager,
                       "MpsResource requires non-null library_manager");
  ORTEAF_THROW_IF(!config.device_handle.isValid(), InvalidParameter,
                  "MpsResource requires a valid DeviceHandle");
  device_ = config.device;
  device_handle_ = config.device_handle;
  heap_ = config.heap;
  usage_ = config.usage;
  staging_heap_ = config.staging_heap ? config.staging_heap : heap_;
  staging_usage_ = config.staging_usage;
  library_manager_ = config.library_manager;
  chunks_.reserve(config.chunk_table_capacity);
  chunk_lookup_.reserve(config.chunk_table_capacity);
  initialized_ = (device_ != nullptr && heap_ != nullptr);
}

MpsResource::BufferView MpsResource::allocate(std::size_t size,
                                              std::size_t /*alignment*/) {
  ORTEAF_THROW_IF(!initialized_, InvalidState,
                  "MpsResource::allocate called before initialize");
  ORTEAF_THROW_IF(size == 0, InvalidParameter,
                  "MpsResource::allocate requires size > 0");

  ::orteaf::internal::runtime::mps::platform::wrapper::MPSBuffer_t buffer =
      ::orteaf::internal::runtime::mps::platform::wrapper::createBuffer(
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
  ::orteaf::internal::runtime::mps::platform::wrapper::destroyBuffer(
      view.raw());
}

bool MpsResource::isCompleted(FenceToken &token) {
  ORTEAF_THROW_IF(!initialized_, InvalidState,
                  "MpsResource::isCompleted called before initialize");
  bool all_completed = true;
  for (auto &ticket : token) {
    if (!ticket.valid()) {
      continue;
    }
    if (::orteaf::internal::runtime::mps::platform::wrapper::isCompleted(
            ticket.commandBuffer())) {
      ticket.reset(); // mark as invalid so subsequent calls skip it
      continue;
    } else {
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
  for (auto &ticket : token) {
    if (!ticket.valid()) {
      continue;
    }
    if (::orteaf::internal::runtime::mps::platform::wrapper::isCompleted(
            ticket.commandBuffer())) {
      ticket.reset();
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

void MpsResource::initializeChunkAsFreelist(BufferView chunk,
                                            std::size_t chunk_size,
                                            std::size_t block_size) {
  ORTEAF_THROW_IF(!initialized_, InvalidState,
                  "MpsResource::initializeChunkAsFreelist before initialize");
  ORTEAF_THROW_IF(block_size == 0, InvalidParameter, "block_size must be > 0");

  freelist_chunk_ = chunk;
  freelist_chunk_id_ = 0;
  freelist_block_size_ = block_size;
  freelist_block_count_ = (block_size == 0) ? 0 : (chunk_size / block_size);

  if (!chunk || freelist_block_count_ == 0) {
    return;
  }

  if (!freelist_head_) {
    freelist_head_ =
        ::orteaf::internal::runtime::mps::platform::wrapper::createBuffer(
            staging_heap_, 2 * sizeof(uint32_t), staging_usage_);
    auto *head_ptr = static_cast<uint32_t *>(
        ::orteaf::internal::runtime::mps::platform::wrapper::getBufferContents(
            freelist_head_));
    ORTEAF_THROW_IF(head_ptr == nullptr, InvalidState,
                    "freelist_head must be host-visible");
    head_ptr[0] = kInvalidIndex;
    head_ptr[1] = kInvalidIndex;
  }
  if (!freelist_out_) {
    freelist_out_ =
        ::orteaf::internal::runtime::mps::platform::wrapper::createBuffer(
            staging_heap_, 2 * sizeof(uint32_t), staging_usage_);
    ORTEAF_THROW_IF(
        ::orteaf::internal::runtime::mps::platform::wrapper::getBufferContents(
            freelist_out_) == nullptr,
        InvalidState, "freelist_out must be host-visible");
  }

  const uint32_t chunk_id = static_cast<uint32_t>(chunks_.size());
  chunks_.pushBack(chunk);
  chunk_lookup_[chunk.raw()] = chunk_id;
  freelist_chunk_id_ = chunk_id;

  if (!freelist_launcher_.initialized(device_handle_)) {
    ResourcePrivateOps::setManager(library_manager_);
    freelist_launcher_.initialize<ResourcePrivateOps>(device_handle_);
  }

  auto *command_queue =
      ::orteaf::internal::runtime::mps::platform::wrapper::createCommandQueue(
          device_);
  ORTEAF_THROW_IF_NULL(command_queue, "Failed to create MPS command queue");
  auto *command_buffer = freelist_launcher_.createCommandBuffer(command_queue);
  ORTEAF_THROW_IF_NULL(command_buffer, "Failed to create MPS command buffer");

  auto *encoder = freelist_launcher_.createComputeEncoder(command_buffer,
                                                          device_handle_, 0);
  ORTEAF_THROW_IF_NULL(encoder,
                       "Failed to create compute encoder for freelist_init");

  freelist_launcher_.setBuffer(encoder, freelist_head_, 0, 0); // head_offset
  freelist_launcher_.setBuffer(encoder, freelist_head_, sizeof(uint32_t),
                               1); // head_chunk_id
  freelist_launcher_.setBuffer(encoder, freelist_chunk_.raw(),
                               freelist_chunk_.offset(), 2);
  const uint32_t chunk_size32 = static_cast<uint32_t>(chunk_size);
  const uint32_t block_size32 = static_cast<uint32_t>(block_size);
  freelist_launcher_.setBytes(encoder, &chunk_size32, sizeof(uint32_t), 3);
  freelist_launcher_.setBytes(encoder, &block_size32, sizeof(uint32_t), 4);
  freelist_launcher_.setBuffer(encoder, freelist_out_, 0, 5);
  freelist_launcher_.setBytes(encoder, &freelist_chunk_id_, sizeof(uint32_t),
                              6);
  uint32_t old_head_offset = kInvalidIndex;
  uint32_t old_head_chunk = kInvalidIndex;
  if (auto *head_ptr = static_cast<uint32_t *>(
          ::orteaf::internal::runtime::mps::platform::wrapper::
              getBufferContents(freelist_head_))) {
    old_head_offset = head_ptr[0];
    old_head_chunk = head_ptr[1];
  }
  freelist_launcher_.setBytes(encoder, &old_head_offset, sizeof(uint32_t), 7);
  freelist_launcher_.setBytes(encoder, &old_head_chunk, sizeof(uint32_t), 8);

  ::orteaf::internal::runtime::mps::platform::wrapper::MPSSize_t tg{1, 1, 1};
  freelist_launcher_.dispatchThreadgroups(encoder, tg, tg);
  freelist_launcher_.endEncoding(encoder);
  freelist_launcher_.commit(command_buffer);
  ::orteaf::internal::runtime::mps::platform::wrapper::waitUntilCompleted(
      command_buffer);
  ::orteaf::internal::runtime::mps::platform::wrapper::destroyCommandBuffer(
      command_buffer);
  ::orteaf::internal::runtime::mps::platform::wrapper::destroyCommandQueue(
      command_queue);
}

MpsResource::BufferView MpsResource::popFreelistNode() {
  ORTEAF_THROW_IF(!initialized_, InvalidState,
                  "MpsResource::popFreelistNode before initialize");
  if (!freelist_head_ || !freelist_out_ || chunks_.empty()) {
    return {};
  }

  // Determine current head chunk to bind.
  BufferView current_chunk{};
  if (auto *head_ptr = static_cast<uint32_t *>(
          ::orteaf::internal::runtime::mps::platform::wrapper::
              getBufferContents(freelist_head_))) {
    const uint32_t head_chunk = head_ptr[1];
    if (head_chunk != kInvalidIndex && head_chunk < chunks_.size()) {
      current_chunk = chunks_[head_chunk];
    }
  }
  if (!current_chunk) {
    return {};
  }

  if (!freelist_launcher_.initialized(device_handle_)) {
    ResourcePrivateOps::setManager(library_manager_);
    freelist_launcher_.initialize<ResourcePrivateOps>(device_handle_);
  }

  auto *command_queue =
      ::orteaf::internal::runtime::mps::platform::wrapper::createCommandQueue(
          device_);
  ORTEAF_THROW_IF_NULL(command_queue, "Failed to create MPS command queue");
  auto *command_buffer = freelist_launcher_.createCommandBuffer(command_queue);
  ORTEAF_THROW_IF_NULL(command_buffer, "Failed to create MPS command buffer");
  auto *encoder = freelist_launcher_.createComputeEncoder(command_buffer,
                                                          device_handle_, 1);
  ORTEAF_THROW_IF_NULL(encoder,
                       "Failed to create compute encoder for freelist_pop");

  freelist_launcher_.setBuffer(encoder, freelist_head_, 0, 0); // head_offset
  freelist_launcher_.setBuffer(encoder, freelist_head_, sizeof(uint32_t),
                               1); // head_chunk_id
  freelist_launcher_.setBuffer(encoder, current_chunk.raw(),
                               current_chunk.offset(), 2);
  freelist_launcher_.setBuffer(encoder, freelist_out_, 0, 3);

  ::orteaf::internal::runtime::mps::platform::wrapper::MPSSize_t tg{1, 1, 1};
  freelist_launcher_.dispatchThreadgroups(encoder, tg, tg);
  freelist_launcher_.endEncoding(encoder);

  freelist_launcher_.commit(command_buffer);
  ::orteaf::internal::runtime::mps::platform::wrapper::waitUntilCompleted(
      command_buffer);
  ::orteaf::internal::runtime::mps::platform::wrapper::destroyCommandBuffer(
      command_buffer);
  ::orteaf::internal::runtime::mps::platform::wrapper::destroyCommandQueue(
      command_queue);

  auto *out_ptr = static_cast<uint32_t *>(
      ::orteaf::internal::runtime::mps::platform::wrapper::getBufferContents(
          freelist_out_));
  const uint32_t index = out_ptr ? out_ptr[0] : kInvalidIndex;
  const uint32_t chunk_id = out_ptr ? out_ptr[1] : kInvalidIndex;

  if (index == kInvalidIndex) {
    return {};
  }

  ORTEAF_THROW_IF(chunk_id >= chunks_.size(), InvalidState,
                  "Invalid chunk_id returned from freelist");
  const BufferView chunk = chunks_[chunk_id];
  const std::size_t offset = chunk.offset() + static_cast<std::size_t>(index);
  return BufferView{chunk.raw(), offset, freelist_block_size_};
}

void MpsResource::pushFreelistNode(BufferView view) {
  ORTEAF_THROW_IF(!initialized_, InvalidState,
                  "MpsResource::pushFreelistNode before initialize");
  if (!freelist_head_ || !view) {
    return;
  }
  auto it = chunk_lookup_.find(view.raw());
  ORTEAF_THROW_IF(it == chunk_lookup_.end(), InvalidParameter,
                  "View must belong to a registered chunk");
  const uint32_t chunk_id = it->second;
  const BufferView chunk = chunks_[chunk_id];
  ORTEAF_THROW_IF(!chunk.contains(view, view.size()), InvalidParameter,
                  "View must belong to freelist chunk");

  const std::size_t diff = view.offset() - chunk.offset();
  ORTEAF_THROW_IF(diff % freelist_block_size_ != 0, InvalidParameter,
                  "View not aligned to block size");
  const uint32_t index = static_cast<uint32_t>(diff); // Offset in bytes

  if (!freelist_launcher_.initialized(device_handle_)) {
    ResourcePrivateOps::setManager(library_manager_);
    freelist_launcher_.initialize<ResourcePrivateOps>(device_handle_);
  }

  auto *command_queue =
      ::orteaf::internal::runtime::mps::platform::wrapper::createCommandQueue(
          device_);
  ORTEAF_THROW_IF_NULL(command_queue, "Failed to create MPS command queue");
  auto *command_buffer = freelist_launcher_.createCommandBuffer(command_queue);
  ORTEAF_THROW_IF_NULL(command_buffer, "Failed to create MPS command buffer");
  auto *encoder = freelist_launcher_.createComputeEncoder(command_buffer,
                                                          device_handle_, 2);
  ORTEAF_THROW_IF_NULL(encoder,
                       "Failed to create compute encoder for freelist_push");

  freelist_launcher_.setBuffer(encoder, freelist_head_, 0, 0); // head_offset
  freelist_launcher_.setBuffer(encoder, freelist_head_, sizeof(uint32_t),
                               1); // head_chunk_id
  freelist_launcher_.setBuffer(encoder, chunk.raw(), chunk.offset(), 2);
  freelist_launcher_.setBytes(encoder, &index, sizeof(uint32_t), 3);
  freelist_launcher_.setBytes(encoder, &chunk_id, sizeof(uint32_t), 4);

  ::orteaf::internal::runtime::mps::platform::wrapper::MPSSize_t tg{1, 1, 1};
  freelist_launcher_.dispatchThreadgroups(encoder, tg, tg);
  freelist_launcher_.endEncoding(encoder);
  freelist_launcher_.commit(command_buffer);
  ::orteaf::internal::runtime::mps::platform::wrapper::waitUntilCompleted(
      command_buffer);
  ::orteaf::internal::runtime::mps::platform::wrapper::destroyCommandBuffer(
      command_buffer);
  ::orteaf::internal::runtime::mps::platform::wrapper::destroyCommandQueue(
      command_queue);
}

void MpsResource::destroyFreelist() {
  if (freelist_head_) {
    ::orteaf::internal::runtime::mps::platform::wrapper::destroyBuffer(
        freelist_head_);
    freelist_head_ = nullptr;
  }
  if (freelist_out_) {
    ::orteaf::internal::runtime::mps::platform::wrapper::destroyBuffer(
        freelist_out_);
    freelist_out_ = nullptr;
  }
  freelist_chunk_ = {};
  freelist_chunk_id_ = 0;
  freelist_block_size_ = 0;
  freelist_block_count_ = 0;
  chunks_.clear();
  chunk_lookup_.clear();
}

} // namespace orteaf::internal::backend::mps
