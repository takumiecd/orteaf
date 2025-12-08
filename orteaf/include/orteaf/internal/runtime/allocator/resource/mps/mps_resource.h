#pragma once

#if ORTEAF_ENABLE_MPS

#include <cstddef>

#include <orteaf/internal/base/handle.h>
#include <orteaf/internal/base/heap_vector.h>
#include <orteaf/internal/runtime/base/backend_traits.h>
#include <orteaf/internal/runtime/mps/manager/mps_library_manager.h>
#include <orteaf/internal/runtime/mps/platform/wrapper/mps_buffer.h>
#include <orteaf/internal/runtime/mps/platform/wrapper/mps_heap.h>
#include <orteaf/internal/runtime/mps/resource/mps_buffer_view.h>
#include <orteaf/internal/runtime/mps/resource/mps_kernel_launcher.h>
#include <orteaf/internal/runtime/mps/resource/mps_reuse_token.h>
#include <unordered_map>

namespace orteaf::internal::backend::mps {

// Simple MPS resource that keeps device/heap handles per instance and creates
// buffers at offset 0.
class MpsResource {
public:
  using BufferView = ::orteaf::internal::runtime::mps::resource::MpsBufferView;
  using FenceToken = ::orteaf::internal::runtime::mps::resource::MpsFenceToken;
  using ReuseToken = ::orteaf::internal::runtime::mps::resource::MpsReuseToken;

  struct Config {
    ::orteaf::internal::base::DeviceHandle device_handle{};
    ::orteaf::internal::runtime::mps::platform::wrapper::MPSDevice_t device{
        nullptr};
    ::orteaf::internal::runtime::mps::platform::wrapper::MPSHeap_t heap{
        nullptr};
    ::orteaf::internal::runtime::mps::platform::wrapper::MPSHeap_t staging_heap{
        nullptr}; // optional host-visible heap for readback
    ::orteaf::internal::runtime::mps::platform::wrapper::MPSBufferUsage_t usage{
        ::orteaf::internal::runtime::mps::platform::wrapper::
            kMPSDefaultBufferUsage};
    ::orteaf::internal::runtime::mps::platform::wrapper::MPSBufferUsage_t
        staging_usage{::orteaf::internal::runtime::mps::platform::wrapper::
                          kMPSDefaultBufferUsage};
    ::orteaf::internal::runtime::mps::manager::MpsLibraryManager
        *library_manager{nullptr};
    std::size_t chunk_table_capacity{16};
  };

  MpsResource() = default;

  explicit MpsResource(const Config &config) { initialize(config); }

  void initialize(const Config &config);

  BufferView allocate(std::size_t size, std::size_t alignment);

  void deallocate(BufferView view, std::size_t size,
                  std::size_t alignment) noexcept;

  ::orteaf::internal::runtime::mps::platform::wrapper::MPSDevice_t
  device() const noexcept {
    return device_;
  }
  ::orteaf::internal::runtime::mps::platform::wrapper::MPSHeap_t
  heap() const noexcept {
    return heap_;
  }

  bool isCompleted(FenceToken &token);
  bool isCompleted(ReuseToken &token);

  static BufferView makeView(BufferView base, std::size_t offset,
                             std::size_t size);

  void initializeChunkAsFreelist(BufferView chunk, std::size_t chunk_size,
                                 std::size_t block_size);
  BufferView popFreelistNode();
  void pushFreelistNode(BufferView view);

private:
  void destroyFreelist();

  ::orteaf::internal::runtime::mps::platform::wrapper::MPSDevice_t device_{
      nullptr};
  ::orteaf::internal::base::DeviceHandle device_handle_{};
  ::orteaf::internal::runtime::mps::platform::wrapper::MPSHeap_t heap_{nullptr};
  ::orteaf::internal::runtime::mps::platform::wrapper::MPSBufferUsage_t usage_{
      ::orteaf::internal::runtime::mps::platform::wrapper::
          kMPSDefaultBufferUsage};
  bool initialized_{false};

  BufferView freelist_chunk_{};
  uint32_t freelist_chunk_id_{0};
  ::orteaf::internal::base::HeapVector<BufferView> chunks_{};
  std::unordered_map<void *, uint32_t> chunk_lookup_{};
  std::size_t freelist_block_size_{0};
  std::size_t freelist_block_count_{0};
  ::orteaf::internal::runtime::mps::platform::wrapper::MPSBuffer_t
      freelist_head_{nullptr};
  ::orteaf::internal::runtime::mps::platform::wrapper::MPSBuffer_t
      freelist_out_{nullptr};
  ::orteaf::internal::runtime::mps::platform::wrapper::MPSHeap_t staging_heap_{
      nullptr};
  ::orteaf::internal::runtime::mps::platform::wrapper::MPSBufferUsage_t
      staging_usage_{::orteaf::internal::runtime::mps::platform::wrapper::
                         kMPSDefaultBufferUsage};

  ::orteaf::internal::runtime::mps::resource::MpsKernelLauncher<3>
      freelist_launcher_{
          {{"freelist_block_embedded", "orteaf_freelist_init_block_embedded"},
           {"freelist_block_embedded", "orteaf_freelist_pop_block_embedded"},
           {"freelist_block_embedded", "orteaf_freelist_push_block_embedded"}}};
  ::orteaf::internal::runtime::mps::manager::MpsLibraryManager
      *library_manager_{nullptr};
};

} // namespace orteaf::internal::backend::mps

#endif // ORTEAF_ENABLE_MPS
