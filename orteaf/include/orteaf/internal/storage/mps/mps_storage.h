#pragma once

#include <utility>

#include <orteaf/internal/diagnostics/error/error.h>
#include <orteaf/internal/execution/mps/api/mps_execution_api.h>
#include <orteaf/internal/execution/allocator/resource/mps/mps_resource.h>
#include <orteaf/internal/execution/mps/manager/mps_buffer_manager.h>
#include <orteaf/internal/execution/mps/manager/mps_heap_manager.h>
#include <orteaf/internal/execution/mps/resource/mps_fence_token.h>
#include <orteaf/internal/storage/mps/mps_storage_layout.h>

namespace orteaf::internal::storage::mps {

class MpsStorage {
public:
  using MpsResource =
      ::orteaf::internal::execution::allocator::resource::mps::MpsResource;
  using BufferManager =
      ::orteaf::internal::execution::mps::manager::MpsBufferManager<
          MpsResource>;
  using BufferLease = BufferManager::StrongBufferLease;
  using HeapManager =
      ::orteaf::internal::execution::mps::manager::MpsHeapManager;
  using HeapLease = HeapManager::HeapLease;
  using HeapDescriptorKey =
      ::orteaf::internal::execution::mps::manager::HeapDescriptorKey;
  using DeviceHandle = ::orteaf::internal::execution::mps::MpsDeviceHandle;
  // TODO: Re-enable fence tokens after revisiting MpsFenceToken ownership/copy
  // rules. using FenceToken =
  // ::orteaf::internal::execution::mps::resource::MpsFenceToken;
  using Layout = ::orteaf::internal::storage::mps::MpsStorageLayout;

  /**
   * @brief Builder for constructing MpsStorage instances.
   *
   * Provides a fluent interface for configuring and creating MpsStorage.
   * The builder acquires a buffer lease from the manager upon build().
   *
   * @par Example
   * @code
   * auto storage = MpsStorage::builder()
   *     .withHeapLease(heap_lease)
   *     .withSize(1024)
   *     .withLayout(layout)
   *     .build();
   * @endcode
   */
  class Builder {
  public:
    Builder() = default;

    Builder &withHeapLease(const HeapLease &lease) {
      heap_lease_ = lease;
      return *this;
    }

    Builder &withDeviceHandle(DeviceHandle handle,
                              const HeapDescriptorKey &key) {
      heap_lease_ = ::orteaf::internal::execution::mps::api::
          MpsExecutionApi::acquireHeap(handle, key);
      return *this;
    }

    Builder &withSize(std::size_t size) {
      size_ = size;
      return *this;
    }

    Builder &withAlignment(std::size_t alignment) {
      alignment_ = alignment;
      return *this;
    }

    Builder &withLayout(Layout layout) {
      layout_ = std::move(layout);
      return *this;
    }

    // TODO: Add withFenceToken once MpsFenceToken is updated.

    /**
     * @brief Build the MpsStorage instance.
     *
     * Acquires a buffer lease from the configured manager and
     * constructs the MpsStorage.
     *
     * @return Constructed MpsStorage instance.
     * @throws If buffer_manager is null or acquisition fails.
     */
    MpsStorage build() {
      if (!heap_lease_) {
        ::orteaf::internal::diagnostics::error::throwError(
            ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
            "MpsStorage requires a valid heap lease");
      }
      BufferLease lease =
          heap_lease_->buffer_manager.acquire(size_, alignment_);
      return MpsStorage(std::move(lease), std::move(layout_));
    }

  private:
    HeapLease heap_lease_{};
    std::size_t size_{0};
    std::size_t alignment_{0};
    Layout layout_{};
  };

  /**
   * @brief Create a new Builder instance.
   * @return A new Builder for constructing MpsStorage.
   */
  static Builder builder() { return Builder{}; }

  MpsStorage() = default;

  MpsStorage(const MpsStorage &) = default;
  MpsStorage &operator=(const MpsStorage &) = default;
  MpsStorage(MpsStorage &&) = default;
  MpsStorage &operator=(MpsStorage &&) = default;
  ~MpsStorage() = default;

private:
  MpsStorage(BufferLease buffer_lease, Layout layout)
      : buffer_lease_(std::move(buffer_lease)), layout_(std::move(layout)) {}

  BufferLease buffer_lease_;
  // TODO: Re-enable once MpsFenceToken is updated.
  // FenceToken fence_token_{};
  Layout layout_;
};

} // namespace orteaf::internal::storage::mps
