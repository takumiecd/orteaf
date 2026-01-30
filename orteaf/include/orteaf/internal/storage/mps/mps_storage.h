#pragma once

#include <utility>

#include <orteaf/internal/diagnostics/error/error.h>
#include <orteaf/internal/dtype/dtype.h>
#include <orteaf/internal/execution/allocator/resource/mps/mps_resource.h>
#include <orteaf/internal/execution/execution.h>
#include <orteaf/internal/execution/mps/manager/mps_buffer_manager.h>
#include <orteaf/internal/execution/mps/manager/mps_heap_manager.h>

#include <orteaf/internal/execution/mps/platform/wrapper/mps_buffer.h>
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
  using HeapHandle = HeapManager::HeapHandle;
  using HeapDescriptorKey =
      ::orteaf::internal::execution::mps::manager::HeapDescriptorKey;
  using DeviceHandle = ::orteaf::internal::execution::mps::MpsDeviceHandle;
  using FenceToken =
      ::orteaf::internal::execution::mps::resource::MpsFenceToken;
  using Layout = ::orteaf::internal::storage::mps::MpsStorageLayout;
  using DType = ::orteaf::internal::DType;
  using Execution = ::orteaf::internal::execution::Execution;
  using MpsBuffer_t =
      ::orteaf::internal::execution::mps::platform::wrapper::MpsBuffer_t;
  using MpsBufferView =
      ::orteaf::internal::execution::mps::resource::MpsBufferView;

  /// @brief The execution backend for this storage type.
  static constexpr Execution kExecution = Execution::Mps;

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
   *     .withDType(DType::F32)
   *     .withNumElements(256)
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

    Builder &withHeapHandle(HeapHandle handle) {
      heap_handle_ = handle;
      return *this;
    }

    Builder &withHeapKey(const HeapDescriptorKey &key) {
      heap_key_ = key;
      has_heap_key_ = true;
      return *this;
    }

    Builder &withDType(DType dtype) {
      dtype_ = dtype;
      return *this;
    }

    Builder &withNumElements(std::size_t numel) {
      numel_ = numel;
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

    Builder &withFenceToken(FenceToken token) {
      fence_token_ = std::move(token);
      return *this;
    }

    /**
     * @brief Build the MpsStorage instance.
     *
     * Acquires a buffer lease from the configured manager and
     * constructs the MpsStorage.
     *
     * @return Constructed MpsStorage instance.
     * @throws If buffer_manager is null or acquisition fails.
     */
    MpsStorage build();

  private:
    HeapLease heap_lease_{};
    HeapHandle heap_handle_{};
    HeapDescriptorKey heap_key_{};
    bool has_heap_key_{false};
    DType dtype_{DType::F32};
    std::size_t numel_{0};
    std::size_t alignment_{0};
    Layout layout_{};
    FenceToken fence_token_{};
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

  /// @brief Return the execution backend for this storage.
  constexpr Execution execution() const { return kExecution; }

  /// @brief Return the data type of elements in this storage.
  DType dtype() const { return dtype_; }

  /// @brief Return the number of elements in this storage.
  std::size_t numel() const { return numel_; }

  /// @brief Return the size of the storage in bytes.
  std::size_t sizeInBytes() const {
    return numel_ * ::orteaf::internal::sizeOf(dtype_);
  }

  /// @brief Get the buffer lease.
  const BufferLease &bufferLease() const { return buffer_lease_; }

  /// @brief Get the buffer lease (mutable).
  BufferLease &bufferLease() { return buffer_lease_; }

  /**
   * @brief Get the buffer view.
   * @return MpsBufferView if valid, empty view otherwise.
   */
  MpsBufferView bufferView() const {
    if (!buffer_lease_) {
      return MpsBufferView{};
    }
    auto *buffer_payload = buffer_lease_.operator->();
    if (buffer_payload == nullptr || !buffer_payload->valid()) {
      return MpsBufferView{};
    }
    return buffer_payload->view;
  }

  /**
   * @brief Get the raw buffer pointer.
   * @return MpsBuffer_t if valid, nullptr otherwise.
   */
  MpsBuffer_t buffer() const {
    auto view = bufferView();
    return view ? view.raw() : nullptr;
  }

  /**
   * @brief Get the buffer offset in bytes.
   * @return Offset in bytes, or 0 if buffer is invalid.
   */
  std::size_t bufferOffset() const {
    auto view = bufferView();
    return view ? view.offset() : 0;
  }

  /**
   * @brief Get the fence token.
   * @return Reference to the fence token
   */
  FenceToken &fenceToken() { return fence_token_; }

  /**
   * @brief Get the fence token (const).
   * @return Const reference to the fence token
   */
  const FenceToken &fenceToken() const { return fence_token_; }

  /**
   * @brief Get the reuse token from the buffer.
   * @return Reference to the reuse token
   * @throws OrteafErrc::InvalidParameter if buffer lease is invalid
   */
  auto &reuseToken() {
    if (!buffer_lease_) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
          "Storage has no buffer lease");
    }
    auto *payload = buffer_lease_.operator->();
    if (!payload) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
          "Buffer lease has no payload");
    }
    return payload->reuse_token;
  }

  /**
   * @brief Get the reuse token from the buffer (const).
   * @return Const reference to the reuse token
   * @throws OrteafErrc::InvalidParameter if buffer lease is invalid
   */
  const auto &reuseToken() const {
    if (!buffer_lease_) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
          "Storage has no buffer lease");
    }
    auto *payload = buffer_lease_.operator->();
    if (!payload) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
          "Buffer lease has no payload");
    }
    return payload->reuse_token;
  }

private:
  MpsStorage(BufferLease buffer_lease, FenceToken fence_token, Layout layout,
             DType dtype, std::size_t numel)
      : buffer_lease_(std::move(buffer_lease)),
        fence_token_(std::move(fence_token)), layout_(std::move(layout)),
        dtype_(dtype), numel_(numel) {}

  BufferLease buffer_lease_;
  FenceToken fence_token_{};
  Layout layout_;
  DType dtype_{DType::F32};
  std::size_t numel_{0};
};

} // namespace orteaf::internal::storage::mps
