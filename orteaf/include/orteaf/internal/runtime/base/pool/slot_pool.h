#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "orteaf/internal/base/heap_vector.h"
#include "orteaf/internal/diagnostics/error/error.h"

namespace orteaf::internal::runtime::base::pool {

template <typename Traits> class SlotPool {
public:
  using Payload = typename Traits::Payload;
  using Handle = typename Traits::Handle;
  using Request = typename Traits::Request;
  using Context = typename Traits::Context;
  using Config = typename Traits::Config;

  struct SlotRef {
    Handle handle{Handle::invalid()};
    Payload *payload_ptr{nullptr};

    bool valid() const noexcept {
      return handle.isValid() && payload_ptr != nullptr;
    }
  };

  SlotPool() = default;
  SlotPool(const SlotPool &) = delete;
  SlotPool &operator=(const SlotPool &) = delete;
  SlotPool(SlotPool &&) = default;
  SlotPool &operator=(SlotPool &&) = default;
  ~SlotPool() = default;

  void initialize(const Config &config) {
    const std::size_t capacity = config.capacity;
    if (capacity > static_cast<std::size_t>(Handle::invalid_index())) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
          "SlotPool capacity exceeds handle range");
    }

    payloads_.resize(capacity);
    generations_.resize(capacity, 0);
    created_.resize(capacity, 0);
    freelist_.clear();
    freelist_.reserve(capacity);
    for (std::size_t i = 0; i < capacity; ++i) {
      freelist_.pushBack(static_cast<index_type>(capacity - 1 - i));
    }
  }

  std::size_t capacity() const noexcept { return payloads_.size(); }
  std::size_t available() const noexcept { return freelist_.size(); }

  SlotRef acquire(const Request &request, const Context &context) {
    SlotRef ref = tryAcquire(request, context);
    if (!ref.valid()) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::OutOfRange,
          "SlotPool is empty");
    }
    return ref;
  }

  SlotRef tryAcquire(const Request &, const Context &) noexcept {
    if (freelist_.empty()) {
      return SlotRef{Handle::invalid(), nullptr};
    }
    const index_type idx = freelist_.back();
    freelist_.resize(freelist_.size() - 1);
    return SlotRef{makeHandle(idx), &payloads_[idx]};
  }

  bool release(Handle handle) noexcept {
    if (!isValid(handle)) {
      return false;
    }
    const std::size_t idx = static_cast<std::size_t>(handle.index);
    if constexpr (Handle::has_generation) {
      ++generations_[idx];
    }
    freelist_.pushBack(static_cast<index_type>(idx));
    return true;
  }

  Payload *get(Handle handle) noexcept {
    if (!isValid(handle)) {
      return nullptr;
    }
    return &payloads_[static_cast<std::size_t>(handle.index)];
  }

  const Payload *get(Handle handle) const noexcept {
    if (!isValid(handle)) {
      return nullptr;
    }
    return &payloads_[static_cast<std::size_t>(handle.index)];
  }

  bool isValid(Handle handle) const noexcept {
    const auto idx = static_cast<std::size_t>(handle.index);
    if (idx >= payloads_.size()) {
      return false;
    }
    if constexpr (Handle::has_generation) {
      return handle.generation == generations_[idx];
    }
    return true;
  }

  bool isCreated(Handle handle) const noexcept {
    if (!isValid(handle)) {
      return false;
    }
    return created_[static_cast<std::size_t>(handle.index)] != 0;
  }

  void setCreated(Handle handle, bool created) noexcept {
    if (!isValid(handle)) {
      return;
    }
    created_[static_cast<std::size_t>(handle.index)] = created ? 1 : 0;
  }

  bool emplace(Handle handle, const Request &request, const Context &context) {
    if (!isValid(handle) || isCreated(handle)) {
      return false;
    }
    auto &payload = payloads_[static_cast<std::size_t>(handle.index)];
    const bool created = Traits::create(payload, request, context);
    if (created) {
      setCreated(handle, true);
    }
    return created;
  }

  template <typename CreateFn>
    requires std::invocable<CreateFn, Payload &, const Request &,
                             const Context &> &&
             std::convertible_to<
                 std::invoke_result_t<CreateFn, Payload &, const Request &,
                                      const Context &>,
                 bool>
  bool emplace(Handle handle, const Request &request, const Context &context,
               CreateFn &&createFn) {
    if (!isValid(handle) || isCreated(handle)) {
      return false;
    }
    auto &payload = payloads_[static_cast<std::size_t>(handle.index)];
    const bool created =
        std::forward<CreateFn>(createFn)(payload, request, context);
    if (created) {
      setCreated(handle, true);
    }
    return created;
  }

  bool destroy(Handle handle, const Request &request, const Context &context) {
    if (!isValid(handle) || !isCreated(handle)) {
      return false;
    }
    auto &payload = payloads_[static_cast<std::size_t>(handle.index)];
    Traits::destroy(payload, request, context);
    setCreated(handle, false);
    return true;
  }

  template <typename DestroyFn>
    requires std::invocable<DestroyFn, Payload &, const Request &,
                             const Context &>
  bool destroy(Handle handle, const Request &request, const Context &context,
               DestroyFn &&destroyFn) {
    if (!isValid(handle) || !isCreated(handle)) {
      return false;
    }
    auto &payload = payloads_[static_cast<std::size_t>(handle.index)];
    if constexpr (std::convertible_to<
                      std::invoke_result_t<DestroyFn, Payload &, const Request &,
                                           const Context &>,
                      bool>) {
      if (!std::forward<DestroyFn>(destroyFn)(payload, request, context)) {
        return false;
      }
    } else {
      std::forward<DestroyFn>(destroyFn)(payload, request, context);
    }
    setCreated(handle, false);
    return true;
  }

private:
  using index_type = typename Handle::index_type;
  using generation_storage_t =
      std::conditional_t<Handle::has_generation, typename Handle::generation_type,
                         std::uint8_t>;

  Handle makeHandle(index_type idx) const noexcept {
    if constexpr (Handle::has_generation) {
      return Handle{idx, generations_[static_cast<std::size_t>(idx)]};
    }
    return Handle{idx};
  }

  ::orteaf::internal::base::HeapVector<Payload> payloads_{};
  ::orteaf::internal::base::HeapVector<generation_storage_t> generations_{};
  ::orteaf::internal::base::HeapVector<std::uint8_t> created_{};
  ::orteaf::internal::base::HeapVector<index_type> freelist_{};
};

} // namespace orteaf::internal::runtime::base::pool
