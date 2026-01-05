#pragma once

#include <cstddef>
#include <utility>

#include "orteaf/internal/base/heap_vector.h"

namespace orteaf::internal::base::manager {

template <typename PayloadHandleT, typename StrongLeaseT>
  requires requires(PayloadHandleT handle) {
    handle.isValid();
    handle.index;
  }
class LeaseLifetimeRegistry {
public:
  using PayloadHandle = PayloadHandleT;
  using StrongLease = StrongLeaseT;

  LeaseLifetimeRegistry() = default;
  LeaseLifetimeRegistry(const LeaseLifetimeRegistry &) = delete;
  LeaseLifetimeRegistry &operator=(const LeaseLifetimeRegistry &) = delete;
  LeaseLifetimeRegistry(LeaseLifetimeRegistry &&) noexcept = default;
  LeaseLifetimeRegistry &operator=(LeaseLifetimeRegistry &&) noexcept = default;
  ~LeaseLifetimeRegistry() = default;

  bool set(StrongLease lease)
    requires requires(StrongLease &lease_ref) { lease_ref.payloadHandle(); }
  {
    if (!lease) {
      return false;
    }
    const auto handle = lease.payloadHandle();
    if (!handle.isValid()) {
      return false;
    }
    ensureSlot(handle);
    auto &entry = entries_[indexOf(handle)];
    if (entry.lease) {
      entry.lease.release();
    } else {
      ++size_;
    }
    entry.lease = std::move(lease);
    return true;
  }

  StrongLease get(PayloadHandle handle) const noexcept {
    const auto *entry = findEntry(handle);
    return entry ? entry->lease : StrongLease{};
  }

  bool has(PayloadHandle handle) const noexcept {
    return findEntry(handle) != nullptr;
  }

  void release(PayloadHandle handle) noexcept {
    auto *entry = findEntry(handle);
    if (!entry) {
      return;
    }
    entry->lease.release();
    --size_;
  }

  void clear() noexcept {
    for (auto &entry : entries_) {
      if (entry.lease) {
        entry.lease.release();
      }
    }
    size_ = 0;
  }

  std::size_t size() const noexcept { return size_; }
  bool empty() const noexcept { return size_ == 0; }

  template <typename Func>
  std::size_t forEachActiveLease(Func &&func) {
    std::size_t visited = 0;
    for (auto &entry : entries_) {
      if (!entry.lease) {
        continue;
      }
      ++visited;
      std::forward<Func>(func)(entry.lease);
    }
    return visited;
  }

private:
  struct Entry {
    StrongLease lease{};
  };

  static std::size_t indexOf(PayloadHandle handle) noexcept {
    return static_cast<std::size_t>(handle.index);
  }

  void ensureSlot(PayloadHandle handle) {
    const auto index = indexOf(handle);
    if (index >= entries_.size()) {
      entries_.resize(index + 1);
    }
  }

  Entry *findEntry(PayloadHandle handle) noexcept {
    if (!handle.isValid()) {
      return nullptr;
    }
    const auto index = indexOf(handle);
    if (index >= entries_.size()) {
      return nullptr;
    }
    auto &entry = entries_[index];
    if (!entry.lease) {
      return nullptr;
    }
    return &entry;
  }

  const Entry *findEntry(PayloadHandle handle) const noexcept {
    if (!handle.isValid()) {
      return nullptr;
    }
    const auto index = indexOf(handle);
    if (index >= entries_.size()) {
      return nullptr;
    }
    const auto &entry = entries_[index];
    if (!entry.lease) {
      return nullptr;
    }
    return &entry;
  }

  ::orteaf::internal::base::HeapVector<Entry> entries_{};
  std::size_t size_{0};
};

} // namespace orteaf::internal::base::manager
