#pragma once

#include <type_traits>

#include <orteaf/internal/base/handle.h>
#include <orteaf/internal/diagnostics/error/error.h>
#include <orteaf/internal/runtime/base/lease/category.h>

namespace orteaf::internal::runtime::base {

/// @brief Lease for lockable shared resources
/// @details Holds a reference to a shared resource but does NOT provide direct
/// access. Access must be obtained via lock()/tryLock() (returns Manager's
/// ScopedLock) or accessConcurrent() (explicit unsafe if manager provides).
template <typename HandleT, typename PayloadT, typename ManagerT>
class LockableSharedLease {
public:
  using Handle = HandleT;
  using Payload = PayloadT;
  using Manager = ManagerT;
  using CompatibleCategory = lease_category::Shared;

  // ScopedLock is defined by the Manager, not by the Lease
  using ScopedLock = typename ManagerT::ScopedLock;

  // =========================================================================
  // Constructors
  // =========================================================================

  LockableSharedLease() = default;

  LockableSharedLease(HandleT handle, ManagerT *manager)
      : handle_(handle), manager_(manager) {}

  LockableSharedLease(const LockableSharedLease &other)
      : handle_(other.handle_), manager_(other.manager_) {
    if (isValid()) {
      manager_->acquireExisting(handle_);
    }
  }

  LockableSharedLease(LockableSharedLease &&other) noexcept
      : handle_(other.handle_), manager_(other.manager_) {
    other.handle_ = HandleT::invalid();
    other.manager_ = nullptr;
  }

  LockableSharedLease &operator=(const LockableSharedLease &other) {
    if (this != &other) {
      release();
      handle_ = other.handle_;
      manager_ = other.manager_;
      if (isValid()) {
        manager_->acquireExisting(handle_);
      }
    }
    return *this;
  }

  LockableSharedLease &operator=(LockableSharedLease &&other) noexcept {
    if (this != &other) {
      release();
      handle_ = other.handle_;
      manager_ = other.manager_;
      other.handle_ = HandleT::invalid();
      other.manager_ = nullptr;
    }
    return *this;
  }

  ~LockableSharedLease() { release(); }

  // =========================================================================
  // Access API
  // =========================================================================

  /// @brief Acquire exclusive lock (blocking)
  /// @return Manager's ScopedLock with payload access
  ScopedLock lock() {
    if (!isValid()) {
      return ScopedLock{}; // Invalid
    }
    return manager_->lock(*this);
  }

  /// @brief Try to acquire exclusive lock (non-blocking)
  /// @return Manager's ScopedLock (check with operator bool)
  ScopedLock tryLock() {
    if (!isValid()) {
      return ScopedLock{}; // Invalid
    }
    return manager_->tryLock(*this);
  }

  // =========================================================================
  // Lifecycle API
  // =========================================================================

  void release() {
    if (isValid() && manager_) {
      manager_->release(*this);
      handle_ = HandleT::invalid();
      manager_ = nullptr;
    }
  }

  void invalidate() noexcept {
    handle_ = HandleT::invalid();
    manager_ = nullptr;
  }

  bool isValid() const noexcept {
    return manager_ != nullptr && handle_.isValid();
  }

  explicit operator bool() const noexcept { return isValid(); }

  HandleT handle() const noexcept { return handle_; }

#if ORTEAF_ENABLE_TEST
  /// @brief Factory for testing - creates lease without manager
  static LockableSharedLease makeForTest(HandleT handle, Payload payload) {
    (void)payload;
    LockableSharedLease lease;
    lease.handle_ = handle;
    lease.manager_ = nullptr;
    return lease;
  }
#endif

private:
  HandleT handle_{HandleT::invalid()};
  ManagerT *manager_{nullptr};
};

} // namespace orteaf::internal::runtime::base
