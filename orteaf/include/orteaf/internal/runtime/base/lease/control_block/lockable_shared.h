#pragma once

#include <concepts>
#include <cstdint>
#include <mutex>
#include <utility>

#include <orteaf/internal/runtime/base/lease/control_block/shared.h>

namespace orteaf::internal::runtime::base {

/// @brief Lockable shared control block - shared ownership with mutex lock
/// @details Extends SharedControlBlock with a mutex for exclusive access.
///          Supports shared ownership (ref counting) and mutex-based locking.
///          canShutdown() returns true when count == 0.
template <typename SlotT>
  requires SlotConcept<SlotT>
class LockableSharedControlBlock : public SharedControlBlock<SlotT> {
public:
  using Base = SharedControlBlock<SlotT>;
  using Category = typename Base::Category;
  using Slot = typename Base::Slot;
  using Payload = typename Base::Payload;

  // Constructor / Assignment
  LockableSharedControlBlock() = default;
  LockableSharedControlBlock(const LockableSharedControlBlock &) = delete;
  LockableSharedControlBlock &
  operator=(const LockableSharedControlBlock &) = delete;

  // Note: std::mutex is not movable, so we must handle move specially
  LockableSharedControlBlock(LockableSharedControlBlock &&other) noexcept
      : Base(std::move(other)) {
    // mutex_ is default-initialized (unlocked state)
    // The moved-from object should not be used anymore
  }

  LockableSharedControlBlock &
  operator=(LockableSharedControlBlock &&other) noexcept {
    if (this != &other) {
      Base::operator=(std::move(other));
      // mutex_ remains in its current state (should be unlocked)
    }
    return *this;
  }

  // =========================================================================
  // Locking API
  // =========================================================================

  /// @brief Acquire exclusive lock (blocking)
  /// @return unique_lock holding the mutex
  std::unique_lock<std::mutex> lock() {
    return std::unique_lock<std::mutex>(mutex_);
  }

  /// @brief Try to acquire exclusive lock (non-blocking)
  /// @return unique_lock that may or may not own the lock
  std::unique_lock<std::mutex> tryLock() {
    return std::unique_lock<std::mutex>(mutex_, std::try_to_lock);
  }

  /// @brief Get reference to the mutex (for advanced use)
  std::mutex &mutex() noexcept { return mutex_; }

  // =========================================================================
  // Lifecycle Overrides
  // =========================================================================

  /// @brief Check if shutdown is allowed
  /// @return true if strong count is 0
  bool canShutdown() const noexcept { return this->count() == 0; }

private:
  std::mutex mutex_;
};

} // namespace orteaf::internal::runtime::base
