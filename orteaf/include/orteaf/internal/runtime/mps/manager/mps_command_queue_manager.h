#pragma once

#if ORTEAF_ENABLE_MPS

#include <cstddef>
#include <cstdint>
#include <mutex>

#include <orteaf/internal/base/handle.h>
#include <orteaf/internal/runtime/base/lease/control_block/lockable_shared.h>
#include <orteaf/internal/runtime/base/lease/lockable_shared_lease.h>
#include <orteaf/internal/runtime/base/lease/slot.h>
#include <orteaf/internal/runtime/base/manager/base_manager_core.h>
#include <orteaf/internal/runtime/mps/platform/mps_slow_ops.h>
#include <orteaf/internal/runtime/mps/platform/wrapper/mps_command_queue.h>

namespace orteaf::internal::runtime::mps::manager {

// Slot type
using CommandQueueSlot = ::orteaf::internal::runtime::base::RawSlot<
    ::orteaf::internal::runtime::mps::platform::wrapper::MpsCommandQueue_t>;

// Control block: LockableShared (shared ownership + mutex lock)
using CommandQueueControlBlock =
    ::orteaf::internal::runtime::base::LockableSharedControlBlock<
        CommandQueueSlot>;

struct MpsCommandQueueManagerTraits {
  using ControlBlock = CommandQueueControlBlock;
  using Handle = ::orteaf::internal::base::CommandQueueHandle;
  static constexpr const char *Name = "MPS command queue manager";
};

class MpsCommandQueueManager
    : protected ::orteaf::internal::runtime::base::BaseManagerCore<
          MpsCommandQueueManagerTraits> {
  using Base = ::orteaf::internal::runtime::base::BaseManagerCore<
      MpsCommandQueueManagerTraits>;

public:
  using SlowOps = ::orteaf::internal::runtime::mps::platform::MpsSlowOps;
  using DeviceType =
      ::orteaf::internal::runtime::mps::platform::wrapper::MpsDevice_t;
  using CommandQueueHandle = ::orteaf::internal::base::CommandQueueHandle;
  using CommandQueueType =
      ::orteaf::internal::runtime::mps::platform::wrapper::MpsCommandQueue_t;

  // Lease Type: LockableSharedLease
  using CommandQueueLease =
      ::orteaf::internal::runtime::base::LockableSharedLease<
          CommandQueueHandle, CommandQueueType, MpsCommandQueueManager>;

private:
  // Friend declaration for Lease copy constructor access
  friend CommandQueueLease;

public:
  // =========================================================================
  // ScopedLock - RAII wrapper with payload access
  // =========================================================================

  /// @brief RAII lock guard that provides access to the command queue payload
  class ScopedLock {
  public:
    ScopedLock() = default;

    ScopedLock(std::unique_lock<std::mutex> lock, CommandQueueType &payload)
        : lock_(std::move(lock)), payload_(&payload) {}

    ScopedLock(const ScopedLock &) = delete;
    ScopedLock &operator=(const ScopedLock &) = delete;

    ScopedLock(ScopedLock &&other) noexcept
        : lock_(std::move(other.lock_)), payload_(other.payload_) {
      other.payload_ = nullptr;
    }

    ScopedLock &operator=(ScopedLock &&other) noexcept {
      if (this != &other) {
        lock_ = std::move(other.lock_);
        payload_ = other.payload_;
        other.payload_ = nullptr;
      }
      return *this;
    }

    ~ScopedLock() = default;

    // Accessors - only valid when lock is held
    CommandQueueType &operator*() const noexcept { return *payload_; }
    CommandQueueType *operator->() const noexcept { return payload_; }

    /// @brief Check if lock was successfully acquired
    explicit operator bool() const noexcept {
      return lock_.owns_lock() && payload_ != nullptr;
    }

    bool isValid() const noexcept { return static_cast<bool>(*this); }

  private:
    std::unique_lock<std::mutex> lock_;
    CommandQueueType *payload_{nullptr};
  };

  MpsCommandQueueManager() = default;
  MpsCommandQueueManager(const MpsCommandQueueManager &) = delete;
  MpsCommandQueueManager &operator=(const MpsCommandQueueManager &) = delete;
  MpsCommandQueueManager(MpsCommandQueueManager &&) = default;
  MpsCommandQueueManager &operator=(MpsCommandQueueManager &&) = default;
  ~MpsCommandQueueManager() = default;

  void initialize(DeviceType device, SlowOps *ops, std::size_t capacity);
  void shutdown();
  void growCapacity(std::size_t additional);

  // Acquire returns a LockableSharedLease (opaque handle)
  CommandQueueLease acquire();

  // Release logic
  void release(CommandQueueLease &lease) noexcept;

  // =========================================================================
  // Locking API (for Lease to call)
  // =========================================================================

  /// @brief Acquire exclusive lock (blocking)
  /// @return ScopedLock with payload access
  ScopedLock lock(const CommandQueueLease &lease);

  /// @brief Try to acquire exclusive lock (non-blocking)
  /// @return ScopedLock (check with operator bool)
  ScopedLock tryLock(const CommandQueueLease &lease);

  // Config
  using Base::growthChunkSize;
  using Base::setGrowthChunkSize;

  // Expose capacity
  using Base::capacity;
  using Base::isAlive;
  using Base::isInitialized;

#if ORTEAF_ENABLE_TEST
  using Base::controlBlockForTest;
  using Base::freeListSizeForTest;
  using Base::isInitializedForTest;
#endif

private:
  using Base::acquireExisting;

  void destroyResource(CommandQueueType &resource);

  DeviceType device_{nullptr};
  SlowOps *ops_{nullptr};
};

} // namespace orteaf::internal::runtime::mps::manager

#endif // ORTEAF_ENABLE_MPS
