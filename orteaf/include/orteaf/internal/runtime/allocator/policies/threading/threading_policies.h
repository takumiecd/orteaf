#pragma once

#include <mutex>

namespace orteaf::internal::runtime::allocator::policies {

// Mutex-based threading policy for multi-threaded contexts.
class LockingThreadingPolicy {
public:
    void lock() { mutex_.lock(); }
    void unlock() { mutex_.unlock(); }

private:
    std::mutex mutex_;
};

// No-op threading policy for single-threaded contexts.
class NoLockThreadingPolicy {
public:
    void lock() {}
    void unlock() {}
};

}  // namespace orteaf::internal::runtime::allocator::policies
