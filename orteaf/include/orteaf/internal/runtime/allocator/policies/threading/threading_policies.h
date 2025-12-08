#pragma once

#include <mutex>

#include "orteaf/internal/runtime/allocator/policies/policy_config.h"

namespace orteaf::internal::runtime::allocator::policies {

// Mutex-based threading policy for multi-threaded contexts.
class LockingThreadingPolicy {
public:
    template <typename Resource>
    using Config = PolicyConfig<Resource>;

    template <typename Resource>
    void initialize(const Config<Resource>&) {}

    void lock() { mutex_.lock(); }
    void unlock() { mutex_.unlock(); }

private:
    std::mutex mutex_;
};

// No-op threading policy for single-threaded contexts.
class NoLockThreadingPolicy {
public:
    template <typename Resource>
    using Config = PolicyConfig<Resource>;

    template <typename Resource>
    void initialize(const Config<Resource>&) {}

    void lock() {}
    void unlock() {}
};

}  // namespace orteaf::internal::runtime::allocator::policies
