#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>

#include "orteaf/internal/runtime/allocator/policies/policy_config.h"

namespace orteaf::internal::runtime::allocator::policies {

// Fast path: round up to next power-of-two size class.
class FastFreePolicy {
public:
    template <typename Resource>
    using Config = PolicyConfig<Resource>;

    template <typename Resource>
    void initialize(const Config<Resource>&) {}

    void error() {}

    std::size_t get_block_size(std::size_t min_block_size, std::size_t size_bytes) const {
        return std::bit_ceil(std::max(min_block_size, size_bytes));
    }
};

// Placeholder for a safety-oriented strategy (to be defined per allocator needs).
class SafeFreePolicy {
public:
    template <typename Resource>
    using Config = PolicyConfig<Resource>;

    template <typename Resource>
    void initialize(const Config<Resource>&) {}

    void error() {}

    std::size_t get_block_size(std::size_t /*min_block_size*/, std::size_t /*size_bytes*/) const {
        return 0;
    }
};

}  // namespace orteaf::internal::runtime::allocator::policies
