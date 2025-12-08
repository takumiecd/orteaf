#pragma once

#include <atomic>
#include <cstdint>
#include <iostream>
#include <orteaf/internal/backend/backend.h>
#include <sstream>

namespace orteaf::internal::runtime::allocator::pool {

template <::orteaf::internal::backend::Backend BackendType>
class SegregatePoolStats {
private:
  // Determine statistics level based on backend type
  static constexpr int StatsLevel = []() {
    if constexpr (BackendType == ::orteaf::internal::backend::Backend::Cpu) {
#ifdef ORTEAF_STATS_LEVEL_CPU_VALUE
      return ORTEAF_STATS_LEVEL_CPU_VALUE;
#else
      return 0;
#endif
    } else if constexpr (BackendType ==
                         ::orteaf::internal::backend::Backend::Mps) {
#ifdef ORTEAF_STATS_LEVEL_MPS_VALUE
      return ORTEAF_STATS_LEVEL_MPS_VALUE;
#else
      return 0;
#endif
    } else if constexpr (BackendType ==
                         ::orteaf::internal::backend::Backend::Cuda) {
#ifdef ORTEAF_STATS_LEVEL_CUDA_VALUE
      return ORTEAF_STATS_LEVEL_CUDA_VALUE;
#else
      return 0;
#endif
    } else {
#ifdef ORTEAF_STATS_LEVEL_GLOBAL_VALUE
      return ORTEAF_STATS_LEVEL_GLOBAL_VALUE;
#else
      return 0;
#endif
    }
  }();

public:
  uint64_t totalAllocations() const noexcept {
    if constexpr (StatsLevel >= 2) {
      return total_allocations_.load(std::memory_order_relaxed);
    }
    return 0;
  }

  uint64_t totalDeallocations() const noexcept {
    if constexpr (StatsLevel >= 2) {
      return total_deallocations_.load(std::memory_order_relaxed);
    }
    return 0;
  }

  uint64_t activeAllocations() const noexcept {
    if constexpr (StatsLevel >= 2) {
      return active_allocations_.load(std::memory_order_relaxed);
    }
    return 0;
  }

  uint64_t currentAllocatedBytes() const noexcept {
    if constexpr (StatsLevel >= 4) {
      return current_allocated_bytes_.load(std::memory_order_relaxed);
    }
    return 0;
  }

  uint64_t peakAllocatedBytes() const noexcept {
    if constexpr (StatsLevel >= 4) {
      return peak_allocated_bytes_.load(std::memory_order_relaxed);
    }
    return 0;
  }

  uint64_t largeAllocations() const noexcept {
    if constexpr (StatsLevel >= 2) {
      return large_allocations_.load(std::memory_order_relaxed);
    }
    return 0;
  }

  uint64_t poolExpansions() const noexcept {
    if constexpr (StatsLevel >= 2) {
      return pool_expansions_.load(std::memory_order_relaxed);
    }
    return 0;
  }

  void updateAlloc(std::size_t size, bool is_large) noexcept {
    if constexpr (StatsLevel >= 2) {
      total_allocations_.fetch_add(1, std::memory_order_relaxed);
      active_allocations_.fetch_add(1, std::memory_order_relaxed);
      if (is_large) {
        large_allocations_.fetch_add(1, std::memory_order_relaxed);
      }
    }
    if constexpr (StatsLevel >= 4) {
      uint64_t current =
          current_allocated_bytes_.fetch_add(size, std::memory_order_relaxed) +
          size;
      uint64_t peak = peak_allocated_bytes_.load(std::memory_order_relaxed);
      while (current > peak && !peak_allocated_bytes_.compare_exchange_weak(
                                   peak, current, std::memory_order_relaxed)) {
        peak = peak_allocated_bytes_.load(std::memory_order_relaxed);
      }
    }
  }

  void updateDealloc(std::size_t size) noexcept {
    if constexpr (StatsLevel >= 2) {
      total_deallocations_.fetch_add(1, std::memory_order_relaxed);
      active_allocations_.fetch_sub(1, std::memory_order_relaxed);
    }
    if constexpr (StatsLevel >= 4) {
      current_allocated_bytes_.fetch_sub(size, std::memory_order_relaxed);
    }
  }

  void updateExpansion() noexcept {
    if constexpr (StatsLevel >= 2) {
      pool_expansions_.fetch_add(1, std::memory_order_relaxed);
    }
  }

  std::string toString() const {
    std::ostringstream oss;
    if constexpr (StatsLevel <= 0) {
      oss << "SegregatePool Stats: Disabled\n";
    } else {
      oss << "SegregatePool Stats:\n";
      if constexpr (StatsLevel >= 2) {
        oss << "  Total Allocations: " << totalAllocations() << "\n";
        oss << "  Total Deallocations: " << totalDeallocations() << "\n";
        oss << "  Active Allocations: " << activeAllocations() << "\n";
        oss << "  Large Allocations: " << largeAllocations() << "\n";
        oss << "  Pool Expansions: " << poolExpansions() << "\n";
      }
      if constexpr (StatsLevel >= 4) {
        oss << "  Current Allocated Bytes: " << currentAllocatedBytes() << "\n";
        oss << "  Peak Allocated Bytes: " << peakAllocatedBytes() << "\n";
      }
    }
    return oss.str();
  }

  void print() const { std::cout << toString(); }

private:
  std::atomic<uint64_t> total_allocations_{0};
  std::atomic<uint64_t> total_deallocations_{0};
  std::atomic<uint64_t> active_allocations_{0};
  std::atomic<uint64_t> large_allocations_{0};
  std::atomic<uint64_t> pool_expansions_{0};
  std::atomic<uint64_t> current_allocated_bytes_{0};
  std::atomic<uint64_t> peak_allocated_bytes_{0};
};

} // namespace orteaf::internal::runtime::allocator::pool
