#pragma once

#include <atomic>
#include <string>
#include <sstream>
#include <iostream>

namespace orteaf::internal::backend::cpu {

/**
 * @brief Statistics tracking for CPU memory allocations.
 *
 * This class tracks various statistics about CPU memory allocations and deallocations.
 * The available statistics depend on the `ORTEAF_STATS_LEVEL_CPU_VALUE` compile-time setting:
 * - STATS_BASIC (2): Tracks total allocations, total deallocations, and active allocations count.
 * - STATS_EXTENDED (4): Additionally tracks current allocated bytes and peak allocated bytes.
 * - Disabled: Provides no-op methods when statistics are disabled.
 *
 * All statistics are thread-safe and use atomic operations with relaxed memory ordering.
 */
class CpuStats {
public:
    // STATS_BASIC(2) or STATS_EXTENDED(4) enabled
#ifdef ORTEAF_STATS_LEVEL_CPU_VALUE
    #if ORTEAF_STATS_LEVEL_CPU_VALUE <= 2
        // Getter methods for basic stats
        /**
         * @brief Get the total number of allocations performed.
         *
         * @return Total number of memory allocations since initialization.
         */
        uint64_t total_allocations() const noexcept {
            return total_allocations_.load(std::memory_order_relaxed);
        }
        
        /**
         * @brief Get the total number of deallocations performed.
         *
         * @return Total number of memory deallocations since initialization.
         */
        uint64_t total_deallocations() const noexcept {
            return total_deallocations_.load(std::memory_order_relaxed);
        }
        
        /**
         * @brief Get the number of currently active allocations.
         *
         * This is the difference between total allocations and total deallocations.
         *
         * @return Number of allocations that have not been deallocated.
         */
        uint64_t active_allocations() const noexcept {
            return active_allocations_.load(std::memory_order_relaxed);
        }
    #endif
    
    #if ORTEAF_STATS_LEVEL_CPU_VALUE <= 4
        // Getter methods for extended stats
        /**
         * @brief Get the current number of allocated bytes.
         *
         * @return Total number of bytes currently allocated (not yet deallocated).
         */
        uint64_t current_allocated_bytes() const noexcept {
            return current_allocated_bytes_.load(std::memory_order_relaxed);
        }
        
        /**
         * @brief Get the peak number of allocated bytes.
         *
         * This represents the maximum number of bytes that were simultaneously allocated
         * at any point since initialization.
         *
         * @return Peak number of allocated bytes.
         */
        uint64_t peak_allocated_bytes() const noexcept {
            return peak_allocated_bytes_.load(std::memory_order_relaxed);
        }
    #endif
    
    // Update methods (handles both BASIC and EXTENDED levels)
    /**
     * @brief Update statistics when memory is allocated.
     *
     * Updates allocation counts and byte tracking based on the configured statistics level.
     * Thread-safe operation using atomic operations.
     *
     * @param size Size of the allocated memory in bytes.
     */
    void update_alloc(size_t size) noexcept {
        #if ORTEAF_STATS_LEVEL_CPU_VALUE <= 2
            total_allocations_.fetch_add(1, std::memory_order_relaxed);
            active_allocations_.fetch_add(1, std::memory_order_relaxed);
        #endif
        #if ORTEAF_STATS_LEVEL_CPU_VALUE <= 4
            uint64_t current = current_allocated_bytes_.fetch_add(size, std::memory_order_relaxed) + size;
            uint64_t peak = peak_allocated_bytes_.load(std::memory_order_relaxed);
            while (current > peak && !peak_allocated_bytes_.compare_exchange_weak(peak, current, std::memory_order_relaxed)) {
                peak = peak_allocated_bytes_.load(std::memory_order_relaxed);
            }
        #endif
    }
    
    /**
     * @brief Update statistics when memory is deallocated.
     *
     * Updates deallocation counts and byte tracking based on the configured statistics level.
     * Thread-safe operation using atomic operations.
     *
     * @param size Size of the deallocated memory in bytes.
     */
    void update_dealloc(size_t size) noexcept {
        #if ORTEAF_STATS_LEVEL_CPU_VALUE <= 2
            total_deallocations_.fetch_add(1, std::memory_order_relaxed);
            active_allocations_.fetch_sub(1, std::memory_order_relaxed);
        #endif
        #if ORTEAF_STATS_LEVEL_CPU_VALUE <= 4
            current_allocated_bytes_.fetch_sub(size, std::memory_order_relaxed);
        #endif
    }
#else
    // When stats are disabled, provide no-op methods
    /**
     * @brief No-op method when statistics are disabled.
     *
     * @param size Ignored.
     */
    void update_alloc(size_t) noexcept {}
    
    /**
     * @brief No-op method when statistics are disabled.
     *
     * @param size Ignored.
     */
    void update_dealloc(size_t) noexcept {}
#endif

    /**
     * @brief Convert statistics to a human-readable string representation.
     *
     * Returns a formatted string containing all available statistics based on
     * the configured statistics level. If statistics are disabled, returns "CPU Stats: Disabled\n".
     *
     * @return String representation of the statistics.
     */
    std::string to_string() const {
        std::ostringstream oss;
#ifdef ORTEAF_STATS_LEVEL_CPU_VALUE
        #if ORTEAF_STATS_LEVEL_CPU_VALUE <= 2
            oss << "CPU Stats:\n";
            oss << "  Total Allocations: " << total_allocations() << "\n";
            oss << "  Total Deallocations: " << total_deallocations() << "\n";
            oss << "  Active Allocations: " << active_allocations() << "\n";
        #endif
        #if ORTEAF_STATS_LEVEL_CPU_VALUE <= 4
            oss << "  Current Allocated Bytes: " << current_allocated_bytes() << "\n";
            oss << "  Peak Allocated Bytes: " << peak_allocated_bytes() << "\n";
        #endif
#else
        oss << "CPU Stats: Disabled\n";
#endif
        return oss.str();
    }
    
    /**
     * @brief Print statistics to standard output.
     *
     * Equivalent to `std::cout << to_string()`.
     */
    void print() const {
        std::cout << to_string();
    }

private:
#ifdef ORTEAF_STATS_LEVEL_CPU_VALUE
    #if ORTEAF_STATS_LEVEL_CPU_VALUE <= 2
        std::atomic<uint64_t> total_allocations_{0};
        std::atomic<uint64_t> total_deallocations_{0};
        std::atomic<uint64_t> active_allocations_{0};
    #endif
    #if ORTEAF_STATS_LEVEL_CPU_VALUE <= 4
        std::atomic<uint64_t> current_allocated_bytes_{0};
        std::atomic<uint64_t> peak_allocated_bytes_{0};
    #endif
#endif
};

/**
 * @brief Stream output operator for CpuStats.
 *
 * Allows streaming CpuStats objects to output streams.
 *
 * @param os Output stream.
 * @param stats CpuStats object to output.
 * @return Reference to the output stream.
 */
inline std::ostream& operator<<(std::ostream& os, const CpuStats& stats) {
    return os << stats.to_string();
}

/**
 * @brief Get the singleton instance of CpuStats.
 *
 * Returns a reference to the global singleton instance of CpuStats.
 * The instance is created on first access and is thread-safe.
 *
 * @return Reference to the global CpuStats instance.
 */
inline CpuStats& stats_instance() {
    static CpuStats stats;
    return stats;
}

/**
 * @brief Get the global CPU statistics instance.
 *
 * Convenience function that returns the same singleton instance as `stats_instance()`.
 *
 * @return Reference to the global CpuStats instance.
 */
inline CpuStats& cpu_stats() {
    return stats_instance();
}

/**
 * @brief Update statistics for a memory allocation.
 *
 * Global convenience function that updates the singleton CpuStats instance
 * when memory is allocated.
 *
 * @param size Size of the allocated memory in bytes.
 */
inline void update_alloc(size_t size) {
    stats_instance().update_alloc(size);
}

/**
 * @brief Update statistics for a memory deallocation.
 *
 * Global convenience function that updates the singleton CpuStats instance
 * when memory is deallocated.
 *
 * @param size Size of the deallocated memory in bytes.
 */
inline void update_dealloc(size_t size) {
    stats_instance().update_dealloc(size);
}

} // namespace orteaf::internal::backend::cpu