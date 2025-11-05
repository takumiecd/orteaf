#pragma once

#include <atomic>
#include <string>
#include <sstream>
#include <iostream>

#ifdef ORTEAF_ENABLE_CUDA
#include <cuda.h>
#endif

namespace orteaf::internal::backend::cuda {

/**
 * @brief Statistics tracking for CUDA memory allocations and operations.
 *
 * This class tracks various statistics about CUDA memory allocations, deallocations,
 * and CUDA-specific operations (events, streams, device switches).
 * The available statistics depend on the `ORTEAF_STATS_LEVEL_CUDA_VALUE` compile-time setting:
 * - STATS_BASIC (2): Tracks total allocations, total deallocations, active allocations count, and device switches.
 * - STATS_EXTENDED (4): Additionally tracks current allocated bytes, peak allocated bytes, active events, and active streams.
 * - Disabled: Provides no-op methods when statistics are disabled.
 *
 * All statistics are thread-safe and use atomic operations with relaxed memory ordering.
 */
class CudaStats {
public:
    // STATS_BASIC(2) or STATS_EXTENDED(4) enabled
#ifdef ORTEAF_STATS_LEVEL_CUDA_VALUE
    #if ORTEAF_STATS_LEVEL_CUDA_VALUE <= 2
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
        
        /**
         * @brief Get the number of device switches.
         *
         * @return Total number of CUDA device context switches since initialization.
         */
        uint64_t device_switches() const noexcept {
            return device_switches_.load(std::memory_order_relaxed);
        }
    #endif
    
    #if ORTEAF_STATS_LEVEL_CUDA_VALUE <= 4
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
        
        /**
         * @brief Get the number of currently active events.
         *
         * @return Number of CUDA events that have been created but not yet destroyed.
         */
        uint64_t active_events() const noexcept {
            return active_events_.load(std::memory_order_relaxed);
        }
        
        /**
         * @brief Get the number of currently active streams.
         *
         * @return Number of CUDA streams that have been created but not yet destroyed.
         */
        uint64_t active_streams() const noexcept {
            return active_streams_.load(std::memory_order_relaxed);
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
        #if ORTEAF_STATS_LEVEL_CUDA_VALUE <= 2
            total_allocations_.fetch_add(1, std::memory_order_relaxed);
            active_allocations_.fetch_add(1, std::memory_order_relaxed);
        #endif
        #if ORTEAF_STATS_LEVEL_CUDA_VALUE <= 4
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
        #if ORTEAF_STATS_LEVEL_CUDA_VALUE <= 2
            total_deallocations_.fetch_add(1, std::memory_order_relaxed);
            active_allocations_.fetch_sub(1, std::memory_order_relaxed);
        #endif
        #if ORTEAF_STATS_LEVEL_CUDA_VALUE <= 4
            current_allocated_bytes_.fetch_sub(size, std::memory_order_relaxed);
        #endif
    }
    
    // CUDA-specific update methods
    #ifdef ORTEAF_STATS_LEVEL_CUDA_VALUE
        #if ORTEAF_STATS_LEVEL_CUDA_VALUE <= 2
            /**
             * @brief Update statistics when a device switch occurs.
             *
             * Increments the device switch counter.
             */
            void update_device_switch() noexcept {
                device_switches_.fetch_add(1, std::memory_order_relaxed);
            }
        #endif
        
        #if ORTEAF_STATS_LEVEL_CUDA_VALUE <= 4
            /**
             * @brief Update statistics when a CUDA event is created.
             *
             * Increments the active events counter.
             */
            void update_create_event() noexcept {
                active_events_.fetch_add(1, std::memory_order_relaxed);
            }
            
            /**
             * @brief Update statistics when a CUDA event is destroyed.
             *
             * Decrements the active events counter.
             */
            void update_destroy_event() noexcept {
                active_events_.fetch_sub(1, std::memory_order_relaxed);
            }
            
            /**
             * @brief Update statistics when a CUDA stream is created.
             *
             * Increments the active streams counter.
             */
            void update_create_stream() noexcept {
                active_streams_.fetch_add(1, std::memory_order_relaxed);
            }
            
            /**
             * @brief Update statistics when a CUDA stream is destroyed.
             *
             * Decrements the active streams counter.
             */
            void update_destroy_stream() noexcept {
                active_streams_.fetch_sub(1, std::memory_order_relaxed);
            }
            
            /**
             * @brief Update statistics when an event becomes active.
             *
             * Increments the active events counter.
             */
            void update_active_event() noexcept {
                active_events_.fetch_add(1, std::memory_order_relaxed);
            }
        #endif
    #endif
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
    
    /**
     * @brief No-op method when statistics are disabled.
     */
    void update_device_switch() noexcept {}
    
    /**
     * @brief No-op method when statistics are disabled.
     */
    void update_create_event() noexcept {}
    
    /**
     * @brief No-op method when statistics are disabled.
     */
    void update_destroy_event() noexcept {}
    
    /**
     * @brief No-op method when statistics are disabled.
     */
    void update_create_stream() noexcept {}
    
    /**
     * @brief No-op method when statistics are disabled.
     */
    void update_destroy_stream() noexcept {}
    
    /**
     * @brief No-op method when statistics are disabled.
     */
    void update_active_event() noexcept {}
#endif

    /**
     * @brief Convert statistics to a human-readable string representation.
     *
     * Returns a formatted string containing all available statistics based on
     * the configured statistics level. If statistics are disabled, returns "CUDA Stats: Disabled\n".
     *
     * @return String representation of the statistics.
     */
    std::string to_string() const {
        std::ostringstream oss;
#ifdef ORTEAF_STATS_LEVEL_CUDA_VALUE
        #if ORTEAF_STATS_LEVEL_CUDA_VALUE <= 2
            oss << "CUDA Stats:\n";
            oss << "  Total Allocations: " << total_allocations() << "\n";
            oss << "  Total Deallocations: " << total_deallocations() << "\n";
            oss << "  Active Allocations: " << active_allocations() << "\n";
            oss << "  Device Switches: " << device_switches() << "\n";
        #endif
        #if ORTEAF_STATS_LEVEL_CUDA_VALUE <= 4
            oss << "  Current Allocated Bytes: " << current_allocated_bytes() << "\n";
            oss << "  Peak Allocated Bytes: " << peak_allocated_bytes() << "\n";
            oss << "  Active Events: " << active_events() << "\n";
            oss << "  Active Streams: " << active_streams() << "\n";
        #endif
#else
        oss << "CUDA Stats: Disabled\n";
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
#ifdef ORTEAF_STATS_LEVEL_CUDA_VALUE
    #if ORTEAF_STATS_LEVEL_CUDA_VALUE <= 2
        std::atomic<uint64_t> total_allocations_{0};
        std::atomic<uint64_t> total_deallocations_{0};
        std::atomic<uint64_t> active_allocations_{0};
        std::atomic<uint64_t> device_switches_{0};
    #endif
    #if ORTEAF_STATS_LEVEL_CUDA_VALUE <= 4
        std::atomic<uint64_t> current_allocated_bytes_{0};
        std::atomic<uint64_t> peak_allocated_bytes_{0};
        std::atomic<uint64_t> active_events_{0};
        std::atomic<uint64_t> active_streams_{0};
    #endif
#endif
};

/**
 * @brief Stream output operator for CudaStats.
 *
 * Allows streaming CudaStats objects to output streams.
 *
 * @param os Output stream.
 * @param stats CudaStats object to output.
 * @return Reference to the output stream.
 */
inline std::ostream& operator<<(std::ostream& os, const CudaStats& stats) {
    return os << stats.to_string();
}

/**
 * @brief Get the singleton instance of CudaStats.
 *
 * Returns a reference to the global singleton instance of CudaStats.
 * The instance is created on first access and is thread-safe.
 *
 * @return Reference to the global CudaStats instance.
 */
inline CudaStats& stats_instance() {
    static CudaStats stats;
    return stats;
}

/**
 * @brief Get the global CUDA statistics instance.
 *
 * Convenience function that returns the same singleton instance as `stats_instance()`.
 *
 * @return Reference to the global CudaStats instance.
 */
inline CudaStats& cuda_stats() {
    return stats_instance();
}

/**
 * @brief Update statistics for a memory allocation.
 *
 * Global convenience function that updates the singleton CudaStats instance
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
 * Global convenience function that updates the singleton CudaStats instance
 * when memory is deallocated.
 *
 * @param size Size of the deallocated memory in bytes.
 */
inline void update_dealloc(size_t size) {
    stats_instance().update_dealloc(size);
}

/**
 * @brief Update statistics for a device switch.
 *
 * Global convenience function that updates the singleton CudaStats instance
 * when a CUDA device context switch occurs.
 */
inline void update_device_switch() {
    stats_instance().update_device_switch();
}

/**
 * @brief Update statistics when a CUDA event is created.
 *
 * Global convenience function that updates the singleton CudaStats instance
 * when a CUDA event is created.
 */
inline void update_create_event() {
    stats_instance().update_create_event();
}

/**
 * @brief Update statistics when a CUDA event is destroyed.
 *
 * Global convenience function that updates the singleton CudaStats instance
 * when a CUDA event is destroyed.
 */
inline void update_destroy_event() {
    stats_instance().update_destroy_event();
}

/**
 * @brief Update statistics when a CUDA stream is created.
 *
 * Global convenience function that updates the singleton CudaStats instance
 * when a CUDA stream is created.
 */
inline void update_create_stream() {
    stats_instance().update_create_stream();
}

/**
 * @brief Update statistics when a CUDA stream is destroyed.
 *
 * Global convenience function that updates the singleton CudaStats instance
 * when a CUDA stream is destroyed.
 */
inline void update_destroy_stream() {
    stats_instance().update_destroy_stream();
}

/**
 * @brief Update statistics when an event becomes active.
 *
 * Global convenience function that updates the singleton CudaStats instance
 * when an event becomes active.
 */
inline void update_active_event() {
    stats_instance().update_active_event();
}

// Legacy function names for backward compatibility
// These functions delegate to the new update_* functions
/**
 * @brief Legacy function name for update_alloc.
 * @deprecated Use update_alloc() instead.
 */
inline void stats_on_alloc(size_t bytes) {
    update_alloc(bytes);
}

/**
 * @brief Legacy function name for update_dealloc.
 * @deprecated Use update_dealloc() instead.
 */
inline void stats_on_dealloc(size_t bytes) {
    update_dealloc(bytes);
}

/**
 * @brief Legacy function name for update_create_event.
 * @deprecated Use update_create_event() instead.
 */
inline void stats_on_create_event() {
    update_create_event();
}

/**
 * @brief Legacy function name for update_destroy_event.
 * @deprecated Use update_destroy_event() instead.
 */
inline void stats_on_destroy_event() {
    update_destroy_event();
}

/**
 * @brief Legacy function name for update_create_stream.
 * @deprecated Use update_create_stream() instead.
 */
inline void stats_on_create_stream() {
    update_create_stream();
}

/**
 * @brief Legacy function name for update_destroy_stream.
 * @deprecated Use update_destroy_stream() instead.
 */
inline void stats_on_destroy_stream() {
    update_destroy_stream();
}

/**
 * @brief Legacy function name for update_device_switch.
 * @deprecated Use update_device_switch() instead.
 */
inline void stats_on_device_switch() {
    update_device_switch();
}

/**
 * @brief Legacy function name for update_active_event.
 * @deprecated Use update_active_event() instead.
 */
inline void stats_on_active_event() {
    update_active_event();
}

} // namespace orteaf::internal::backend::cuda
