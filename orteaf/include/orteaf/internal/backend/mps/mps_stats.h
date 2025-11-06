/**
 * @file mps_stats.h
 * @brief MPS runtime statistics counters and update helpers.
 */
#pragma once

#include <atomic>
#include <string>
#include <sstream>
#include <iostream>

namespace orteaf::internal::backend::mps {

/**
 * @brief Aggregates allocation and object-lifetime statistics for MPS.
 *
 * Build-time macro `ORTEAF_STATS_LEVEL_MPS_VALUE` controls which counters are
 * available: 2 = basic, 4 = extended; undefined = all updates are no-ops.
 *
 * All public methods are always declared, ensuring a consistent interface
 * regardless of the statistics level configuration.
 */
class MpsStats {
public:
    // Getter methods - always defined, return values based on stats level
    /**
     * @brief Get the total number of allocations performed.
     *
     * @return Total number of memory allocations since initialization (0 if stats disabled).
     */
    uint64_t total_allocations() const noexcept {
        #ifdef ORTEAF_STATS_LEVEL_MPS_VALUE
            #if ORTEAF_STATS_LEVEL_MPS_VALUE <= 2
                return total_allocations_.load(std::memory_order_relaxed);
            #endif
        #endif
        return 0;
    }
    
    /**
     * @brief Get the total number of deallocations performed.
     *
     * @return Total number of memory deallocations since initialization (0 if stats disabled).
     */
    uint64_t total_deallocations() const noexcept {
        #ifdef ORTEAF_STATS_LEVEL_MPS_VALUE
            #if ORTEAF_STATS_LEVEL_MPS_VALUE <= 2
                return total_deallocations_.load(std::memory_order_relaxed);
            #endif
        #endif
        return 0;
    }
    
    /**
     * @brief Get the number of currently active allocations.
     *
     * This is the difference between total allocations and total deallocations.
     *
     * @return Number of allocations that have not been deallocated (0 if stats disabled).
     */
    uint64_t active_allocations() const noexcept {
        #ifdef ORTEAF_STATS_LEVEL_MPS_VALUE
            #if ORTEAF_STATS_LEVEL_MPS_VALUE <= 2
                return active_allocations_.load(std::memory_order_relaxed);
            #endif
        #endif
        return 0;
    }
    
    /**
     * @brief Get the current number of allocated bytes.
     *
     * @return Total number of bytes currently allocated (0 if stats disabled or not tracking bytes).
     */
    uint64_t current_allocated_bytes() const noexcept {
        #ifdef ORTEAF_STATS_LEVEL_MPS_VALUE
            #if ORTEAF_STATS_LEVEL_MPS_VALUE <= 4
                return current_allocated_bytes_.load(std::memory_order_relaxed);
            #endif
        #endif
        return 0;
    }
    
    /**
     * @brief Get the peak number of allocated bytes.
     *
     * This represents the maximum number of bytes that were simultaneously allocated
     * at any point since initialization.
     *
     * @return Peak number of allocated bytes (0 if stats disabled or not tracking bytes).
     */
    uint64_t peak_allocated_bytes() const noexcept {
        #ifdef ORTEAF_STATS_LEVEL_MPS_VALUE
            #if ORTEAF_STATS_LEVEL_MPS_VALUE <= 4
                return peak_allocated_bytes_.load(std::memory_order_relaxed);
            #endif
        #endif
        return 0;
    }
    
    /**
     * @brief Get the number of currently active events.
     *
     * @return Number of MPS events that have been created but not yet destroyed (0 if stats disabled or not tracking events).
     */
    uint64_t active_events() const noexcept {
        #ifdef ORTEAF_STATS_LEVEL_MPS_VALUE
            #if ORTEAF_STATS_LEVEL_MPS_VALUE <= 4
                return active_events_.load(std::memory_order_relaxed);
            #endif
        #endif
        return 0;
    }
    
    /**
     * @brief Get the number of currently active streams.
     *
     * @return Number of MPS streams/command queues that have been created but not yet destroyed (0 if stats disabled or not tracking streams).
     */
    uint64_t active_streams() const noexcept {
        #ifdef ORTEAF_STATS_LEVEL_MPS_VALUE
            #if ORTEAF_STATS_LEVEL_MPS_VALUE <= 4
                return active_streams_.load(std::memory_order_relaxed);
            #endif
        #endif
        return 0;
    }
    
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
        #ifdef ORTEAF_STATS_LEVEL_MPS_VALUE
            #if ORTEAF_STATS_LEVEL_MPS_VALUE <= 2
                total_allocations_.fetch_add(1, std::memory_order_relaxed);
                active_allocations_.fetch_add(1, std::memory_order_relaxed);
            #endif
            #if ORTEAF_STATS_LEVEL_MPS_VALUE <= 4
                uint64_t current = current_allocated_bytes_.fetch_add(size, std::memory_order_relaxed) + size;
                uint64_t peak = peak_allocated_bytes_.load(std::memory_order_relaxed);
                while (current > peak && !peak_allocated_bytes_.compare_exchange_weak(peak, current, std::memory_order_relaxed)) {
                    peak = peak_allocated_bytes_.load(std::memory_order_relaxed);
                }
            #endif
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
        #ifdef ORTEAF_STATS_LEVEL_MPS_VALUE
            #if ORTEAF_STATS_LEVEL_MPS_VALUE <= 2
                total_deallocations_.fetch_add(1, std::memory_order_relaxed);
                active_allocations_.fetch_sub(1, std::memory_order_relaxed);
            #endif
            #if ORTEAF_STATS_LEVEL_MPS_VALUE <= 4
                current_allocated_bytes_.fetch_sub(size, std::memory_order_relaxed);
            #endif
        #endif
    }
    
    // MPS-specific update methods - always defined, implementation depends on stats level
    /**
     * @brief Update statistics when an MPS event is created.
     *
     * Increments the active events counter (only if STATS_EXTENDED is enabled).
     */
    void update_create_event() noexcept {
        #ifdef ORTEAF_STATS_LEVEL_MPS_VALUE
            #if ORTEAF_STATS_LEVEL_MPS_VALUE <= 4
                active_events_.fetch_add(1, std::memory_order_relaxed);
            #endif
        #endif
    }
    
    /**
     * @brief Update statistics when an MPS event is destroyed.
     *
     * Decrements the active events counter (only if STATS_EXTENDED is enabled).
     */
    void update_destroy_event() noexcept {
        #ifdef ORTEAF_STATS_LEVEL_MPS_VALUE
            #if ORTEAF_STATS_LEVEL_MPS_VALUE <= 4
                active_events_.fetch_sub(1, std::memory_order_relaxed);
            #endif
        #endif
    }
    
    /**
     * @brief Update statistics when an MPS stream is created.
     *
     * Increments the active streams counter (only if STATS_EXTENDED is enabled).
     */
    void update_create_stream() noexcept {
        #ifdef ORTEAF_STATS_LEVEL_MPS_VALUE
            #if ORTEAF_STATS_LEVEL_MPS_VALUE <= 4
                active_streams_.fetch_add(1, std::memory_order_relaxed);
            #endif
        #endif
    }
    
    /**
     * @brief Update statistics when an MPS stream is destroyed.
     *
     * Decrements the active streams counter (only if STATS_EXTENDED is enabled).
     */
    void update_destroy_stream() noexcept {
        #ifdef ORTEAF_STATS_LEVEL_MPS_VALUE
            #if ORTEAF_STATS_LEVEL_MPS_VALUE <= 4
                active_streams_.fetch_sub(1, std::memory_order_relaxed);
            #endif
        #endif
    }
    
    /**
     * @brief Update statistics when an event becomes active.
     *
     * Increments the active events counter (only if STATS_EXTENDED is enabled).
     */
    void update_active_event() noexcept {
        #ifdef ORTEAF_STATS_LEVEL_MPS_VALUE
            #if ORTEAF_STATS_LEVEL_MPS_VALUE <= 4
                active_events_.fetch_add(1, std::memory_order_relaxed);
            #endif
        #endif
    }
    
    /**
     * @brief Update statistics when an MPS command queue is created.
     *
     * Increments the active streams counter (only if STATS_EXTENDED is enabled).
     */
    void update_create_command_queue() noexcept {
        #ifdef ORTEAF_STATS_LEVEL_MPS_VALUE
            #if ORTEAF_STATS_LEVEL_MPS_VALUE <= 4
                active_streams_.fetch_add(1, std::memory_order_relaxed);
            #endif
        #endif
    }
    
    /**
     * @brief Update statistics when an MPS command queue is destroyed.
     *
     * Decrements the active streams counter (only if STATS_EXTENDED is enabled).
     */
    void update_destroy_command_queue() noexcept {
        #ifdef ORTEAF_STATS_LEVEL_MPS_VALUE
            #if ORTEAF_STATS_LEVEL_MPS_VALUE <= 4
                active_streams_.fetch_sub(1, std::memory_order_relaxed);
            #endif
        #endif
    }

    // String representation
    std::string to_string() const {
        std::ostringstream oss;
#if defined(ORTEAF_STATS_LEVEL_MPS_VALUE) && (ORTEAF_STATS_LEVEL_MPS_VALUE <= 4)
        oss << "MPS Stats:\n";
    #if ORTEAF_STATS_LEVEL_MPS_VALUE <= 2
        oss << "  Total Allocations: " << total_allocations() << "\n";
        oss << "  Total Deallocations: " << total_deallocations() << "\n";
        oss << "  Active Allocations: " << active_allocations() << "\n";
    #endif
    #if ORTEAF_STATS_LEVEL_MPS_VALUE <= 4
        oss << "  Current Allocated Bytes: " << current_allocated_bytes() << "\n";
        oss << "  Peak Allocated Bytes: " << peak_allocated_bytes() << "\n";
        oss << "  Active Events: " << active_events() << "\n";
        oss << "  Active Streams: " << active_streams() << "\n";
    #endif
#else
        oss << "MPS Stats: Disabled\n";
#endif
        return oss.str();
    }
    
    // Print to standard output
    void print() const {
        std::cout << to_string();
    }

private:
#ifdef ORTEAF_STATS_LEVEL_MPS_VALUE
    #if ORTEAF_STATS_LEVEL_MPS_VALUE <= 2
        std::atomic<uint64_t> total_allocations_{0};
        std::atomic<uint64_t> total_deallocations_{0};
        std::atomic<uint64_t> active_allocations_{0};
    #endif
    #if ORTEAF_STATS_LEVEL_MPS_VALUE <= 4
        std::atomic<uint64_t> current_allocated_bytes_{0};
        std::atomic<uint64_t> peak_allocated_bytes_{0};
        std::atomic<uint64_t> active_events_{0};
        std::atomic<uint64_t> active_streams_{0};
    #endif
#endif
};

// Stream output operator
inline std::ostream& operator<<(std::ostream& os, const MpsStats& stats) {
    return os << stats.to_string();
}

// Singleton instance accessor
inline MpsStats& stats_instance() {
    static MpsStats stats;
    return stats;
}

// Global accessor function
inline MpsStats& mps_stats() {
    return stats_instance();
}

// Global update functions
inline void update_alloc(size_t size) {
    stats_instance().update_alloc(size);
}

inline void update_dealloc(size_t size) {
    stats_instance().update_dealloc(size);
}

// MPS-specific update functions
inline void update_create_event() {
    stats_instance().update_create_event();
}

inline void update_destroy_event() {
    stats_instance().update_destroy_event();
}

inline void update_create_stream() {
    stats_instance().update_create_stream();
}

inline void update_destroy_stream() {
    stats_instance().update_destroy_stream();
}

inline void update_active_event() {
    stats_instance().update_active_event();
}

inline void update_create_command_queue() {
    stats_instance().update_create_command_queue();
}

inline void update_destroy_command_queue() {
    stats_instance().update_destroy_command_queue();
}

} // namespace orteaf::internal::backend::mps
