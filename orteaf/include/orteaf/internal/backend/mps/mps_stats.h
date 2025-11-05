#pragma once

#include <atomic>
#include <string>
#include <sstream>
#include <iostream>

namespace orteaf::internal::backend::mps {

class MpsStats {
public:
    // STATS_BASIC(2) or STATS_EXTENDED(4) enabled
#ifdef ORTEAF_STATS_LEVEL_MPS_VALUE
    #if ORTEAF_STATS_LEVEL_MPS_VALUE <= 2
        // Getter methods for basic stats
        uint64_t total_allocations() const noexcept {
            return total_allocations_.load(std::memory_order_relaxed);
        }
        
        uint64_t total_deallocations() const noexcept {
            return total_deallocations_.load(std::memory_order_relaxed);
        }
        
        uint64_t active_allocations() const noexcept {
            return active_allocations_.load(std::memory_order_relaxed);
        }
    #endif
    
    #if ORTEAF_STATS_LEVEL_MPS_VALUE <= 4
        // Getter methods for extended stats
        uint64_t current_allocated_bytes() const noexcept {
            return current_allocated_bytes_.load(std::memory_order_relaxed);
        }
        
        uint64_t peak_allocated_bytes() const noexcept {
            return peak_allocated_bytes_.load(std::memory_order_relaxed);
        }
        
        // MPS-specific stats
        uint64_t active_events() const noexcept {
            return active_events_.load(std::memory_order_relaxed);
        }
        
        uint64_t active_streams() const noexcept {
            return active_streams_.load(std::memory_order_relaxed);
        }
    #endif
    
    // Update methods (handles both BASIC and EXTENDED levels)
    void update_alloc(size_t size) noexcept {
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
    }
    
    void update_dealloc(size_t size) noexcept {
        #if ORTEAF_STATS_LEVEL_MPS_VALUE <= 2
            total_deallocations_.fetch_add(1, std::memory_order_relaxed);
            active_allocations_.fetch_sub(1, std::memory_order_relaxed);
        #endif
        #if ORTEAF_STATS_LEVEL_MPS_VALUE <= 4
            current_allocated_bytes_.fetch_sub(size, std::memory_order_relaxed);
        #endif
    }
    
    // MPS-specific update methods
    #ifdef ORTEAF_STATS_LEVEL_MPS_VALUE
        #if ORTEAF_STATS_LEVEL_MPS_VALUE <= 4
            void update_create_event() noexcept {
                active_events_.fetch_add(1, std::memory_order_relaxed);
            }
            
            void update_destroy_event() noexcept {
                active_events_.fetch_sub(1, std::memory_order_relaxed);
            }
            
            void update_create_stream() noexcept {
                active_streams_.fetch_add(1, std::memory_order_relaxed);
            }
            
            void update_destroy_stream() noexcept {
                active_streams_.fetch_sub(1, std::memory_order_relaxed);
            }
            
            void update_active_event() noexcept {
                active_events_.fetch_add(1, std::memory_order_relaxed);
            }
            
            void update_create_command_queue() noexcept {
                active_streams_.fetch_add(1, std::memory_order_relaxed);
            }
            
            void update_destroy_command_queue() noexcept {
                active_streams_.fetch_sub(1, std::memory_order_relaxed);
            }
        #endif
    #endif
#else
    // When stats are disabled, provide no-op methods
    void update_alloc(size_t) noexcept {}
    void update_dealloc(size_t) noexcept {}
    void update_create_event() noexcept {}
    void update_destroy_event() noexcept {}
    void update_create_stream() noexcept {}
    void update_destroy_stream() noexcept {}
    void update_active_event() noexcept {}
    void update_create_command_queue() noexcept {}
    void update_destroy_command_queue() noexcept {}
#endif

    // String representation
    std::string to_string() const {
        std::ostringstream oss;
#ifdef ORTEAF_STATS_LEVEL_MPS_VALUE
        #if ORTEAF_STATS_LEVEL_MPS_VALUE <= 2
            oss << "MPS Stats:\n";
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
