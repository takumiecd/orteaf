/**
 * @file cpu_stats_test.cpp
 * @brief Tests for CPU statistics tracking.
 */

#include "orteaf/internal/backend/cpu/cpu_stats.h"
#include "orteaf/internal/backend/cpu/cpu_alloc.h"

#include <gtest/gtest.h>
#include <thread>
#include <vector>

namespace cpu = orteaf::internal::backend::cpu;

#if defined(ORTEAF_ENABLE_CPU) && defined(ORTEAF_STATS_LEVEL_CPU_VALUE)

/**
 * @brief Test fixture for CPU stats tests.
 */
class CpuStatsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // CPU stats don't require explicit initialization
    }
};

#if ORTEAF_STATS_LEVEL_CPU_VALUE <= 2

/**
 * @brief Test that update_alloc increments counters.
 */
TEST_F(CpuStatsTest, UpdateAllocIncrementsCounters) {
    auto& stats = cpu::stats_instance();
    uint64_t initial = stats.total_allocations();
    
    cpu::update_alloc(1024);
    
    EXPECT_EQ(stats.total_allocations(), initial + 1);
    EXPECT_EQ(stats.active_allocations(), 1);
}

/**
 * @brief Test that update_dealloc decrements active allocations.
 */
TEST_F(CpuStatsTest, UpdateDeallocDecrementsActive) {
    auto& stats = cpu::stats_instance();
    
    cpu::update_alloc(1024);
    EXPECT_EQ(stats.active_allocations(), 1);
    EXPECT_EQ(stats.total_deallocations(), 0);
    
    cpu::update_dealloc(1024);
    EXPECT_EQ(stats.active_allocations(), 0);
    EXPECT_EQ(stats.total_deallocations(), 1);
}

/**
 * @brief Test that multiple allocations are tracked correctly.
 */
TEST_F(CpuStatsTest, MultipleAllocationsTracked) {
    auto& stats = cpu::stats_instance();
    uint64_t initial_total = stats.total_allocations();
    uint64_t initial_active = stats.active_allocations();
    
    cpu::update_alloc(1024);
    cpu::update_alloc(2048);
    cpu::update_alloc(4096);
    
    EXPECT_EQ(stats.total_allocations(), initial_total + 3);
    EXPECT_EQ(stats.active_allocations(), initial_active + 3);
    
    cpu::update_dealloc(1024);
    EXPECT_EQ(stats.active_allocations(), initial_active + 2);
    
    cpu::update_dealloc(2048);
    cpu::update_dealloc(4096);
    EXPECT_EQ(stats.active_allocations(), initial_active);
}

#endif  // ORTEAF_STATS_LEVEL_CPU_VALUE <= 2

#if ORTEAF_STATS_LEVEL_CPU_VALUE <= 4

/**
 * @brief Test that current_allocated_bytes tracks correctly.
 */
TEST_F(CpuStatsTest, CurrentAllocatedBytesTracked) {
    auto& stats = cpu::stats_instance();
    uint64_t initial = stats.current_allocated_bytes();
    
    cpu::update_alloc(1024);
    EXPECT_EQ(stats.current_allocated_bytes(), initial + 1024);
    
    cpu::update_alloc(2048);
    EXPECT_EQ(stats.current_allocated_bytes(), initial + 1024 + 2048);
    
    cpu::update_dealloc(1024);
    EXPECT_EQ(stats.current_allocated_bytes(), initial + 2048);
}

/**
 * @brief Test that peak_allocated_bytes tracks maximum.
 */
TEST_F(CpuStatsTest, PeakAllocatedBytesTracksMaximum) {
    auto& stats = cpu::stats_instance();
    uint64_t initial_peak = stats.peak_allocated_bytes();
    uint64_t initial_current = stats.current_allocated_bytes();
    
    cpu::update_alloc(1024);
    uint64_t peak1 = stats.peak_allocated_bytes();
    EXPECT_GE(peak1, initial_peak);
    
    cpu::update_alloc(2048);
    uint64_t peak2 = stats.peak_allocated_bytes();
    EXPECT_GE(peak2, peak1);
    
    cpu::update_dealloc(1024);
    uint64_t peak3 = stats.peak_allocated_bytes();
    EXPECT_EQ(peak3, peak2);  // Peak should not decrease
    
    cpu::update_dealloc(2048);
    EXPECT_EQ(stats.current_allocated_bytes(), initial_current);
    EXPECT_EQ(stats.peak_allocated_bytes(), peak2);  // Peak should remain
}

#endif  // ORTEAF_STATS_LEVEL_CPU_VALUE <= 4

#if ORTEAF_STATS_LEVEL_CPU_VALUE <= 2

/**
 * @brief Test that statistics are updated when using alloc/dealloc.
 */
TEST_F(CpuStatsTest, AllocDeallocUpdatesStats) {
    auto& stats = cpu::stats_instance();
    uint64_t initial_total = stats.total_allocations();
    uint64_t initial_active = stats.active_allocations();
    
    void* ptr1 = cpu::alloc(1024);
    EXPECT_EQ(stats.total_allocations(), initial_total + 1);
    EXPECT_EQ(stats.active_allocations(), initial_active + 1);
    
    void* ptr2 = cpu::alloc(2048);
    EXPECT_EQ(stats.total_allocations(), initial_total + 2);
    EXPECT_EQ(stats.active_allocations(), initial_active + 2);
    
    cpu::dealloc(ptr1, 1024);
    EXPECT_EQ(stats.active_allocations(), initial_active + 1);
    
    cpu::dealloc(ptr2, 2048);
    EXPECT_EQ(stats.active_allocations(), initial_active);
}

/**
 * @brief Test that statistics are thread-safe.
 */
TEST_F(CpuStatsTest, StatisticsAreThreadSafe) {
    auto& stats = cpu::stats_instance();
    constexpr int num_threads = 4;
    constexpr int ops_per_thread = 100;
    
    std::vector<std::thread> threads;
    
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&stats]() {
            for (int j = 0; j < ops_per_thread; ++j) {
                cpu::update_alloc(1024);
                cpu::update_dealloc(1024);
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    // All allocations should be deallocated
    EXPECT_EQ(stats.active_allocations(), 0);
    EXPECT_EQ(stats.total_allocations(), num_threads * ops_per_thread);
    EXPECT_EQ(stats.total_deallocations(), num_threads * ops_per_thread);
}

/**
 * @brief Test that statistics work with actual allocations.
 */
TEST_F(CpuStatsTest, StatisticsWithActualAllocations) {
    auto& stats = cpu::stats_instance();
    uint64_t initial_total = stats.total_allocations();
    uint64_t initial_active = stats.active_allocations();
    
    std::vector<void*> ptrs;
    constexpr size_t size = 256;
    constexpr int num_allocs = 10;
    
    for (int i = 0; i < num_allocs; ++i) {
        ptrs.push_back(cpu::alloc(size));
    }
    
    EXPECT_EQ(stats.total_allocations(), initial_total + num_allocs);
    EXPECT_EQ(stats.active_allocations(), initial_active + num_allocs);
    
    for (auto ptr : ptrs) {
        cpu::dealloc(ptr, size);
    }
    
    EXPECT_EQ(stats.active_allocations(), initial_active);
}

#endif  // ORTEAF_STATS_LEVEL_CPU_VALUE <= 2

/**
 * @brief Test that to_string produces valid output.
 */
TEST_F(CpuStatsTest, ToStringProducesValidOutput) {
    auto& stats = cpu::stats_instance();
    std::string str = stats.to_string();
    
    EXPECT_FALSE(str.empty());
    EXPECT_NE(str.find("CPU Stats"), std::string::npos);
}

/**
 * @brief Test that to_string is not empty even with no operations.
 */
TEST_F(CpuStatsTest, ToStringNotEmptyWhenEmpty) {
    auto& stats = cpu::stats_instance();
    std::string str = stats.to_string();
    
    EXPECT_FALSE(str.empty());
}

/**
 * @brief Test that stream output operator works.
 */
TEST_F(CpuStatsTest, StreamOutputOperatorWorks) {
    auto& stats = cpu::stats_instance();
    std::ostringstream oss;
    
    oss << stats;
    
    EXPECT_FALSE(oss.str().empty());
}

/**
 * @brief Test that stats_instance returns singleton.
 */
TEST_F(CpuStatsTest, StatsInstanceIsSingleton) {
    cpu::CpuStats& stats1 = cpu::stats_instance();
    cpu::CpuStats& stats2 = cpu::stats_instance();
    
    EXPECT_EQ(&stats1, &stats2);
}

/**
 * @brief Test that cpu_stats returns same instance.
 */
TEST_F(CpuStatsTest, CpuStatsReturnsSameInstance) {
    cpu::CpuStats& stats1 = cpu::stats_instance();
    cpu::CpuStats& stats2 = cpu::cpu_stats();
    
    EXPECT_EQ(&stats1, &stats2);
}

#else  // !ORTEAF_ENABLE_CPU || !ORTEAF_STATS_LEVEL_CPU_VALUE

/**
 * @brief Test that statistics are disabled when stats level is not set.
 */
TEST(CpuStats, DisabledWhenStatsLevelNotSet) {
    auto& stats = cpu::stats_instance();
    
    // All update methods should be no-ops
    EXPECT_NO_THROW(stats.update_alloc(1024));
    EXPECT_NO_THROW(stats.update_dealloc(1024));
    
    // to_string should indicate disabled state
    std::string str = stats.to_string();
    EXPECT_NE(str.find("Disabled"), std::string::npos);
}

#endif  // ORTEAF_ENABLE_CPU && ORTEAF_STATS_LEVEL_CPU_VALUE
