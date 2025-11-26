#include "orteaf/internal/runtime/allocator/lowlevel/hierarchical_slot_allocator.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "orteaf/internal/backend/backend.h"
#include "orteaf/internal/backend/backend_traits.h"
#include "tests/internal/runtime/allocator/testing/mock_resource.h"
#include "tests/internal/testing/error_assert.h"

using ::testing::_;
using ::testing::NiceMock;
using ::testing::Return;

namespace allocator = ::orteaf::internal::runtime::allocator;
namespace policies = ::orteaf::internal::runtime::allocator::policies;
using Backend = ::orteaf::internal::backend::Backend;
using Traits = ::orteaf::internal::backend::BackendTraits<Backend::Cpu>;
using BufferView = Traits::BufferView;
using HeapRegion = Traits::HeapRegion;
using ::orteaf::internal::runtime::allocator::testing::MockCpuHeapOps;
using ::orteaf::internal::runtime::allocator::testing::MockCpuHeapOpsImpl;
using OrteafErrc = ::orteaf::internal::diagnostics::error::OrteafErrc;

namespace {

using Allocator = policies::HierarchicalSlotAllocator<MockCpuHeapOps, Backend::Cpu>;

static BufferView MapReturn(HeapRegion region) {
    return BufferView{region.data(), 0, region.size()};
}

class HierarchicalSlotAllocatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        MockCpuHeapOps::set(&impl_);
    }

    void TearDown() override {
        MockCpuHeapOps::reset();
    }

    NiceMock<MockCpuHeapOpsImpl> impl_;
    MockCpuHeapOps heap_ops_;
    Allocator allocator_;
};

TEST_F(HierarchicalSlotAllocatorTest, AllocateReturnsBufferView) {
    void* base = reinterpret_cast<void*>(0x1000);
    EXPECT_CALL(impl_, reserve(256)).WillOnce(Return(HeapRegion{base, 256}));
    EXPECT_CALL(impl_, map(_)).WillOnce(::testing::Invoke(MapReturn));

    Allocator::Config cfg{};
    cfg.levels = {256};
    allocator_.initialize(cfg, &heap_ops_);

    auto view = allocator_.allocate(256);
    EXPECT_TRUE(view);
    EXPECT_EQ(view.data(), base);
    EXPECT_EQ(view.size(), 256);
}

TEST_F(HierarchicalSlotAllocatorTest, DeallocateCallsUnmap) {
    void* base = reinterpret_cast<void*>(0x2000);
    EXPECT_CALL(impl_, reserve(256)).WillOnce(Return(HeapRegion{base, 256}));
    EXPECT_CALL(impl_, map(_)).WillOnce(::testing::Invoke(MapReturn));
    EXPECT_CALL(impl_, unmap(_, 256)).Times(1);

    Allocator::Config cfg{};
    cfg.levels = {256};
    allocator_.initialize(cfg, &heap_ops_);

    auto view = allocator_.allocate(256);
    allocator_.deallocate(view);
}

TEST_F(HierarchicalSlotAllocatorTest, AllocateSmallSizeFromLargerSlot) {
    // levels = {256, 128} で 128 バイトを要求
    // 256 バイトのスロットから 128 バイトのスロットに分割されるはず
    void* base = reinterpret_cast<void*>(0x3000);
    EXPECT_CALL(impl_, reserve(256)).WillOnce(Return(HeapRegion{base, 256}));
    EXPECT_CALL(impl_, map(_)).WillOnce(::testing::Invoke(MapReturn));

    Allocator::Config cfg{};
    cfg.levels = {256, 128};
    cfg.initial_bytes = 256;
    allocator_.initialize(cfg, &heap_ops_);

    auto view = allocator_.allocate(128);
    EXPECT_TRUE(view);
    EXPECT_EQ(view.size(), 128);
}

TEST_F(HierarchicalSlotAllocatorTest, DeallocateSplitSlotCallsUnmap) {
    // levels = {256, 128} で 128 バイトを確保して解放
    void* base = reinterpret_cast<void*>(0x4000);
    EXPECT_CALL(impl_, reserve(256)).WillOnce(Return(HeapRegion{base, 256}));
    EXPECT_CALL(impl_, map(_)).WillOnce(::testing::Invoke(MapReturn));
    EXPECT_CALL(impl_, unmap(_, 128)).Times(1);

    Allocator::Config cfg{};
    cfg.levels = {256, 128};
    cfg.initial_bytes = 256;
    allocator_.initialize(cfg, &heap_ops_);

    auto view = allocator_.allocate(128);
    EXPECT_TRUE(view);
    EXPECT_EQ(view.size(), 128);

    allocator_.deallocate(view);
}

// ============================================================================
// Validation tests
// ============================================================================

TEST_F(HierarchicalSlotAllocatorTest, InitializeThrowsOnEmptyLevels) {
    Allocator::Config cfg{};
    cfg.levels = {};
    orteaf::tests::ExpectError(OrteafErrc::InvalidParameter, [&] {
        allocator_.initialize(cfg, &heap_ops_);
    });
}

TEST_F(HierarchicalSlotAllocatorTest, InitializeThrowsOnInvalidInitialBytes) {
    Allocator::Config cfg{};
    cfg.levels = {256};
    cfg.initial_bytes = 100;  // 256の倍数ではない
    orteaf::tests::ExpectError(OrteafErrc::InvalidParameter, [&] {
        allocator_.initialize(cfg, &heap_ops_);
    });
}

TEST_F(HierarchicalSlotAllocatorTest, InitializeAcceptsValidInitialBytes) {
    void* base = reinterpret_cast<void*>(0x5000);
    EXPECT_CALL(impl_, reserve(512)).WillOnce(Return(HeapRegion{base, 512}));

    Allocator::Config cfg{};
    cfg.levels = {256};
    cfg.initial_bytes = 512;  // 256の倍数
    EXPECT_NO_THROW(allocator_.initialize(cfg, &heap_ops_));
}

TEST_F(HierarchicalSlotAllocatorTest, InitializeThrowsOnZeroLevel) {
    Allocator::Config cfg{};
    cfg.levels = {256, 0, 64};
    orteaf::tests::ExpectError(OrteafErrc::InvalidParameter, [&] {
        allocator_.initialize(cfg, &heap_ops_);
    });
}

TEST_F(HierarchicalSlotAllocatorTest, InitializeThrowsOnNonDecreasingLevels) {
    Allocator::Config cfg{};
    cfg.levels = {128, 256};  // 増加している
    orteaf::tests::ExpectError(OrteafErrc::InvalidParameter, [&] {
        allocator_.initialize(cfg, &heap_ops_);
    });
}

TEST_F(HierarchicalSlotAllocatorTest, InitializeThrowsOnNonDivisibleLevels) {
    Allocator::Config cfg{};
    cfg.levels = {256, 100};  // 256 % 100 != 0
    orteaf::tests::ExpectError(OrteafErrc::InvalidParameter, [&] {
        allocator_.initialize(cfg, &heap_ops_);
    });
}

// ============================================================================
// Threshold validation tests
// ============================================================================

TEST_F(HierarchicalSlotAllocatorTest, InitializeThrowsOnThresholdTooSmall) {
    Allocator::Config cfg{};
    cfg.levels = {256};
    cfg.threshold = 4;  // システム最小閾値より小さい
    orteaf::tests::ExpectError(OrteafErrc::InvalidParameter, [&] {
        allocator_.initialize(cfg, &heap_ops_);
    });
}

TEST_F(HierarchicalSlotAllocatorTest, InitializeThrowsOnNonPowerOfTwoThreshold) {
    Allocator::Config cfg{};
    cfg.levels = {256};
    cfg.threshold = 100;  // 2の冪乗ではない
    orteaf::tests::ExpectError(OrteafErrc::InvalidParameter, [&] {
        allocator_.initialize(cfg, &heap_ops_);
    });
}

TEST_F(HierarchicalSlotAllocatorTest, InitializeThrowsOnLevelBelowThresholdNotPowerOfTwo) {
    Allocator::Config cfg{};
    cfg.levels = {256, 64, 48};  // 48はthreshold=64未満で2の冪乗ではない
    cfg.threshold = 64;
    orteaf::tests::ExpectError(OrteafErrc::InvalidParameter, [&] {
        allocator_.initialize(cfg, &heap_ops_);
    });
}

TEST_F(HierarchicalSlotAllocatorTest, InitializeThrowsOnLevelAboveThresholdNotDivisible) {
    Allocator::Config cfg{};
    cfg.levels = {300, 64};  // 300 % 64 != 0
    cfg.threshold = 64;
    orteaf::tests::ExpectError(OrteafErrc::InvalidParameter, [&] {
        allocator_.initialize(cfg, &heap_ops_);
    });
}

TEST_F(HierarchicalSlotAllocatorTest, InitializeAcceptsValidThreshold) {
    void* base = reinterpret_cast<void*>(0x6000);
    EXPECT_CALL(impl_, reserve(256)).WillOnce(Return(HeapRegion{base, 256}));

    Allocator::Config cfg{};
    cfg.levels = {256, 64, 32, 16};
    cfg.threshold = 64;
    EXPECT_NO_THROW(allocator_.initialize(cfg, &heap_ops_));
}

}  // namespace
