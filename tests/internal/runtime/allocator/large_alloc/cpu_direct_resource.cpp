#include "orteaf/internal/runtime/allocator/policies/large_alloc/direct_resource_large_alloc.h"
#include "orteaf/internal/backend/backend.h"

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "tests/internal/runtime/allocator/testing/mock_cpu_resource.h"

using ::testing::_;
using ::testing::Return;

namespace allocator = ::orteaf::internal::runtime::allocator;
namespace policies = ::orteaf::internal::runtime::allocator::policies;
using Backend = ::orteaf::internal::backend::Backend;
using ::orteaf::internal::runtime::allocator::testing::MockCpuResource;
using ::orteaf::internal::runtime::allocator::testing::MockCpuResourceImpl;

namespace {

TEST(DirectResourceLargeAlloc, AllocateReturnsMemoryBlockWithId) {
    policies::DirectResourceLargeAllocPolicy<Backend::cpu, MockCpuResource> policy;
    // initialize placeholders (unused for CPU)
    policy.initialize(0, 0, nullptr);

    ::testing::NiceMock<MockCpuResourceImpl> impl;
    MockCpuResource::set(&impl);
    EXPECT_CALL(impl, allocate(128, 64))
        .WillOnce(Return(::orteaf::internal::backend::cpu::CpuBufferView{reinterpret_cast<void*>(0x1), 0, 128}));

    auto block = policy.allocate(128, 64);
    EXPECT_TRUE(block.valid());
    EXPECT_TRUE(policy.isLargeAlloc(block.id));
    MockCpuResource::reset();
}

TEST(DirectResourceLargeAlloc, DeallocateCallsResource) {
    policies::DirectResourceLargeAllocPolicy<Backend::cpu, MockCpuResource> policy;
    policy.initialize(0, 0, nullptr);

    ::orteaf::internal::backend::cpu::CpuBufferView view{reinterpret_cast<void*>(0x2), 0, 256};
    ::testing::NiceMock<MockCpuResourceImpl> impl;
    MockCpuResource::set(&impl);
    EXPECT_CALL(impl, allocate(256, 16)).WillOnce(Return(view));
    EXPECT_CALL(impl, deallocate(view, 256, 16)).Times(1);

    auto block = policy.allocate(256, 16);
    policy.deallocate(block.id, 256, 16);
    MockCpuResource::reset();
}

}  // namespace
