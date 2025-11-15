#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <limits>

#include "orteaf/internal/backend/mps/mps_command_queue.h"
#include "orteaf/internal/backend/mps/mps_event.h"
#include "orteaf/internal/runtime/manager/mps/mps_command_queue_manager.h"
#include "orteaf/internal/diagnostics/error/error.h"
#include "tests/internal/runtime/mps/testing/backend_mock.h"
#include "tests/internal/testing/error_assert.h"

namespace backend = orteaf::internal::backend;
namespace mps_rt = orteaf::internal::runtime::mps;
namespace test_mps = orteaf::tests::runtime::mps;

using ::testing::NiceMock;
using ::testing::Return;
using ::testing::InSequence;
using ::testing::_;
namespace diag_error = orteaf::internal::diagnostics::error;
using orteaf::tests::ExpectError;

namespace {

backend::mps::MPSCommandQueue_t makeQueue(std::uintptr_t value) {
    return reinterpret_cast<backend::mps::MPSCommandQueue_t>(value);
}

backend::mps::MPSEvent_t makeEvent(std::uintptr_t value) {
    return reinterpret_cast<backend::mps::MPSEvent_t>(value);
}

using MockManager = mps_rt::MpsCommandQueueManager<test_mps::MpsBackendOpsMockAdapter>;

class MpsCommandQueueManagerTest : public ::testing::Test {
protected:
    MpsCommandQueueManagerTest()
        : guard_(mock_) {}

    void TearDown() override {
        manager_.shutdown();
        test_mps::MpsBackendOpsMockRegistry::unbind(mock_);
    }

    NiceMock<test_mps::MpsBackendOpsMock> mock_;
    test_mps::MpsBackendOpsMockRegistry::Guard guard_;
    MockManager manager_;
};

}  // namespace

TEST_F(MpsCommandQueueManagerTest, InitializeCreatesConfiguredNumberOfResources) {
    EXPECT_CALL(mock_, createCommandQueue(_)).WillOnce(Return(makeQueue(0x1))).WillOnce(Return(makeQueue(0x2)));
    EXPECT_CALL(mock_, createEvent(_)).WillOnce(Return(makeEvent(0x10))).WillOnce(Return(makeEvent(0x20)));

    manager_.initialize(2);
    EXPECT_EQ(manager_.capacity(), 2u);

    {
        InSequence seq;
        EXPECT_CALL(mock_, destroyEvent(makeEvent(0x10)));
        EXPECT_CALL(mock_, destroyCommandQueue(makeQueue(0x1)));
        EXPECT_CALL(mock_, destroyEvent(makeEvent(0x20)));
        EXPECT_CALL(mock_, destroyCommandQueue(makeQueue(0x2)));
    }
}

TEST_F(MpsCommandQueueManagerTest, InitializeRejectsCapacityAboveLimit) {
    ExpectError(diag_error::OrteafErrc::InvalidArgument, [&] {
        manager_.initialize(std::numeric_limits<std::size_t>::max());
    });
}

TEST_F(MpsCommandQueueManagerTest, CapacityReflectsPoolSize) {
    EXPECT_EQ(manager_.capacity(), 0u);

    EXPECT_CALL(mock_, createCommandQueue(_)).WillRepeatedly(Return(makeQueue(0x3)));
    EXPECT_CALL(mock_, createEvent(_)).WillRepeatedly(Return(makeEvent(0x30)));
    manager_.initialize(3);
    EXPECT_EQ(manager_.capacity(), 3u);

    EXPECT_CALL(mock_, destroyEvent(_)).Times(3);
    EXPECT_CALL(mock_, destroyCommandQueue(_)).Times(3);
    manager_.shutdown();
    EXPECT_EQ(manager_.capacity(), 0u);
}
