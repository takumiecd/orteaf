#pragma once

#include <initializer_list>

#include <gmock/gmock.h>

#include "orteaf/internal/backend/mps/mps_command_queue.h"
#include "orteaf/internal/backend/mps/mps_device.h"
#include "orteaf/internal/backend/mps/mps_event.h"
#include "orteaf/internal/base/strong_id.h"
#include "orteaf/internal/architecture/architecture.h"
#include "tests/internal/runtime/mps/testing/backend_mock.h"

namespace orteaf::tests::runtime::mps {

/**
 * Helpers for expressing BackendOps expectations in tests.
 *
 * Each helper mirrors a BackendOps entry point so tests stay concise
 * while remaining explicit about expected call counts and return values.
 */
struct BackendMockExpectations {
    static void expectGetDeviceCount(MpsBackendOpsMock& mock, int value) {
        EXPECT_CALL(mock, getDeviceCount()).WillRepeatedly(::testing::Return(value));
    }

    static void expectGetDevices(
        MpsBackendOpsMock& mock,
        std::initializer_list<std::pair<::orteaf::internal::backend::mps::MPSInt_t,
                                        ::orteaf::internal::backend::mps::MPSDevice_t>> expectations) {
        for (const auto& [index, device] : expectations) {
            EXPECT_CALL(mock, getDevice(index)).WillOnce(::testing::Return(device));
        }
    }

    static void expectReleaseDevices(
        MpsBackendOpsMock& mock,
        std::initializer_list<::orteaf::internal::backend::mps::MPSDevice_t> devices) {
        if (devices.size() == 0) {
            EXPECT_CALL(mock, releaseDevice(::testing::_)).Times(0);
            return;
        }
        for (auto device : devices) {
            EXPECT_CALL(mock, releaseDevice(device)).Times(1);
        }
    }

    static void expectDetectArchitectures(
        MpsBackendOpsMock& mock,
        std::initializer_list<std::pair<::orteaf::internal::base::DeviceId,
                                        ::orteaf::internal::architecture::Architecture>> expectations) {
        for (const auto& [id, arch] : expectations) {
            EXPECT_CALL(mock, detectArchitecture(id)).WillOnce(::testing::Return(arch));
        }
    }

    static void expectCreateCommandQueues(
        MpsBackendOpsMock& mock,
        std::initializer_list<::orteaf::internal::backend::mps::MPSCommandQueue_t> handles,
        ::testing::Matcher<::orteaf::internal::backend::mps::MPSDevice_t> device_matcher = ::testing::_) {
        if (handles.size() == 0) {
            EXPECT_CALL(mock, createCommandQueue(device_matcher)).Times(0);
            return;
        }
        for (auto handle : handles) {
            EXPECT_CALL(mock, createCommandQueue(device_matcher)).WillOnce(::testing::Return(handle));
        }
    }

    static void expectDestroyCommandQueues(
        MpsBackendOpsMock& mock,
        std::initializer_list<::orteaf::internal::backend::mps::MPSCommandQueue_t> handles) {
        if (handles.size() == 0) {
            EXPECT_CALL(mock, destroyCommandQueue(::testing::_)).Times(0);
            return;
        }
        for (auto handle : handles) {
            EXPECT_CALL(mock, destroyCommandQueue(handle)).Times(1);
        }
    }

    static void expectCreateEvents(
        MpsBackendOpsMock& mock,
        std::initializer_list<::orteaf::internal::backend::mps::MPSEvent_t> handles,
        ::testing::Matcher<::orteaf::internal::backend::mps::MPSDevice_t> device_matcher = ::testing::_) {
        if (handles.size() == 0) {
            EXPECT_CALL(mock, createEvent(device_matcher)).Times(0);
            return;
        }
        for (auto handle : handles) {
            EXPECT_CALL(mock, createEvent(device_matcher)).WillOnce(::testing::Return(handle));
        }
    }

    static void expectDestroyEvents(
        MpsBackendOpsMock& mock,
        std::initializer_list<::orteaf::internal::backend::mps::MPSEvent_t> handles) {
        if (handles.size() == 0) {
            EXPECT_CALL(mock, destroyEvent(::testing::_)).Times(0);
            return;
        }
        for (auto handle : handles) {
            EXPECT_CALL(mock, destroyEvent(handle)).Times(1);
        }
    }
};

}  // namespace orteaf::tests::runtime::mps

