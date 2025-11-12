#pragma once

#include "orteaf/internal/architecture/architecture.h"
#include "orteaf/internal/architecture/mps_detect.h"
#include "orteaf/internal/backend/mps/mps_device.h"
#include "orteaf/internal/base/strong_id.h"

namespace orteaf::internal::runtime::backend_ops::mps {

struct MpsBackendOps {
    static int getDeviceCount() {
        return ::orteaf::internal::backend::mps::getDeviceCount();
    }

    static ::orteaf::internal::backend::mps::MPSDevice_t getDevice(::orteaf::internal::backend::mps::MPSInt_t index) {
        return ::orteaf::internal::backend::mps::getDevice(index);
    }

    static void releaseDevice(::orteaf::internal::backend::mps::MPSDevice_t device) {
        ::orteaf::internal::backend::mps::deviceRelease(device);
    }

    static ::orteaf::internal::architecture::Architecture detectArchitecture(::orteaf::internal::base::DeviceId device_id) {
        return ::orteaf::internal::architecture::detectMpsArchitectureForDeviceId(device_id);
    }
};

}  // namespace orteaf::internal::runtime::backend_ops::mps

