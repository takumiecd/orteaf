#pragma once

#include "orteaf/internal/architecture/architecture.h"
#include "orteaf/internal/architecture/cpu_detect.h"
#include "orteaf/internal/runtime/strong_id.h"
#include "orteaf/internal/diagnostics/error/error.h"

namespace orteaf::internal::runtime::cpu {

struct CpuDeviceManager {
    void initializeDevices() {
        if (initialized_) {
            return;
        }
        state_.arch = ::orteaf::internal::architecture::detect_cpu_architecture();
        state_.is_alive = true;
        initialized_ = true;
    }

    void shutdown() {
        if (!initialized_) {
            return;
        }
        state_.is_alive = false;
        initialized_ = false;
    }

    std::size_t getDeviceCount() const {
        return initialized_ ? 1u : 0u;
    }

    ::orteaf::internal::architecture::Architecture getArch(DeviceId id) const {
        ensureValid(id);
        return state_.arch;
    }

    bool isAlive(DeviceId id) const {
        ensureValid(id);
        return state_.is_alive;
    }

private:
    void ensureValid(DeviceId id) const {
        if (!initialized_ || id != kPrimaryDevice) {
            ::orteaf::internal::diagnostics::error::throw_error(
                ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
                "invalid CPU device");
        }
    }

    struct State {
        ::orteaf::internal::architecture::Architecture arch{::orteaf::internal::architecture::Architecture::cpu_generic};
        bool is_alive{false};
    };

    static constexpr DeviceId kPrimaryDevice{0};

    State state_{};
    bool initialized_{false};
};

inline CpuDeviceManager CpuDeviceManager{};

} // namespace orteaf::internal::runtime::cpu
