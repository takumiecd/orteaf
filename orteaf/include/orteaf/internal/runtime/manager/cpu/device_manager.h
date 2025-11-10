#pragma once

#include "orteaf/internal/architecture/architecture.h"
#include "orteaf/internal/runtime/strong_id.h"

#include <stdexcept>

namespace orteaf::internal::runtime::cpu {

struct CpuDeviceManager {
    void initialize_devices() {
        if (initialized_) {
            return;
        }
        state_.arch = ::orteaf::internal::architecture::Architecture::cpu_generic;
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

    std::size_t get_device_count() const {
        return initialized_ ? 1u : 0u;
    }

    ::orteaf::internal::architecture::Architecture get_arch(DeviceId id) const {
        ensure_valid(id);
        return state_.arch;
    }

    bool is_alive(DeviceId id) const {
        ensure_valid(id);
        return state_.is_alive;
    }

private:
    void ensure_valid(DeviceId id) const {
        if (!initialized_ || id != kPrimaryDevice) {
            throw std::runtime_error("invalid CPU device");
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
