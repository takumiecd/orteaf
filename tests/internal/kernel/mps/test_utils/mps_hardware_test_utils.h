#pragma once

#include <optional>
#include <string>
#include <system_error>

#include <orteaf/internal/execution/mps/platform/wrapper/mps_command_buffer.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_compute_command_encoder.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_device.h>

namespace orteaf::tests::internal::kernel::mps::test_utils {

namespace mps_wrapper = ::orteaf::internal::execution::mps::platform::wrapper;

struct MpsHardwareContext {
  mps_wrapper::MpsDevice_t device{nullptr};
  mps_wrapper::MpsCommandQueue_t queue{nullptr};
  mps_wrapper::MpsCommandBuffer_t command_buffer{nullptr};
  mps_wrapper::MpsComputeCommandEncoder_t encoder{nullptr};

  MpsHardwareContext() = default;
  MpsHardwareContext(MpsHardwareContext &&other) noexcept
      : device(other.device), queue(other.queue),
        command_buffer(other.command_buffer), encoder(other.encoder) {
    other.device = nullptr;
    other.queue = nullptr;
    other.command_buffer = nullptr;
    other.encoder = nullptr;
  }
  MpsHardwareContext &operator=(MpsHardwareContext &&other) noexcept {
    if (this != &other) {
      cleanup();
      device = other.device;
      queue = other.queue;
      command_buffer = other.command_buffer;
      encoder = other.encoder;
      other.device = nullptr;
      other.queue = nullptr;
      other.command_buffer = nullptr;
      other.encoder = nullptr;
    }
    return *this;
  }
  MpsHardwareContext(const MpsHardwareContext &) = delete;
  MpsHardwareContext &operator=(const MpsHardwareContext &) = delete;
  ~MpsHardwareContext() { cleanup(); }

private:
  void cleanup() {
    if (encoder != nullptr) {
      mps_wrapper::destroyComputeCommandEncoder(encoder);
      encoder = nullptr;
    }
    if (command_buffer != nullptr) {
      mps_wrapper::destroyCommandBuffer(command_buffer);
      command_buffer = nullptr;
    }
    if (queue != nullptr) {
      mps_wrapper::destroyCommandQueue(queue);
      queue = nullptr;
    }
    if (device != nullptr) {
      mps_wrapper::deviceRelease(device);
      device = nullptr;
    }
  }
};

struct HardwareAcquireResult {
  std::optional<MpsHardwareContext> context;
  std::string reason;
};

inline HardwareAcquireResult acquireHardware(bool need_buffer = false,
                                             bool need_encoder = false) {
  HardwareAcquireResult result{};
  MpsHardwareContext hardware{};

  try {
    hardware.device = mps_wrapper::getDevice();
  } catch (const std::system_error &err) {
    result.reason = err.what();
    return result;
  }

  if (hardware.device == nullptr) {
    result.reason = "No Metal devices available";
    return result;
  }

  hardware.queue = mps_wrapper::createCommandQueue(hardware.device);
  if (hardware.queue == nullptr) {
    result.reason = "Failed to create command queue";
    return result;
  }

  if (need_buffer || need_encoder) {
    hardware.command_buffer = mps_wrapper::createCommandBuffer(hardware.queue);
    if (hardware.command_buffer == nullptr) {
      result.reason = "Failed to create command buffer";
      return result;
    }
  }

  if (need_encoder) {
    hardware.encoder =
        mps_wrapper::createComputeCommandEncoder(hardware.command_buffer);
    if (hardware.encoder == nullptr) {
      result.reason = "Failed to create compute command encoder";
      return result;
    }
  }

  result.context = std::move(hardware);
  return result;
}

} // namespace orteaf::tests::internal::kernel::mps::test_utils
