#pragma once

#include <cstddef>

#include "orteaf/internal/architecture/architecture.h"
#include "orteaf/internal/architecture/cpu_detect.h"
#include "orteaf/internal/execution/cpu/cpu_handles.h"
#include "orteaf/internal/execution/cpu/platform/wrapper/cpu_alloc.h"

namespace orteaf::internal::execution::cpu::platform {

/**
 * @brief Virtual interface for CPU slow-path operations.
 *
 * This interface abstracts CPU operations that are typically performed
 * during initialization or resource management (not on the hot path).
 * Using a virtual interface allows for easy mocking in tests.
 *
 * Unlike MpsSlowOps which has many GPU-specific operations, CpuSlowOps
 * is minimal since CPU has fewer resources to manage.
 */
struct CpuSlowOps {
  virtual ~CpuSlowOps() = default;

  // =========================================================================
  // Device operations
  // =========================================================================

  /**
   * @brief Get the number of available CPU devices.
   *
   * Currently always returns 1 (the host CPU).
   */
  virtual int getDeviceCount() = 0;

  /**
   * @brief Detect the CPU architecture for a given device.
   *
   * @param device_id Device handle (typically DeviceHandle{0} for host CPU)
   * @return Detected Architecture enum value
   */
  virtual ::orteaf::internal::architecture::Architecture
  detectArchitecture(
      ::orteaf::internal::execution::cpu::CpuDeviceHandle device_id) = 0;

  // =========================================================================
  // Buffer operations
  // =========================================================================

  /**
   * @brief Allocate a memory buffer with specified size and alignment.
   *
   * @param size Size in bytes to allocate
   * @param alignment Alignment requirement (0 for default)
   * @return Pointer to allocated memory, or nullptr on failure
   */
  virtual void *allocBuffer(std::size_t size, std::size_t alignment) = 0;

  /**
   * @brief Deallocate a previously allocated buffer.
   *
   * @param ptr Pointer to memory to deallocate
   * @param size Size of the buffer (used for statistics)
   */
  virtual void deallocBuffer(void *ptr, std::size_t size) = 0;
};

/**
 * @brief Concrete implementation of CpuSlowOps.
 *
 * Uses the existing CPU platform utilities (cpu_detect.h, cpu_alloc.h).
 */
struct CpuSlowOpsImpl final : public CpuSlowOps {
  int getDeviceCount() override {
    // CPU always has exactly one device (the host)
    return 1;
  }

  ::orteaf::internal::architecture::Architecture detectArchitecture(
      [[maybe_unused]] ::orteaf::internal::execution::cpu::CpuDeviceHandle
          device_id) override {
    return ::orteaf::internal::architecture::detectCpuArchitecture();
  }

  void *allocBuffer(std::size_t size, std::size_t alignment) override {
    if (size == 0) {
      return nullptr;
    }
    try {
      if (alignment == 0) {
        return ::orteaf::internal::execution::cpu::platform::wrapper::alloc(
            size);
      }
      return ::orteaf::internal::execution::cpu::platform::wrapper::
          allocAligned(size, alignment);
    } catch (...) {
      return nullptr;
    }
  }

  void deallocBuffer(void *ptr, std::size_t size) override {
    ::orteaf::internal::execution::cpu::platform::wrapper::dealloc(ptr, size);
  }
};

} // namespace orteaf::internal::execution::cpu::platform
