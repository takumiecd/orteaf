#pragma once

#if ORTEAF_ENABLE_MPS

#include <cstddef>
#include <limits>
#include <string>
#include <utility>

#include "orteaf/internal/base/heap_vector.h"
#include "orteaf/internal/execution/mps/manager/mps_compute_pipeline_state_manager.h"
#include "orteaf/internal/execution/mps/manager/mps_library_manager.h"
#include "orteaf/internal/execution/mps/mps_handles.h"
#include "orteaf/internal/execution/mps/platform/wrapper/mps_command_buffer.h"
#include "orteaf/internal/execution_context/mps/context.h"

namespace orteaf::internal::kernel::mps {

/**
 * @brief Kernel base structure that caches MPS compute pipeline states.
 *
 * Each device has its own set of initialized pipeline leases.
 * This is analogous to CUDA's CUmodule - it caches expensive-to-create
 * resources (MTLComputePipelineState) and provides them via leases for
 * kernel execution.
 *
 * Each MpsKernelBase can manage multiple kernels (library/function pairs).
 */
struct MpsKernelBase {
  using PipelineLease =
      ::orteaf::internal::execution::mps::manager::
          MpsComputePipelineStateManager::PipelineLease;
  using LibraryKey =
      ::orteaf::internal::execution::mps::manager::LibraryKey;
  using FunctionKey =
      ::orteaf::internal::execution::mps::manager::FunctionKey;
  using Key = std::pair<LibraryKey, FunctionKey>;

  MpsKernelBase() = default;

  MpsKernelBase(const MpsKernelBase &) = delete;
  MpsKernelBase &operator=(const MpsKernelBase &) = delete;
  MpsKernelBase(MpsKernelBase &&) = default;
  MpsKernelBase &operator=(MpsKernelBase &&) = default;
  ~MpsKernelBase() = default;

  /**
   * @brief Check if pipelines are configured for the given device.
   */
  bool configured(
      ::orteaf::internal::execution::mps::MpsDeviceHandle device) const
      noexcept {
    const auto idx = findDeviceIndex(device);
    return idx != kInvalidIndex && device_pipelines_[idx].configured;
  }

  /**
   * @brief Configure all pipeline leases for the given context's device.
   *
   * Acquires pipeline leases from the device resource's library and pipeline managers.
   * If already configured for this device, clears and re-configures.
   *
   * @param context Execution context containing device lease
   */
  void configure(
      ::orteaf::internal::execution_context::mps::Context &context);

  /**
   * @brief Get a mutable pipeline lease for the specified device and kernel index.
   *
   * @return Pointer to PipelineLease, or nullptr if not initialized or invalid index
   */
  PipelineLease *
  getPipeline(::orteaf::internal::execution::mps::MpsDeviceHandle device,
              std::size_t index) noexcept {
    const auto idx = findDeviceIndex(device);
    if (idx == kInvalidIndex)
      return nullptr;
    auto &entry = device_pipelines_[idx];
    if (!entry.configured || index >= entry.pipelines.size())
      return nullptr;
    return &entry.pipelines[index];
  }

  /**
   * @brief Get a const pipeline lease for the specified device and kernel index.
   */
  const PipelineLease *
  getPipeline(::orteaf::internal::execution::mps::MpsDeviceHandle device,
              std::size_t index) const noexcept {
    const auto idx = findDeviceIndex(device);
    if (idx == kInvalidIndex)
      return nullptr;
    const auto &entry = device_pipelines_[idx];
    if (!entry.configured || index >= entry.pipelines.size())
      return nullptr;
    return &entry.pipelines[index];
  }

  /**
   * @brief Add a kernel function key.
   */
  void addKey(const char *library, const char *function) {
    keys_.pushBack(Key{LibraryKey::Named(std::string(library)),
                       FunctionKey::Named(std::string(function))});
  }

  /**
   * @brief Reserve space for kernel function keys.
   */
  void reserveKeys(std::size_t count) { keys_.reserve(count); }

  /**
   * @brief Get the total number of kernel functions registered.
   */
  std::size_t kernelCount() const noexcept { return keys_.size(); }

  /**
   * @brief Create a command buffer from the context's command queue.
   *
   * @param context Execution context containing the command queue lease
   * @return Opaque command buffer handle, or nullptr when unavailable/disabled
   */
  ::orteaf::internal::execution::mps::platform::wrapper::MpsCommandBuffer_t
  createCommandBuffer(
      ::orteaf::internal::execution_context::mps::Context &context) const {
    if (!context.command_queue) {
      return nullptr;
    }
    auto *queue_resource = context.command_queue.operator->();
    if (queue_resource == nullptr || !queue_resource->hasQueue()) {
      return nullptr;
    }
    return ::orteaf::internal::execution::mps::platform::wrapper::
        createCommandBuffer(queue_resource->queue());
  }

#if ORTEAF_ENABLE_TESTING
  ::orteaf::internal::base::HeapVector<Key> &keysForTest() noexcept {
    return keys_;
  }
  std::size_t deviceCountForTest() const noexcept {
    return device_pipelines_.size();
  }
#endif

private:
  struct DevicePipelines {
    ::orteaf::internal::execution::mps::MpsDeviceHandle device{};
    ::orteaf::internal::base::HeapVector<PipelineLease> pipelines{};
    bool configured{false};
  };

  std::size_t findDeviceIndex(
      ::orteaf::internal::execution::mps::MpsDeviceHandle device) const
      noexcept {
    for (std::size_t i = 0; i < device_pipelines_.size(); ++i) {
      if (device_pipelines_[i].device == device) {
        return i;
      }
    }
    return kInvalidIndex;
  }

  ::orteaf::internal::base::HeapVector<DevicePipelines> device_pipelines_{};
  ::orteaf::internal::base::HeapVector<Key> keys_{};
  static constexpr std::size_t kInvalidIndex =
      std::numeric_limits<std::size_t>::max();
};

} // namespace orteaf::internal::kernel::mps

#endif // ORTEAF_ENABLE_MPS
