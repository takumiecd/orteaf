#pragma once

#if ORTEAF_ENABLE_MPS

#include <cstddef>
#include <limits>
#include <utility>

#include "orteaf/internal/base/heap_vector.h"
#include "orteaf/internal/execution/mps/manager/mps_compute_pipeline_state_manager.h"
#include "orteaf/internal/execution/mps/manager/mps_device_manager.h"
#include "orteaf/internal/execution/mps/manager/mps_library_manager.h"
#include "orteaf/internal/execution/mps/mps_handles.h"

namespace orteaf::internal::kernel {
class KernelArgs;
} // namespace orteaf::internal::kernel

namespace orteaf::internal::kernel::core {
class KernelEntry;
} // namespace orteaf::internal::kernel::core

namespace orteaf::internal::execution::mps::manager {
struct KernelBasePayloadPoolTraits;
}

namespace orteaf::internal::execution::mps::resource {

struct MpsKernelMetadata;

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
  using ExecuteFunc = void (*)(MpsKernelBase &,
                               ::orteaf::internal::kernel::KernelArgs &);
  using MetadataType = MpsKernelMetadata;
  using PipelineLease = ::orteaf::internal::execution::mps::manager::
      MpsComputePipelineStateManager::PipelineLease;
  using LibraryKey = ::orteaf::internal::execution::mps::manager::LibraryKey;
  using FunctionKey = ::orteaf::internal::execution::mps::manager::FunctionKey;
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
  bool configured(::orteaf::internal::execution::mps::MpsDeviceHandle device)
      const noexcept {
    const auto idx = findDeviceIndex(device);
    return idx != kInvalidIndex && device_pipelines_[idx].configured;
  }

  /**
   * @brief Configure pipeline leases for the given device lease.
   *
   * Acquires pipeline leases from the device resource's library and pipeline
   * managers. If already configured for this device, clears and re-configures.
   *
   * @param device_lease Device lease providing library/pipeline managers
   */
  void configurePipelines(
      ::orteaf::internal::execution::mps::manager::MpsDeviceManager::DeviceLease
          &device_lease);

  /**
   * @brief Ensure pipeline leases are configured for the given device lease.
   *
   * Returns false if the device lease is invalid or any pipeline acquisition
   * fails.
   */
  bool ensurePipelines(
      ::orteaf::internal::execution::mps::manager::MpsDeviceManager::DeviceLease
          &device_lease);

  /**
   * @brief Clear keys, cached pipelines, and execute callback.
   */
  void reset() noexcept {
    device_pipelines_.clear();
    keys_.clear();
    execute_ = nullptr;
  }

  /**
   * @brief Get a pipeline lease for the specified device and kernel index.
   *
   * Returns an invalid lease when the device is not configured or index is out
   * of range.
   */
  PipelineLease
  getPipelineLease(::orteaf::internal::execution::mps::MpsDeviceHandle device,
                   std::size_t index) const noexcept {
    const auto idx = findDeviceIndex(device);
    if (idx == kInvalidIndex) {
      return PipelineLease{};
    }
    const auto &entry = device_pipelines_[idx];
    if (!entry.configured || index >= entry.pipelines.size()) {
      return PipelineLease{};
    }
    return entry.pipelines[index];
  }

  /**
   * @brief Get the total number of kernel functions registered.
   */
  std::size_t kernelCount() const noexcept { return keys_.size(); }

  /**
   * @brief Get the registered library/function key pairs.
   *
   * Used by KernelRegistry to extract metadata when demoting entries
   * from Main Memory to Secondary Storage.
   *
   * @return Const reference to the keys vector
   */
  const ::orteaf::internal::base::HeapVector<Key> &keys() const noexcept {
    return keys_;
  }

  // Public getter for ExecuteFunc (needed by Metadata)
  ExecuteFunc execute() const noexcept { return execute_; }

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

  std::size_t
  findDeviceIndex(::orteaf::internal::execution::mps::MpsDeviceHandle device)
      const noexcept {
    for (std::size_t i = 0; i < device_pipelines_.size(); ++i) {
      if (device_pipelines_[i].device == device) {
        return i;
      }
    }
    return kInvalidIndex;
  }

private:
  friend class ::orteaf::internal::kernel::core::KernelEntry;
  friend struct MpsKernelMetadata;
  friend struct ::orteaf::internal::execution::mps::manager::
      KernelBasePayloadPoolTraits;

  void run(::orteaf::internal::kernel::KernelArgs &args);

  bool setKeys(const ::orteaf::internal::base::HeapVector<Key> &keys);

  void setExecute(ExecuteFunc execute) noexcept { execute_ = execute; }

  ::orteaf::internal::base::HeapVector<DevicePipelines> device_pipelines_{};
  ::orteaf::internal::base::HeapVector<Key> keys_{};
  ExecuteFunc execute_{nullptr};
  static constexpr std::size_t kInvalidIndex =
      std::numeric_limits<std::size_t>::max();
};

} // namespace orteaf::internal::execution::mps::resource

#endif // ORTEAF_ENABLE_MPS
