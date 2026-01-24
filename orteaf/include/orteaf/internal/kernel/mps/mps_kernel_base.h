#pragma once

#if ORTEAF_ENABLE_MPS

#include <array>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <string>
#include <utility>

#include "orteaf/internal/base/heap_vector.h"
#include "orteaf/internal/execution/mps/manager/mps_compute_pipeline_state_manager.h"
#include "orteaf/internal/execution/mps/manager/mps_library_manager.h"
#include "orteaf/internal/execution/mps/mps_handles.h"

namespace orteaf::internal::kernel::mps {

/**
 * @brief Kernel base structure that caches MPS compute pipeline states.
 *
 * Template parameter N specifies the maximum number of kernel functions
 * (library/function pairs) this base can hold. Each device has its own
 * set of initialized pipeline leases.
 *
 * This is analogous to CUDA's CUmodule - it caches expensive-to-create
 * resources (MTLComputePipelineState) and provides them via leases for
 * kernel execution.
 */
template <std::size_t N> struct MpsKernelBase {
  using PipelineLease =
      ::orteaf::internal::execution::mps::manager::
          MpsComputePipelineStateManager::PipelineLease;
  using LibraryKey =
      ::orteaf::internal::execution::mps::manager::LibraryKey;
  using FunctionKey =
      ::orteaf::internal::execution::mps::manager::FunctionKey;
  using Key = std::pair<LibraryKey, FunctionKey>;

  struct KeyLiteral {
    const char *library;
    const char *function;
  };

  MpsKernelBase() = default;

  /**
   * @brief Construct with initializer list of library/function name pairs.
   */
  explicit MpsKernelBase(std::initializer_list<KeyLiteral> keys) {
    std::size_t idx = 0;
    for (const auto &k : keys) {
      if (idx >= N)
        break;
      keys_[idx++] = Key{LibraryKey::Named(std::string(k.library)),
                         FunctionKey::Named(std::string(k.function))};
    }
    size_ = idx;
  }

  MpsKernelBase(const MpsKernelBase &) = delete;
  MpsKernelBase &operator=(const MpsKernelBase &) = delete;
  MpsKernelBase(MpsKernelBase &&) = default;
  MpsKernelBase &operator=(MpsKernelBase &&) = default;
  ~MpsKernelBase() = default;

  /**
   * @brief Check if pipelines are initialized for the given device.
   */
  bool initialized(
      ::orteaf::internal::execution::mps::MpsDeviceHandle device) const
      noexcept {
    const auto idx = findDeviceIndex(device);
    return idx != kInvalidIndex && device_pipelines_[idx].initialized;
  }

  /**
   * @brief Initialize all pipeline leases for the given device.
   *
   * Uses RuntimeApi to acquire pipeline leases for each kernel function.
   * If already initialized for this device, clears and re-initializes.
   *
   * @tparam RuntimeApi API providing acquirePipeline(device, lib, func)
   */
  template <typename RuntimeApi>
  void initialize(
      ::orteaf::internal::execution::mps::MpsDeviceHandle device) {
    auto entry_idx = findDeviceIndex(device);
    if (entry_idx == kInvalidIndex) {
      device_pipelines_.pushBack(DevicePipelines{device});
      entry_idx = device_pipelines_.size() - 1;
    }
    auto &entry = device_pipelines_[entry_idx];
    entry.pipelines.clear();
    entry.pipelines.reserve(size_);
    for (std::size_t i = 0; i < size_; ++i) {
      const auto &key = keys_[i];
      entry.pipelines.pushBack(
          RuntimeApi::acquirePipeline(device, key.first, key.second));
    }
    entry.initialized = true;
  }

  /**
   * @brief Get a mutable pipeline lease for the specified device and index.
   *
   * @return Pointer to PipelineLease, or nullptr if not initialized or
   * invalid index
   */
  PipelineLease *
  getPipeline(::orteaf::internal::execution::mps::MpsDeviceHandle device,
              std::size_t index) noexcept {
    const auto idx = findDeviceIndex(device);
    if (idx == kInvalidIndex)
      return nullptr;
    auto &entry = device_pipelines_[idx];
    if (!entry.initialized || index >= entry.pipelines.size())
      return nullptr;
    return &entry.pipelines[index];
  }

  /**
   * @brief Get a const pipeline lease for the specified device and index.
   */
  const PipelineLease *
  getPipeline(::orteaf::internal::execution::mps::MpsDeviceHandle device,
              std::size_t index) const noexcept {
    const auto idx = findDeviceIndex(device);
    if (idx == kInvalidIndex)
      return nullptr;
    const auto &entry = device_pipelines_[idx];
    if (!entry.initialized || index >= entry.pipelines.size())
      return nullptr;
    return &entry.pipelines[index];
  }

  /**
   * @brief Get the total number of kernel functions registered.
   */
  std::size_t kernelCount() const noexcept { return size_; }

#if ORTEAF_ENABLE_TEST
  const std::array<Key, N> &keysForTest() const noexcept { return keys_; }
  std::size_t sizeForTest() const noexcept { return size_; }
  std::size_t deviceCountForTest() const noexcept {
    return device_pipelines_.size();
  }
#endif

private:
  struct DevicePipelines {
    ::orteaf::internal::execution::mps::MpsDeviceHandle device{};
    ::orteaf::internal::base::HeapVector<PipelineLease> pipelines{};
    bool initialized{false};
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
  std::array<Key, N> keys_{};
  std::size_t size_{0};
  static constexpr std::size_t kInvalidIndex =
      std::numeric_limits<std::size_t>::max();
};

} // namespace orteaf::internal::kernel::mps

#endif // ORTEAF_ENABLE_MPS
