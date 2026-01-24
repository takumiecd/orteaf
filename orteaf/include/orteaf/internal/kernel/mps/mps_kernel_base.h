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
#include "orteaf/internal/execution/mps/platform/wrapper/mps_compute_command_encoder.h"
#include "orteaf/internal/execution/mps/platform/wrapper/mps_size.h"
#include "orteaf/internal/execution_context/mps/context.h"
#include "orteaf/internal/kernel/param.h"
#include "orteaf/internal/storage/mps/mps_storage.h"

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

  /**
   * @brief Create a compute command encoder from a command buffer.
   *
   * @param command_buffer Command buffer to create the encoder from
   * @return Opaque compute command encoder handle, or nullptr when unavailable/disabled
   */
  ::orteaf::internal::execution::mps::platform::wrapper::MpsComputeCommandEncoder_t
  createComputeCommandEncoder(
      ::orteaf::internal::execution::mps::platform::wrapper::MpsCommandBuffer_t
          command_buffer) const {
    if (command_buffer == nullptr) {
      return nullptr;
    }
    return ::orteaf::internal::execution::mps::platform::wrapper::
        createComputeCommandEncoder(command_buffer);
  }

  /**
   * @brief End encoding on a compute command encoder.
   *
   * Finalizes the encoder and makes it ready for execution.
   * Must be called before committing the command buffer.
   *
   * @param encoder Compute command encoder to end encoding on
   */
  void endEncoding(
      ::orteaf::internal::execution::mps::platform::wrapper::MpsComputeCommandEncoder_t
          encoder) const {
    if (encoder == nullptr) {
      return;
    }
    ::orteaf::internal::execution::mps::platform::wrapper::endEncoding(encoder);
  }

  /**
   * @brief Commit a command buffer for execution.
   *
   * Submits the command buffer to the GPU for execution.
   * The command buffer will be executed asynchronously.
   *
   * @param command_buffer Command buffer to commit
   */
  void commit(
      ::orteaf::internal::execution::mps::platform::wrapper::MpsCommandBuffer_t
          command_buffer) const {
    if (command_buffer == nullptr) {
      return;
    }
    ::orteaf::internal::execution::mps::platform::wrapper::commit(
        command_buffer);
  }

  /**
   * @brief Wait until a command buffer has completed execution.
   *
   * Blocks the current thread until the GPU has finished executing
   * the command buffer.
   *
   * @param command_buffer Command buffer to wait for
   */
  void waitUntilCompleted(
      ::orteaf::internal::execution::mps::platform::wrapper::MpsCommandBuffer_t
          command_buffer) const {
    if (command_buffer == nullptr) {
      return;
    }
    ::orteaf::internal::execution::mps::platform::wrapper::waitUntilCompleted(
        command_buffer);
  }

  /**
   * @brief Set a buffer on the compute command encoder.
   *
   * Retrieves the buffer from the storage and binds it to the encoder
   * at the specified index. The buffer offset from the storage is automatically used.
   *
   * @param encoder Compute command encoder to bind the buffer to
   * @param storage MPS storage containing the buffer to bind
   * @param index Binding index for the buffer
   */
  void setBuffer(
      ::orteaf::internal::execution::mps::platform::wrapper::MpsComputeCommandEncoder_t
          encoder,
      const ::orteaf::internal::storage::mps::MpsStorage &storage,
      std::size_t index) const {
    if (encoder == nullptr) {
      return;
    }
    auto buffer = storage.buffer();
    if (buffer == nullptr) {
      return;
    }
    const std::size_t offset = storage.bufferOffset();
    ::orteaf::internal::execution::mps::platform::wrapper::setBuffer(
        encoder, buffer, offset, index);
  }

  /**
   * @brief Set bytes on the compute command encoder.
   *
   * Binds raw bytes to the encoder at the specified index.
   * Useful for passing small constant data directly without creating a buffer.
   *
   * @param encoder Compute command encoder to bind the bytes to
   * @param bytes Pointer to the data to bind
   * @param length Size of the data in bytes
   * @param index Binding index for the data
   */
  void setBytes(
      ::orteaf::internal::execution::mps::platform::wrapper::MpsComputeCommandEncoder_t
          encoder,
      const void *bytes, std::size_t length, std::size_t index) const {
    if (encoder == nullptr) {
      return;
    }
    if (bytes == nullptr) {
      return;
    }
    ::orteaf::internal::execution::mps::platform::wrapper::setBytes(
        encoder, bytes, length, index);
  }

  /**
   * @brief Set a parameter on the compute command encoder.
   *
   * Binds parameter data to the encoder at the specified index.
   * Useful for passing small constant data directly without creating a buffer.
   * The parameter value is automatically converted to bytes based on its type.
   *
   * @param encoder Compute command encoder to bind the bytes to
   * @param param Parameter containing the data to bind
   * @param index Binding index for the data
   */
  void setParam(
      ::orteaf::internal::execution::mps::platform::wrapper::MpsComputeCommandEncoder_t
          encoder,
      const ::orteaf::internal::kernel::Param &param,
      std::size_t index) const {
    if (encoder == nullptr) {
      return;
    }
    param.visit([encoder, index](const auto &value) {
      using T = std::decay_t<decltype(value)>;
      ::orteaf::internal::execution::mps::platform::wrapper::setBytes(
          encoder, &value, sizeof(T), index);
    });
  }

  /**
   * @brief Set compute pipeline state on the encoder from a pipeline lease.
   *
   * Binds the compute pipeline state to the encoder for kernel execution.
   * The pipeline lease must be valid and configured.
   *
   * @param encoder Compute command encoder to bind the pipeline state to
   * @param pipeline Pipeline lease containing the pipeline state
   */
  void setPipelineState(
      ::orteaf::internal::execution::mps::platform::wrapper::MpsComputeCommandEncoder_t
          encoder,
      const PipelineLease &pipeline) const {
    if (encoder == nullptr) {
      return;
    }
    if (!pipeline) {
      return;
    }
    auto *resource = pipeline.operator->();
    if (resource == nullptr || resource->pipeline_state == nullptr) {
      return;
    }
    ::orteaf::internal::execution::mps::platform::wrapper::setPipelineState(
        encoder, resource->pipeline_state);
  }

  /**
   * @brief Dispatch threadgroups for compute kernel execution.
   *
   * Configures the grid dimensions for kernel execution.
   * Both parameters specify 3D sizes (width, height, depth).
   *
   * @param encoder Compute command encoder to dispatch on
   * @param threadgroups Number of threadgroups to dispatch
   * @param threads_per_threadgroup Number of threads in each threadgroup
   */
  void dispatchThreadgroups(
      ::orteaf::internal::execution::mps::platform::wrapper::MpsComputeCommandEncoder_t
          encoder,
      ::orteaf::internal::execution::mps::platform::wrapper::MpsSize_t threadgroups,
      ::orteaf::internal::execution::mps::platform::wrapper::MpsSize_t
          threads_per_threadgroup) const {
    if (encoder == nullptr) {
      return;
    }
    ::orteaf::internal::execution::mps::platform::wrapper::setThreadgroups(
        encoder, threadgroups, threads_per_threadgroup);
  }

  /**
   * @brief Dispatch threads for compute kernel execution (grid-stride).
   *
   * Specifies the total number of threads to execute (grid size).
   * The system automatically calculates the number of threadgroups.
   * This is useful for kernels using grid-stride loops.
   *
   * @param encoder Compute command encoder to dispatch on
   * @param threads_per_grid Total number of threads to execute
   * @param threads_per_threadgroup Number of threads in each threadgroup
   */
  void dispatchThreads(
      ::orteaf::internal::execution::mps::platform::wrapper::MpsComputeCommandEncoder_t
          encoder,
      ::orteaf::internal::execution::mps::platform::wrapper::MpsSize_t threads_per_grid,
      ::orteaf::internal::execution::mps::platform::wrapper::MpsSize_t
          threads_per_threadgroup) const {
    if (encoder == nullptr) {
      return;
    }
    ::orteaf::internal::execution::mps::platform::wrapper::dispatchThreads(
        encoder, threads_per_grid, threads_per_threadgroup);
  }

  /**
   * @brief Get the GPU start time of a command buffer (in seconds).
   *
   * Returns the time when the GPU started executing the command buffer.
   * Only valid after the command buffer has been scheduled.
   *
   * @param command_buffer Command buffer to query
   * @return GPU start time in seconds, or 0.0 if not available
   */
  double getGPUStartTime(
      ::orteaf::internal::execution::mps::platform::wrapper::MpsCommandBuffer_t
          command_buffer) const {
    return ::orteaf::internal::execution::mps::platform::wrapper::
        getGPUStartTime(command_buffer);
  }

  /**
   * @brief Get the GPU end time of a command buffer (in seconds).
   *
   * Returns the time when the GPU finished executing the command buffer.
   * Only valid after the command buffer has completed.
   *
   * @param command_buffer Command buffer to query
   * @return GPU end time in seconds, or 0.0 if not available
   */
  double getGPUEndTime(
      ::orteaf::internal::execution::mps::platform::wrapper::MpsCommandBuffer_t
          command_buffer) const {
    return ::orteaf::internal::execution::mps::platform::wrapper::
        getGPUEndTime(command_buffer);
  }

  /**
   * @brief Get the GPU execution duration of a command buffer (in seconds).
   *
   * Returns the elapsed time between GPU start and end.
   * Only valid after the command buffer has completed.
   *
   * @param command_buffer Command buffer to query
   * @return GPU execution duration in seconds, or 0.0 if not available
   */
  double getGPUDuration(
      ::orteaf::internal::execution::mps::platform::wrapper::MpsCommandBuffer_t
          command_buffer) const {
    return ::orteaf::internal::execution::mps::platform::wrapper::
        getGPUDuration(command_buffer);
  }

  /**
   * @brief Create a 3D size for grid dimensions.
   *
   * Helper function to create MpsSize_t for threadgroup counts or thread counts.
   *
   * @param width Width dimension
   * @param height Height dimension (default 1)
   * @param depth Depth dimension (default 1)
   * @return MpsSize_t with specified dimensions
   */
  static ::orteaf::internal::execution::mps::platform::wrapper::MpsSize_t
  makeGridSize(std::size_t width, std::size_t height = 1,
               std::size_t depth = 1) {
    return ::orteaf::internal::execution::mps::platform::wrapper::makeSize(
        static_cast<::orteaf::internal::execution::mps::platform::wrapper::
                        MpsInt_t>(width),
        static_cast<::orteaf::internal::execution::mps::platform::wrapper::
                        MpsInt_t>(height),
        static_cast<::orteaf::internal::execution::mps::platform::wrapper::
                        MpsInt_t>(depth));
  }

  /**
   * @brief Create threads per threadgroup size.
   *
   * Helper function to create MpsSize_t for threads per threadgroup.
   * Common values are (256, 1, 1) for 1D, (16, 16, 1) for 2D, etc.
   *
   * @param width Width dimension
   * @param height Height dimension (default 1)
   * @param depth Depth dimension (default 1)
   * @return MpsSize_t with specified dimensions
   */
  static ::orteaf::internal::execution::mps::platform::wrapper::MpsSize_t
  makeThreadsPerThreadgroup(std::size_t width, std::size_t height = 1,
                            std::size_t depth = 1) {
    return ::orteaf::internal::execution::mps::platform::wrapper::makeSize(
        static_cast<::orteaf::internal::execution::mps::platform::wrapper::
                        MpsInt_t>(width),
        static_cast<::orteaf::internal::execution::mps::platform::wrapper::
                        MpsInt_t>(height),
        static_cast<::orteaf::internal::execution::mps::platform::wrapper::
                        MpsInt_t>(depth));
  }

  /**
   * @brief Calculate grid size from total threads and threads per threadgroup.
   *
   * Computes the number of threadgroups needed to cover the total number
   * of threads, rounding up to ensure all threads are covered.
   *
   * @param total_threads Total number of threads to execute
   * @param threads_per_threadgroup Number of threads in each threadgroup
   * @return Grid size (number of threadgroups)
   */
  static ::orteaf::internal::execution::mps::platform::wrapper::MpsSize_t
  calculateGridSize(
      ::orteaf::internal::execution::mps::platform::wrapper::MpsSize_t
          total_threads,
      ::orteaf::internal::execution::mps::platform::wrapper::MpsSize_t
          threads_per_threadgroup) {
    const auto grid_width = (total_threads.width + threads_per_threadgroup.width - 1) /
                            threads_per_threadgroup.width;
    const auto grid_height =
        (total_threads.height + threads_per_threadgroup.height - 1) /
        threads_per_threadgroup.height;
    const auto grid_depth = (total_threads.depth + threads_per_threadgroup.depth - 1) /
                            threads_per_threadgroup.depth;
    return makeGridSize(static_cast<std::size_t>(grid_width),
                        static_cast<std::size_t>(grid_height),
                        static_cast<std::size_t>(grid_depth));
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
