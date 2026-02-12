#pragma once

#if ORTEAF_ENABLE_MPS

#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>

#include "orteaf/internal/diagnostics/error/error.h"
#include "orteaf/internal/execution/mps/platform/wrapper/mps_command_buffer.h"
#include "orteaf/internal/execution/mps/platform/wrapper/mps_compute_command_encoder.h"
#include "orteaf/internal/execution/mps/platform/wrapper/mps_fence.h"
#include "orteaf/internal/execution/mps/platform/wrapper/mps_size.h"
#include "orteaf/internal/execution/mps/resource/mps_kernel_base.h"
#include "orteaf/internal/execution_context/mps/context.h"
#include "orteaf/internal/kernel/core/kernel_args.h"
#include "orteaf/internal/kernel/param/param.h"
#include "orteaf/internal/kernel/schema/kernel_param_schema.h"
#include "orteaf/internal/kernel/schema/kernel_storage_schema.h"
#include "orteaf/internal/kernel/storage/storage_binding.h"
#include "orteaf/internal/storage/mps/mps_storage.h"
#include "orteaf/internal/storage/registry/storage_types.h"

namespace orteaf::internal::kernel::mps {

using MpsKernelBase = ::orteaf::internal::execution::mps::resource::MpsKernelBase;

/**
 * @brief RAII helper for MPS kernel execution.
 *
 * Encapsulates the common boilerplate for kernel execution:
 * - Command buffer and encoder creation
 * - Pipeline state setup
 * - Automatic endEncoding and commit on destruction
 *
 * Usage:
 * @code
 * auto session = MpsKernelSession::begin(base, args, 0);
 * if (!session) return;
 *
 * session->waitDependencies(storages.a, storages.b, storages.c);
 * session->bindStorages<0, 1, 2>(storages.a, storages.b, storages.c);
 * session->bindParams<3>(params.num_elements);
 * session->dispatch1D(params.num_elements);
 * session->updateTokens(storages.a, storages.b, storages.c);
 * // RAII: auto endEncoding + commit
 * @endcode
 */
class MpsKernelSession {
public:
  using MpsSize_t =
      ::orteaf::internal::execution::mps::platform::wrapper::MpsSize_t;
  using MpsCommandBuffer_t =
      ::orteaf::internal::execution::mps::platform::wrapper::MpsCommandBuffer_t;
  using MpsComputeCommandEncoder_t = ::orteaf::internal::execution::mps::
      platform::wrapper::MpsComputeCommandEncoder_t;
  using PipelineLease = MpsKernelBase::PipelineLease;

  /**
   * @brief Begin a new kernel session.
   *
   * Creates command buffer, encoder, and sets pipeline state.
   * Returns nullopt if any step fails.
   *
   * @param base Configured MpsKernelBase
   * @param args Kernel arguments
   * @param kernel_index Index of the kernel to execute (default 0)
   * @return Optional session, empty if creation failed
   */
  static std::optional<MpsKernelSession>
  begin(MpsKernelBase &base, ::orteaf::internal::kernel::KernelArgs &args,
        std::size_t kernel_index = 0) {
    auto *context =
        args.context()
            .tryAs<::orteaf::internal::execution_context::mps::Context>();
    if (context == nullptr) {
      return std::nullopt;
    }
    auto command_buffer = createCommandBuffer(*context);
    if (!command_buffer) {
      return std::nullopt;
    }

    auto encoder = createComputeCommandEncoder(command_buffer);
    if (!encoder) {
      return std::nullopt;
    }

    auto pipeline =
        base.getPipelineLease(context->device.payloadHandle(), kernel_index);
    if (!pipeline) {
      endEncoding(encoder);
      return std::nullopt;
    }
    setPipelineState(encoder, pipeline);

    return MpsKernelSession(args, command_buffer, encoder);
  }

  // Move only
  MpsKernelSession(MpsKernelSession &&other) noexcept
      : args_(other.args_), command_buffer_(other.command_buffer_),
        encoder_(other.encoder_), committed_(other.committed_) {
    other.committed_ = true; // Prevent double commit
  }

  MpsKernelSession &operator=(MpsKernelSession &&other) noexcept {
    if (this != &other) {
      finish();
      args_ = other.args_;
      command_buffer_ = other.command_buffer_;
      encoder_ = other.encoder_;
      committed_ = other.committed_;
      other.committed_ = true;
    }
    return *this;
  }

  // No copy
  MpsKernelSession(const MpsKernelSession &) = delete;
  MpsKernelSession &operator=(const MpsKernelSession &) = delete;

  /**
   * @brief Destructor - commits if not already done.
   */
  ~MpsKernelSession() { finish(); }

  /**
   * @brief Create a command buffer from the context's command queue.
   */
  static MpsCommandBuffer_t
  createCommandBuffer(
      ::orteaf::internal::execution_context::mps::Context &context) {
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
   */
  static MpsComputeCommandEncoder_t
  createComputeCommandEncoder(MpsCommandBuffer_t command_buffer) {
    if (command_buffer == nullptr) {
      return nullptr;
    }
    return ::orteaf::internal::execution::mps::platform::wrapper::
        createComputeCommandEncoder(command_buffer);
  }

  /**
   * @brief End encoding on a compute command encoder.
   */
  static void endEncoding(MpsComputeCommandEncoder_t encoder) {
    if (encoder == nullptr) {
      return;
    }
    ::orteaf::internal::execution::mps::platform::wrapper::endEncoding(encoder);
  }

  /**
   * @brief Commit a command buffer for execution.
   */
  static void commit(MpsCommandBuffer_t command_buffer) {
    if (command_buffer == nullptr) {
      return;
    }
    ::orteaf::internal::execution::mps::platform::wrapper::commit(
        command_buffer);
  }

  /**
   * @brief Wait until a command buffer has completed execution.
   */
  static void waitUntilCompleted(MpsCommandBuffer_t command_buffer) {
    if (command_buffer == nullptr) {
      return;
    }
    ::orteaf::internal::execution::mps::platform::wrapper::waitUntilCompleted(
        command_buffer);
  }

  /**
   * @brief Set a buffer on the compute command encoder.
   */
  static void
  setBuffer(MpsComputeCommandEncoder_t encoder,
            const ::orteaf::internal::storage::mps::MpsStorage &storage,
            std::size_t index) {
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
   * @brief Set a buffer from a storage field.
   */
  template <::orteaf::internal::kernel::OperandId ID,
            ::orteaf::internal::kernel::Role AccessRole =
                ::orteaf::internal::kernel::Role::Data>
  static void setBuffer(
      MpsComputeCommandEncoder_t encoder,
      const ::orteaf::internal::kernel::StorageField<ID, AccessRole> &field,
      std::size_t index) {
    if (encoder == nullptr) {
      return;
    }
    using AnyBinding = ::orteaf::internal::kernel::StorageBinding;
    using MpsLease = ::orteaf::internal::storage::MpsStorageLease;
    const auto &binding = field.template binding<AnyBinding>();
    const auto &storage_lease = binding.lease;
    auto *mps_lease = storage_lease.template tryAs<MpsLease>();
    if (!mps_lease) {
      return;
    }
    auto *storage_ptr = mps_lease->operator->();
    if (!storage_ptr) {
      return;
    }
    setBuffer(encoder, *storage_ptr, index);
  }

  /**
   * @brief Type alias for compile-time index sequence.
   */
  template <std::size_t... Is> using Indices = std::index_sequence<Is...>;

  /**
   * @brief Bind multiple storage fields at explicit indices.
   */
  template <std::size_t... Is, typename... Fields>
  static void bindStoragesAt(MpsComputeCommandEncoder_t encoder,
                             std::index_sequence<Is...>,
                             const Fields &...fields) {
    static_assert(sizeof...(Is) == sizeof...(Fields),
                  "Number of indices must match number of fields");
    (setBuffer(encoder, fields, Is), ...);
  }

  /**
   * @brief Bind multiple parameter fields at explicit indices.
   */
  template <std::size_t... Is, typename... Fields>
  static void bindParamsAt(MpsComputeCommandEncoder_t encoder,
                           std::index_sequence<Is...>,
                           const Fields &...fields) {
    static_assert(sizeof...(Is) == sizeof...(Fields),
                  "Number of indices must match number of fields");
    (setParam(encoder, fields, Is), ...);
  }

  /**
   * @brief Set bytes on the compute command encoder.
   */
  static void setBytes(MpsComputeCommandEncoder_t encoder, const void *bytes,
                       std::size_t length, std::size_t index) {
    if (encoder == nullptr || bytes == nullptr) {
      return;
    }
    ::orteaf::internal::execution::mps::platform::wrapper::setBytes(
        encoder, bytes, length, index);
  }

  /**
   * @brief Set bytes on the session's encoder.
   */
  void setBytes(const void *bytes, std::size_t length, std::size_t index) {
    setBytes(encoder_, bytes, length, index);
  }

  /**
   * @brief Set a parameter on the compute command encoder.
   */
  static void setParam(MpsComputeCommandEncoder_t encoder,
                       const ::orteaf::internal::kernel::Param &param,
                       std::size_t index) {
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
   * @brief Set a parameter field on the compute command encoder.
   */
  template <::orteaf::internal::kernel::ParamId ID, typename T>
  static void setParam(MpsComputeCommandEncoder_t encoder,
                       const ::orteaf::internal::kernel::Field<ID, T> &field,
                       std::size_t index) {
    if (encoder == nullptr) {
      return;
    }
    setBytes(encoder, &field.value, sizeof(T), index);
  }

  /**
   * @brief Set compute pipeline state on the encoder from a pipeline lease.
   */
  static void setPipelineState(MpsComputeCommandEncoder_t encoder,
                               const PipelineLease &pipeline) {
    if (encoder == nullptr || !pipeline) {
      return;
    }
    auto *resource = pipeline.operator->();
    if (resource == nullptr || resource->pipelineState() == nullptr) {
      return;
    }
    ::orteaf::internal::execution::mps::platform::wrapper::setPipelineState(
        encoder, resource->pipelineState());
  }

  /**
   * @brief Dispatch threadgroups for compute kernel execution.
   */
  static void dispatchThreadgroups(MpsComputeCommandEncoder_t encoder,
                                   MpsSize_t threadgroups,
                                   MpsSize_t threads_per_threadgroup) {
    if (encoder == nullptr) {
      return;
    }
    ::orteaf::internal::execution::mps::platform::wrapper::setThreadgroups(
        encoder, threadgroups, threads_per_threadgroup);
  }

  /**
   * @brief Dispatch threads for compute kernel execution.
   */
  static void dispatchThreads(MpsComputeCommandEncoder_t encoder,
                              MpsSize_t threads_per_grid,
                              MpsSize_t threads_per_threadgroup) {
    if (encoder == nullptr) {
      return;
    }
    ::orteaf::internal::execution::mps::platform::wrapper::dispatchThreads(
        encoder, threads_per_grid, threads_per_threadgroup);
  }

  /**
   * @brief Get the GPU start time of a command buffer.
   */
  static double getGPUStartTime(MpsCommandBuffer_t command_buffer) {
    return ::orteaf::internal::execution::mps::platform::wrapper::
        getGPUStartTime(command_buffer);
  }

  /**
   * @brief Get the GPU end time of a command buffer.
   */
  static double getGPUEndTime(MpsCommandBuffer_t command_buffer) {
    return ::orteaf::internal::execution::mps::platform::wrapper::getGPUEndTime(
        command_buffer);
  }

  /**
   * @brief Get the GPU execution duration of a command buffer.
   */
  static double getGPUDuration(MpsCommandBuffer_t command_buffer) {
    return ::orteaf::internal::execution::mps::platform::wrapper::
        getGPUDuration(command_buffer);
  }

  /**
   * @brief Create a 3D size for grid dimensions.
   */
  static MpsSize_t makeGridSize(std::size_t width, std::size_t height = 1,
                                std::size_t depth = 1) {
    return ::orteaf::internal::execution::mps::platform::wrapper::makeSize(
        static_cast<
            ::orteaf::internal::execution::mps::platform::wrapper::MpsInt_t>(
            width),
        static_cast<
            ::orteaf::internal::execution::mps::platform::wrapper::MpsInt_t>(
            height),
        static_cast<
            ::orteaf::internal::execution::mps::platform::wrapper::MpsInt_t>(
            depth));
  }

  /**
   * @brief Create threads per threadgroup size.
   */
  static MpsSize_t makeThreadsPerThreadgroup(std::size_t width,
                                             std::size_t height = 1,
                                             std::size_t depth = 1) {
    return ::orteaf::internal::execution::mps::platform::wrapper::makeSize(
        static_cast<
            ::orteaf::internal::execution::mps::platform::wrapper::MpsInt_t>(
            width),
        static_cast<
            ::orteaf::internal::execution::mps::platform::wrapper::MpsInt_t>(
            height),
        static_cast<
            ::orteaf::internal::execution::mps::platform::wrapper::MpsInt_t>(
            depth));
  }

  /**
   * @brief Calculate grid size from total threads and threads per threadgroup.
   */
  static MpsSize_t calculateGridSize(MpsSize_t total_threads,
                                     MpsSize_t threads_per_threadgroup) {
    const auto grid_width =
        (total_threads.width + threads_per_threadgroup.width - 1) /
        threads_per_threadgroup.width;
    const auto grid_height =
        (total_threads.height + threads_per_threadgroup.height - 1) /
        threads_per_threadgroup.height;
    const auto grid_depth =
        (total_threads.depth + threads_per_threadgroup.depth - 1) /
        threads_per_threadgroup.depth;
    return makeGridSize(static_cast<std::size_t>(grid_width),
                        static_cast<std::size_t>(grid_height),
                        static_cast<std::size_t>(grid_depth));
  }

  /**
   * @brief Wait for all storage dependencies (RAW/WAW/WAR hazards).
   */
  template <typename... Fields> void waitDependencies(Fields &...fields) {
    waitAllStorageDependencies(encoder_, fields...);
  }

  /**
   * @brief Bind storage fields at explicit indices.
   *
   * @tparam Is Buffer indices matching Metal shader [[buffer(N)]]
   */
  template <std::size_t... Is, typename... Fields>
  void bindStorages(const Fields &...fields) {
    bindStoragesAt(encoder_, std::index_sequence<Is...>{}, fields...);
  }

  /**
   * @brief Bind parameter fields at explicit indices.
   *
   * @tparam Is Buffer indices matching Metal shader [[buffer(N)]]
   */
  template <std::size_t... Is, typename... Fields>
  void bindParams(const Fields &...fields) {
    bindParamsAt(encoder_, std::index_sequence<Is...>{}, fields...);
  }

  /**
   * @brief Dispatch 1D compute threads.
   */
  void dispatch1D(std::size_t count, std::size_t threads_per_group = 256) {
    auto grid = makeGridSize(count);
    auto tpg = makeThreadsPerThreadgroup(threads_per_group);
    dispatchThreads(encoder_, grid, tpg);
  }

  /**
   * @brief Dispatch with explicit grid and threadgroup sizes.
   */
  void dispatch(MpsSize_t grid, MpsSize_t threads_per_group) {
    dispatchThreadgroups(encoder_, grid, threads_per_group);
  }

  /**
   * @brief Update fence and reuse tokens for all relevant storages.
   *
   * @return true if successful, false if fence acquisition failed
   */
  template <typename... Fields> [[nodiscard]] bool updateTokens(Fields &...fields) {
    auto *context =
        args_->context()
            .tryAs<::orteaf::internal::execution_context::mps::Context>();
    if (context == nullptr) {
      return false;
    }
    return updateAllStorageTokens(*context, command_buffer_, encoder_,
                                  fields...);
  }

  /**
   * @brief Get the underlying encoder for advanced operations.
   */
  MpsComputeCommandEncoder_t encoder() const noexcept { return encoder_; }

  /**
   * @brief Get the underlying command buffer.
   */
  MpsCommandBuffer_t commandBuffer() const noexcept { return command_buffer_; }

private:
  MpsKernelSession(::orteaf::internal::kernel::KernelArgs &args,
                   MpsCommandBuffer_t command_buffer,
                   MpsComputeCommandEncoder_t encoder)
      : args_(&args), command_buffer_(command_buffer), encoder_(encoder) {}

  static auto acquireFence(
      ::orteaf::internal::execution_context::mps::Context &context,
      MpsCommandBuffer_t command_buffer) {
    if (!context.command_queue) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
          "Context has no command queue");
    }
    auto *queue_resource = context.command_queue.operator->();
    if (queue_resource == nullptr) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
          "Command queue lease has no resource");
    }
    return queue_resource->lifetime().acquire(command_buffer);
  }

  static void updateFence(
      MpsComputeCommandEncoder_t encoder,
      ::orteaf::internal::execution::mps::platform::wrapper::MpsFence_t fence) {
    if (encoder == nullptr || fence == nullptr) {
      return;
    }
    ::orteaf::internal::execution::mps::platform::wrapper::updateFence(encoder,
                                                                       fence);
  }

  static void
  waitForFence(MpsComputeCommandEncoder_t encoder,
               ::orteaf::internal::execution::mps::platform::wrapper::MpsFence_t
                   fence) {
    if (encoder == nullptr || fence == nullptr) {
      return;
    }
    ::orteaf::internal::execution::mps::platform::wrapper::waitForFence(encoder,
                                                                        fence);
  }

  template <typename... Fields>
  static void waitAllStorageDependencies(MpsComputeCommandEncoder_t encoder,
                                         Fields &...fields) {
    (waitStorageDependency(encoder, fields), ...);
  }

  template <typename... Fields>
  [[nodiscard]] static bool updateAllStorageTokens(
      ::orteaf::internal::execution_context::mps::Context &context,
      MpsCommandBuffer_t command_buffer, MpsComputeCommandEncoder_t encoder,
      Fields &...fields) {
    auto fence_lease = acquireFence(context, command_buffer);

    auto *payload = fence_lease.operator->();
    if (!payload || !payload->hasFence()) {
      return false;
    }

    updateFence(encoder, payload->fence());
    (updateStorageToken(fence_lease, fields), ...);
    return true;
  }

  template <typename Field>
  static void waitStorageDependency(MpsComputeCommandEncoder_t encoder,
                                    const Field &field) {
    using Access = ::orteaf::internal::kernel::Access;
    constexpr auto access = Field::access();

    if (!field) {
      return;
    }

    using AnyBinding = ::orteaf::internal::kernel::StorageBinding;
    using MpsLease = ::orteaf::internal::storage::MpsStorageLease;
    auto &storage_lease_any = field.template lease<AnyBinding>();
    auto *mps_lease = storage_lease_any.template tryAs<MpsLease>();
    if (!mps_lease) {
      return;
    }
    auto *storage_ptr = mps_lease->operator->();
    if (!storage_ptr) {
      return;
    }
    auto &fence_token = storage_ptr->fenceToken();

    if constexpr (access == Access::Read) {
      if (fence_token.hasWriteFence()) {
        auto *payload = fence_token.writeFence().operator->();
        if (payload && payload->hasFence()) {
          waitForFence(encoder, payload->fence());
        }
      }
    } else if constexpr (access == Access::Write ||
                         access == Access::ReadWrite) {
      if (fence_token.hasWriteFence()) {
        auto *payload = fence_token.writeFence().operator->();
        if (payload && payload->hasFence()) {
          waitForFence(encoder, payload->fence());
        }
      }
      for (std::size_t i = 0; i < fence_token.readFenceCount(); ++i) {
        auto *payload = fence_token.readFence(i).operator->();
        if (payload && payload->hasFence()) {
          waitForFence(encoder, payload->fence());
        }
      }
    }
  }

  template <typename FenceLease, typename Field>
  static void updateStorageToken(FenceLease &fence_lease, Field &field) {
    using Access = ::orteaf::internal::kernel::Access;
    constexpr auto access = Field::access();

    if (!field) {
      return;
    }

    using AnyBinding = ::orteaf::internal::kernel::StorageBinding;
    using MpsLease = ::orteaf::internal::storage::MpsStorageLease;
    auto &storage_lease_any = field.template lease<AnyBinding>();
    auto *mps_lease = storage_lease_any.template tryAs<MpsLease>();
    if (!mps_lease) {
      return;
    }
    auto *storage_ptr = mps_lease->operator->();
    if (!storage_ptr) {
      return;
    }
    auto &fence_token = storage_ptr->fenceToken();

    if constexpr (access == Access::Read) {
      fence_token.addReadFence(FenceLease(fence_lease));
    } else if constexpr (access == Access::Write) {
      fence_token.setWriteFence(FenceLease(fence_lease));
    } else if constexpr (access == Access::ReadWrite) {
      fence_token.addReadFence(FenceLease(fence_lease));
      fence_token.setWriteFence(FenceLease(fence_lease));
    }

    auto &reuse_token = storage_ptr->reuseToken();
    auto *payload = fence_lease.operator->();
    if (payload) {
      typename std::remove_reference_t<decltype(reuse_token)>::Hazard hazard;
      hazard.setCommandQueueHandle(payload->commandQueueHandle());
      hazard.setCommandBuffer(payload->commandBuffer());
      reuse_token.addOrReplaceHazard(std::move(hazard));
    }
  }

  void finish() {
    if (!committed_) {
      endEncoding(encoder_);
      commit(command_buffer_);
      committed_ = true;
    }
  }

  ::orteaf::internal::kernel::KernelArgs *args_;
  MpsCommandBuffer_t command_buffer_;
  MpsComputeCommandEncoder_t encoder_;
  bool committed_ = false;
};

} // namespace orteaf::internal::kernel::mps

#endif // ORTEAF_ENABLE_MPS
