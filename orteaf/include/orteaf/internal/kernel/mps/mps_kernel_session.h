#pragma once

#if ORTEAF_ENABLE_MPS

#include <cstddef>
#include <optional>
#include <utility>

#include "orteaf/internal/execution/mps/platform/wrapper/mps_command_buffer.h"
#include "orteaf/internal/execution/mps/platform/wrapper/mps_compute_command_encoder.h"
#include "orteaf/internal/execution/mps/platform/wrapper/mps_size.h"
#include "orteaf/internal/execution_context/mps/context.h"
#include "orteaf/internal/kernel/core/kernel_args.h"
#include "orteaf/internal/kernel/mps/mps_kernel_base.h"

namespace orteaf::internal::kernel::mps {

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
  static std::optional<MpsKernelSession> begin(MpsKernelBase &base,
                                               ::orteaf::internal::kernel::KernelArgs &args,
                                               std::size_t kernel_index = 0) {
    auto *context =
        args.context().tryAs<::orteaf::internal::execution_context::mps::Context>();
    if (context == nullptr) {
      return std::nullopt;
    }
    auto command_buffer = base.createCommandBuffer(*context);
    if (!command_buffer) {
      return std::nullopt;
    }

    auto encoder = base.createComputeCommandEncoder(command_buffer);
    if (!encoder) {
      return std::nullopt;
    }

    auto *pipeline = base.getPipeline(context->device.payloadHandle(),
                                      kernel_index);
    if (!pipeline) {
      base.endEncoding(encoder);
      return std::nullopt;
    }
    base.setPipelineState(encoder, *pipeline);

    return MpsKernelSession(base, args, command_buffer, encoder);
  }

  // Move only
  MpsKernelSession(MpsKernelSession &&other) noexcept
      : base_(other.base_), args_(other.args_),
        command_buffer_(other.command_buffer_), encoder_(other.encoder_),
        committed_(other.committed_) {
    other.committed_ = true; // Prevent double commit
  }

  MpsKernelSession &operator=(MpsKernelSession &&other) noexcept {
    if (this != &other) {
      finish();
      base_ = other.base_;
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
   * @brief Wait for storage dependencies (RAW hazards).
   */
  template <typename... Fields> void waitDependencies(Fields &...fields) {
    base_->waitAllStorageDependencies(encoder_, fields...);
  }

  /**
   * @brief Bind storage fields at explicit indices.
   *
   * @tparam Is Buffer indices matching Metal shader [[buffer(N)]]
   */
  template <std::size_t... Is, typename... Fields>
  void bindStorages(const Fields &...fields) {
    base_->bindStoragesAt(encoder_, std::index_sequence<Is...>{}, fields...);
  }

  /**
   * @brief Bind parameter fields at explicit indices.
   *
   * @tparam Is Buffer indices matching Metal shader [[buffer(N)]]
   */
  template <std::size_t... Is, typename... Fields>
  void bindParams(const Fields &...fields) {
    base_->bindParamsAt(encoder_, std::index_sequence<Is...>{}, fields...);
  }

  /**
   * @brief Dispatch 1D compute threads.
   *
   * @param count Total number of elements to process
   * @param threads_per_group Threads per threadgroup (default 256)
   */
  void dispatch1D(std::size_t count, std::size_t threads_per_group = 256) {
    auto grid = MpsKernelBase::makeGridSize(count);
    auto tpg = MpsKernelBase::makeThreadsPerThreadgroup(threads_per_group);
    base_->dispatchThreads(encoder_, grid, tpg);
  }

  /**
   * @brief Dispatch with explicit grid and threadgroup sizes.
   */
  void dispatch(MpsSize_t grid, MpsSize_t threads_per_group) {
    base_->dispatchThreadgroups(encoder_, grid, threads_per_group);
  }

  /**
   * @brief Update fence tokens for all output storages.
   *
   * @return true if successful, false if fence acquisition failed
   */
  template <typename... Fields>
  [[nodiscard]] bool updateTokens(Fields &...fields) {
    auto *context =
        args_->context().tryAs<::orteaf::internal::execution_context::mps::Context>();
    if (context == nullptr) {
      return false;
    }
    return base_->updateAllStorageTokens(*context, command_buffer_, encoder_,
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
  MpsKernelSession(MpsKernelBase &base,
                   ::orteaf::internal::kernel::KernelArgs &args,
                   MpsCommandBuffer_t command_buffer,
                   MpsComputeCommandEncoder_t encoder)
      : base_(&base), args_(&args), command_buffer_(command_buffer),
        encoder_(encoder) {}

  void finish() {
    if (!committed_) {
      base_->endEncoding(encoder_);
      base_->commit(command_buffer_);
      committed_ = true;
    }
  }

  MpsKernelBase *base_;
  ::orteaf::internal::kernel::KernelArgs *args_;
  MpsCommandBuffer_t command_buffer_;
  MpsComputeCommandEncoder_t encoder_;
  bool committed_ = false;
};

} // namespace orteaf::internal::kernel::mps

#endif // ORTEAF_ENABLE_MPS
