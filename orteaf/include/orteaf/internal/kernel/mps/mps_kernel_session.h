#pragma once

#if ORTEAF_ENABLE_MPS

#include <cstddef>
#include <optional>
#include <utility>

#include "orteaf/internal/execution/mps/resource/mps_kernel_base.h"
#include "orteaf/internal/execution_context/mps/context.h"
#include "orteaf/internal/kernel/core/kernel_args.h"
#include "orteaf/internal/kernel/mps/mps_kernel_session_ops.h"

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
  using Ops = MpsKernelSessionOps;
  using MpsSize_t = Ops::MpsSize_t;
  using MpsCommandBuffer_t = Ops::MpsCommandBuffer_t;
  using MpsComputeCommandEncoder_t = Ops::MpsComputeCommandEncoder_t;

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

    auto command_buffer = Ops::createCommandBuffer(*context);
    if (!command_buffer) {
      return std::nullopt;
    }

    auto encoder = Ops::createComputeCommandEncoder(command_buffer);
    if (!encoder) {
      return std::nullopt;
    }

    auto pipeline =
        base.getPipelineLease(context->device.payloadHandle(), kernel_index);
    if (!pipeline) {
      Ops::endEncoding(encoder);
      return std::nullopt;
    }
    Ops::setPipelineState(encoder, pipeline);

    return MpsKernelSession(args, command_buffer, encoder);
  }

  MpsKernelSession(MpsKernelSession &&other) noexcept
      : args_(other.args_), command_buffer_(other.command_buffer_),
        encoder_(other.encoder_), committed_(other.committed_) {
    other.committed_ = true;
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

  MpsKernelSession(const MpsKernelSession &) = delete;
  MpsKernelSession &operator=(const MpsKernelSession &) = delete;

  ~MpsKernelSession() { finish(); }

  template <typename... Fields> void waitDependencies(Fields &...fields) {
    Ops::waitAllStorageDependencies(encoder_, fields...);
  }

  template <std::size_t... Is, typename... Fields>
  void bindStorages(const Fields &...fields) {
    Ops::bindStoragesAt(encoder_, std::index_sequence<Is...>{}, fields...);
  }

  template <std::size_t... Is, typename... Fields>
  void bindParams(const Fields &...fields) {
    Ops::bindParamsAt(encoder_, std::index_sequence<Is...>{}, fields...);
  }

  void dispatch1D(std::size_t count, std::size_t threads_per_group = 256) {
    auto grid = Ops::makeGridSize(count);
    auto tpg = Ops::makeThreadsPerThreadgroup(threads_per_group);
    Ops::dispatchThreads(encoder_, grid, tpg);
  }

  void dispatch(MpsSize_t grid, MpsSize_t threads_per_group) {
    Ops::dispatchThreadgroups(encoder_, grid, threads_per_group);
  }

  template <typename... Fields> [[nodiscard]] bool updateTokens(Fields &...fields) {
    auto *context =
        args_->context()
            .tryAs<::orteaf::internal::execution_context::mps::Context>();
    if (context == nullptr) {
      return false;
    }
    return Ops::updateAllStorageTokens(*context, command_buffer_, encoder_,
                                       fields...);
  }

  void setBytes(const void *bytes, std::size_t length, std::size_t index) {
    Ops::setBytes(encoder_, bytes, length, index);
  }

  MpsComputeCommandEncoder_t encoder() const noexcept { return encoder_; }
  MpsCommandBuffer_t commandBuffer() const noexcept { return command_buffer_; }

private:
  MpsKernelSession(::orteaf::internal::kernel::KernelArgs &args,
                   MpsCommandBuffer_t command_buffer,
                   MpsComputeCommandEncoder_t encoder)
      : args_(&args), command_buffer_(command_buffer), encoder_(encoder) {}

  void finish() {
    if (!committed_) {
      Ops::endEncoding(encoder_);
      Ops::commit(command_buffer_);
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
