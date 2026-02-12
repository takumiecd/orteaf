#pragma once

#if ORTEAF_ENABLE_MPS

#include <cstddef>
#include <type_traits>
#include <utility>

#include "orteaf/internal/execution/mps/platform/mps_fast_ops.h"
#include "orteaf/internal/execution/mps/platform/wrapper/mps_size.h"
#include "orteaf/internal/execution/mps/resource/mps_kernel_base.h"
#include "orteaf/internal/execution_context/mps/context.h"
#include "orteaf/internal/kernel/mps/mps_kernel_session_sync_ops.h"
#include "orteaf/internal/kernel/param/param.h"
#include "orteaf/internal/kernel/param/param_id.h"
#include "orteaf/internal/kernel/storage/operand_id.h"
#include "orteaf/internal/kernel/storage/role.h"
#include "orteaf/internal/kernel/storage/storage_binding.h"
#include "orteaf/internal/storage/mps/mps_storage.h"

namespace orteaf::internal::kernel {
template <ParamId ID, typename T> struct Field;
template <OperandId ID, Role AccessRole> struct StorageField;
} // namespace orteaf::internal::kernel

namespace orteaf::internal::kernel::mps {

using MpsKernelBase = ::orteaf::internal::execution::mps::resource::MpsKernelBase;

struct MpsKernelSessionOps {
  using PipelineLease = MpsKernelBase::PipelineLease;
  using MpsSize_t =
      ::orteaf::internal::execution::mps::platform::wrapper::MpsSize_t;
  using MpsCommandBuffer_t =
      ::orteaf::internal::execution::mps::platform::wrapper::MpsCommandBuffer_t;
  using MpsComputeCommandEncoder_t = ::orteaf::internal::execution::mps::
      platform::wrapper::MpsComputeCommandEncoder_t;

  template <std::size_t... Is> using Indices = std::index_sequence<Is...>;

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
    return ::orteaf::internal::execution::mps::platform::MpsFastOps::
        createCommandBuffer(queue_resource->queue());
  }

  static MpsComputeCommandEncoder_t
  createComputeCommandEncoder(MpsCommandBuffer_t command_buffer) {
    if (command_buffer == nullptr) {
      return nullptr;
    }
    return ::orteaf::internal::execution::mps::platform::MpsFastOps::
        createComputeCommandEncoder(command_buffer);
  }

  static void
  destroyComputeCommandEncoder(MpsComputeCommandEncoder_t encoder) {
    if (encoder == nullptr) {
      return;
    }
    ::orteaf::internal::execution::mps::platform::MpsFastOps::
        destroyComputeCommandEncoder(encoder);
  }

  static void destroyCommandBuffer(MpsCommandBuffer_t command_buffer) {
    if (command_buffer == nullptr) {
      return;
    }
    ::orteaf::internal::execution::mps::platform::MpsFastOps::
        destroyCommandBuffer(command_buffer);
  }

  static void endEncoding(MpsComputeCommandEncoder_t encoder) {
    if (encoder == nullptr) {
      return;
    }
    ::orteaf::internal::execution::mps::platform::MpsFastOps::endEncoding(
        encoder);
  }

  static void commit(MpsCommandBuffer_t command_buffer) {
    if (command_buffer == nullptr) {
      return;
    }
    ::orteaf::internal::execution::mps::platform::MpsFastOps::commit(
        command_buffer);
  }

  static void waitUntilCompleted(MpsCommandBuffer_t command_buffer) {
    if (command_buffer == nullptr) {
      return;
    }
    ::orteaf::internal::execution::mps::platform::MpsFastOps::
        waitUntilCompleted(command_buffer);
  }

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
    ::orteaf::internal::execution::mps::platform::MpsFastOps::setBuffer(
        encoder, buffer, offset, index);
  }

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

  template <std::size_t... Is, typename... Fields>
  static void bindStoragesAt(MpsComputeCommandEncoder_t encoder,
                             std::index_sequence<Is...>,
                             const Fields &...fields) {
    static_assert(sizeof...(Is) == sizeof...(Fields),
                  "Number of indices must match number of fields");
    (setBuffer(encoder, fields, Is), ...);
  }

  template <std::size_t... Is, typename... Fields>
  static void bindParamsAt(MpsComputeCommandEncoder_t encoder,
                           std::index_sequence<Is...>,
                           const Fields &...fields) {
    static_assert(sizeof...(Is) == sizeof...(Fields),
                  "Number of indices must match number of fields");
    (setParam(encoder, fields, Is), ...);
  }

  static void setBytes(MpsComputeCommandEncoder_t encoder, const void *bytes,
                       std::size_t length, std::size_t index) {
    if (encoder == nullptr || bytes == nullptr) {
      return;
    }
    ::orteaf::internal::execution::mps::platform::MpsFastOps::setBytes(
        encoder, bytes, length, index);
  }

  static void setParam(MpsComputeCommandEncoder_t encoder,
                       const ::orteaf::internal::kernel::Param &param,
                       std::size_t index) {
    if (encoder == nullptr) {
      return;
    }
    param.visit([encoder, index](const auto &value) {
      using T = std::decay_t<decltype(value)>;
      ::orteaf::internal::execution::mps::platform::MpsFastOps::setBytes(
          encoder, &value, sizeof(T), index);
    });
  }

  template <::orteaf::internal::kernel::ParamId ID, typename T>
  static void setParam(MpsComputeCommandEncoder_t encoder,
                       const ::orteaf::internal::kernel::Field<ID, T> &field,
                       std::size_t index) {
    if (encoder == nullptr) {
      return;
    }
    setBytes(encoder, &field.value, sizeof(T), index);
  }

  static void setPipelineState(MpsComputeCommandEncoder_t encoder,
                               const PipelineLease &pipeline) {
    if (encoder == nullptr || !pipeline) {
      return;
    }
    auto *resource = pipeline.operator->();
    if (resource == nullptr || resource->pipelineState() == nullptr) {
      return;
    }
    ::orteaf::internal::execution::mps::platform::MpsFastOps::setPipelineState(
        encoder, resource->pipelineState());
  }

  static void dispatchThreadgroups(MpsComputeCommandEncoder_t encoder,
                                   MpsSize_t threadgroups,
                                   MpsSize_t threads_per_threadgroup) {
    if (encoder == nullptr) {
      return;
    }
    ::orteaf::internal::execution::mps::platform::MpsFastOps::
        dispatchThreadgroups(encoder, threadgroups, threads_per_threadgroup);
  }

  static void dispatchThreads(MpsComputeCommandEncoder_t encoder,
                              MpsSize_t threads_per_grid,
                              MpsSize_t threads_per_threadgroup) {
    if (encoder == nullptr) {
      return;
    }
    ::orteaf::internal::execution::mps::platform::MpsFastOps::dispatchThreads(
        encoder, threads_per_grid, threads_per_threadgroup);
  }

  static double getGPUStartTime(MpsCommandBuffer_t command_buffer) {
    return ::orteaf::internal::execution::mps::platform::MpsFastOps::
        getGPUStartTime(command_buffer);
  }

  static double getGPUEndTime(MpsCommandBuffer_t command_buffer) {
    return ::orteaf::internal::execution::mps::platform::MpsFastOps::
        getGPUEndTime(command_buffer);
  }

  static double getGPUDuration(MpsCommandBuffer_t command_buffer) {
    return ::orteaf::internal::execution::mps::platform::MpsFastOps::
        getGPUDuration(command_buffer);
  }

  static MpsSize_t makeGridSize(std::size_t width, std::size_t height = 1,
                                std::size_t depth = 1) {
    return makeSize(width, height, depth);
  }

  static MpsSize_t makeThreadsPerThreadgroup(std::size_t width,
                                             std::size_t height = 1,
                                             std::size_t depth = 1) {
    return makeSize(width, height, depth);
  }

  static MpsSize_t makeSize(std::size_t width, std::size_t height = 1,
                            std::size_t depth = 1) {
    return ::orteaf::internal::execution::mps::platform::MpsFastOps::makeSize(
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

  template <typename... Fields>
  static void waitAllStorageDependencies(MpsComputeCommandEncoder_t encoder,
                                         Fields &...fields) {
    ::orteaf::internal::kernel::mps::detail::MpsKernelSessionSyncOps::
        waitAllStorageDependencies(encoder, fields...);
  }

  template <typename... Fields>
  [[nodiscard]] static bool updateAllStorageTokens(
      ::orteaf::internal::execution_context::mps::Context &context,
      MpsCommandBuffer_t command_buffer, MpsComputeCommandEncoder_t encoder,
      Fields &...fields) {
    return ::orteaf::internal::kernel::mps::detail::MpsKernelSessionSyncOps::
        updateAllStorageTokens(context, command_buffer, encoder, fields...);
  }
};

} // namespace orteaf::internal::kernel::mps

#endif // ORTEAF_ENABLE_MPS
