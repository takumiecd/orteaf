#pragma once

#if ORTEAF_ENABLE_MPS

#include <cstddef>
#include <type_traits>
#include <utility>

#include "orteaf/internal/diagnostics/error/error.h"
#include "orteaf/internal/execution/mps/manager/mps_fence_manager.h"
#include "orteaf/internal/execution/mps/platform/mps_fast_ops.h"
#include "orteaf/internal/execution_context/mps/context.h"
#include "orteaf/internal/kernel/core/access.h"
#include "orteaf/internal/kernel/storage/storage_binding.h"
#include "orteaf/internal/storage/registry/storage_types.h"

namespace orteaf::internal::kernel::mps::detail {

struct MpsKernelSessionSyncOps {
  using MpsCommandBuffer_t =
      ::orteaf::internal::execution::mps::platform::wrapper::MpsCommandBuffer_t;
  using MpsComputeCommandEncoder_t = ::orteaf::internal::execution::mps::
      platform::wrapper::MpsComputeCommandEncoder_t;

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

private:
  static ::orteaf::internal::execution::mps::manager::MpsFenceManager::
      StrongFenceLease
  acquireFence(
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
    ::orteaf::internal::execution::mps::platform::MpsFastOps::updateFence(
        encoder, fence);
  }

  static void
  waitForFence(MpsComputeCommandEncoder_t encoder,
               ::orteaf::internal::execution::mps::platform::wrapper::MpsFence_t
                   fence) {
    if (encoder == nullptr || fence == nullptr) {
      return;
    }
    ::orteaf::internal::execution::mps::platform::MpsFastOps::waitForFence(
        encoder, fence);
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
};

} // namespace orteaf::internal::kernel::mps::detail

#endif // ORTEAF_ENABLE_MPS
