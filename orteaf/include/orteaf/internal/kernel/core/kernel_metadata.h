#pragma once

#include <type_traits>
#include <utility>
#include <variant>

#if ORTEAF_ENABLE_MPS
#include "orteaf/internal/execution/mps/api/mps_execution_api.h"
#include "orteaf/internal/execution/mps/manager/mps_kernel_metadata_manager.h"
#endif
#include "orteaf/internal/kernel/core/kernel_entry.h"

namespace orteaf::internal::kernel::core {

/**
 * @brief Type-erased kernel metadata lease.
 */
class KernelMetadataLease {
public:
#if ORTEAF_ENABLE_MPS
  using MpsKernelMetadataLease =
      ::orteaf::internal::execution::mps::manager::MpsKernelMetadataManager::
          KernelMetadataLease;
#endif

  using Variant = std::variant<
      std::monostate
#if ORTEAF_ENABLE_MPS
      ,
      MpsKernelMetadataLease
#endif
      >;

  KernelMetadataLease() = default;

  explicit KernelMetadataLease(Variant lease) noexcept
      : lease_(std::move(lease)) {}

  Variant &lease() noexcept { return lease_; }
  const Variant &lease() const noexcept { return lease_; }

  void setLease(Variant lease) noexcept { lease_ = std::move(lease); }

  ::orteaf::internal::kernel::core::KernelEntry rebuild() const {
    ::orteaf::internal::kernel::core::KernelEntry entry;
    std::visit(
        [&](const auto &lease_value) {
          using LeaseT = std::decay_t<decltype(lease_value)>;
          if constexpr (std::is_same_v<LeaseT, std::monostate>) {
            return;
#if ORTEAF_ENABLE_MPS
          } else if constexpr (std::is_same_v<LeaseT, MpsKernelMetadataLease>) {
            if (!lease_value) {
              return;
            }
            auto *payload = lease_value.operator->();
            if (!payload) {
              return;
            }
            entry.setBase(::orteaf::internal::execution::mps::api::
                              MpsExecutionApi::acquireKernelBase(
                                  payload->keys()));
            entry.setExecute(payload->execute());
#endif
          }
        },
        lease_);
    return entry;
  }

private:
  Variant lease_{};
};

} // namespace orteaf::internal::kernel::core
