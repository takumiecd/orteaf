#pragma once

#include <concepts>
#include <type_traits>
#include <utility>
#include <variant>

#include "orteaf/internal/execution/cpu/resource/cpu_kernel_metadata.h"
#if ORTEAF_ENABLE_MPS
#include "orteaf/internal/execution/mps/manager/mps_kernel_metadata_manager.h"
#include "orteaf/internal/execution/mps/resource/mps_kernel_metadata.h"
#endif
#include "orteaf/internal/kernel/core/kernel_entry.h"

namespace orteaf::internal::kernel::core {

class KernelMetadataLease;

namespace detail {

template <class LeaseT>
concept KernelMetadataPayloadRebuildable = requires(
    const LeaseT &lease,
    ::orteaf::internal::kernel::core::KernelEntry &entry) {
  { static_cast<bool>(lease) } -> std::same_as<bool>;
  { lease.operator->()->rebuildKernelEntry(entry) } -> std::same_as<void>;
};

template <class BaseT>
concept HasMetadataType = requires {
  typename BaseT::MetadataType;
};

template <class LeaseT>
concept KernelMetadataFromEntryBuildable =
    requires(const LeaseT &lease) {
      { static_cast<bool>(lease) } -> std::same_as<bool>;
    } && requires(const LeaseT &lease_value) {
      requires HasMetadataType<std::remove_cvref_t<decltype(*lease_value.operator->())>>;
      { std::remove_cvref_t<decltype(*lease_value.operator->())>::MetadataType::buildMetadataLeaseFromBase(*lease_value.operator->()) } ->
          std::same_as<::orteaf::internal::kernel::core::KernelMetadataLease>;
    };
} // namespace detail

/**
 * @brief Type-erased kernel metadata lease.
 */
class KernelMetadataLease {
public:
  using CpuKernelMetadata =
      ::orteaf::internal::execution::cpu::resource::CpuKernelMetadata;

#if ORTEAF_ENABLE_MPS
  using MpsKernelMetadataLease =
      ::orteaf::internal::execution::mps::manager::MpsKernelMetadataManager::
          MpsKernelMetadataLease;
#endif

  using Variant = std::variant<
      std::monostate,
      CpuKernelMetadata
#if ORTEAF_ENABLE_MPS
      ,
      MpsKernelMetadataLease
#endif
      >;

  using ExecuteFunc = KernelEntry::ExecuteFunc;

  KernelMetadataLease() = default;

  explicit KernelMetadataLease(Variant lease) noexcept
      : lease_(std::move(lease)) {}

  Variant &lease() noexcept { return lease_; }
  const Variant &lease() const noexcept { return lease_; }

  void setLease(Variant lease) noexcept { lease_ = std::move(lease); }

  ExecuteFunc execute() const noexcept { return execute_; }
  void setExecute(ExecuteFunc execute) noexcept { execute_ = execute; }

  ::orteaf::internal::kernel::core::KernelEntry rebuild() const {
    ::orteaf::internal::kernel::core::KernelEntry entry;
    std::visit(
        [&](const auto &lease_value) {
          using LeaseT = std::decay_t<decltype(lease_value)>;
          if constexpr (detail::KernelMetadataPayloadRebuildable<LeaseT>) {
            if (!lease_value) {
              return;
            }
            auto *payload = lease_value.operator->();
            if (!payload) {
              return;
            }
            payload->rebuildKernelEntry(entry);
          }
        },
        lease_);
    entry.setExecute(execute_);
    return entry;
  }

  static KernelMetadataLease fromEntry(
      const ::orteaf::internal::kernel::core::KernelEntry &entry) {
    KernelMetadataLease metadata;
    metadata.setExecute(entry.execute());
    std::visit(
        [&](const auto &lease_value) {
          using LeaseT = std::decay_t<decltype(lease_value)>;
          if constexpr (detail::KernelMetadataFromEntryBuildable<LeaseT>) {
            if (!lease_value) {
              return;
            }
            auto *base_ptr = lease_value.operator->();
            if (!base_ptr) {
              return;
            }
            using BaseT = std::remove_cvref_t<decltype(*base_ptr)>;
            metadata = BaseT::MetadataType::buildMetadataLeaseFromBase(*base_ptr);
            metadata.setExecute(entry.execute());
          }
        },
        entry.base());
    return metadata;
  }

private:
  Variant lease_{};
  ExecuteFunc execute_{nullptr};
};

} // namespace orteaf::internal::kernel::core
