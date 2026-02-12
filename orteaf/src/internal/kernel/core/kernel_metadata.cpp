#include "orteaf/internal/kernel/core/kernel_metadata.h"

#include "orteaf/internal/execution/cpu/resource/cpu_kernel_metadata.h"
#if ORTEAF_ENABLE_MPS
#include "orteaf/internal/execution/mps/resource/mps_kernel_metadata.h"
#endif
#if ORTEAF_ENABLE_CUDA
#include "orteaf/internal/execution/cuda/resource/cuda_kernel_metadata.h"
#endif

namespace orteaf::internal::kernel::core {

namespace detail {

template <class LeaseT>
void rebuildFromLease(const LeaseT &lease, KernelEntry &entry) {
  using T = std::decay_t<LeaseT>;
  if constexpr (std::is_same_v<T, std::monostate>) {
    // No metadata - entry remains empty
    return;
  } else if constexpr (requires(const LeaseT &l) {
                         static_cast<bool>(l);
                         l.operator->();
                       }) {
    if (!lease) {
      return;
    }
    auto *payload = lease.operator->();
    if (!payload) {
      return;
    }
    if constexpr (requires(decltype(*payload) &p, KernelEntry &e) {
                    p.rebuild(e);
                  }) {
      payload->rebuild(entry);
    }
  }
}

} // namespace detail

KernelEntry KernelMetadataLease::rebuild() const {
  KernelEntry entry;
  std::visit([&](const auto &metadata_value) {
    detail::rebuildFromLease(metadata_value, entry);
  }, lease_);
  return entry;
}

KernelMetadataLease
KernelMetadataLease::fromEntry(const KernelEntry &entry) {
  KernelMetadataLease metadata;
  std::visit(
      [&](const auto &base_lease) {
        using T = std::decay_t<decltype(base_lease)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          // No base - metadata remains empty
          return;
        } else {
          // Backend-agnostic: delegate to MetadataType
          if (!base_lease) {
            return;
          }
          auto *base_ptr = base_lease.operator->();
          if (!base_ptr) {
            return;
          }
          using BaseT = std::remove_cvref_t<decltype(*base_ptr)>;
          metadata =
              BaseT::MetadataType::buildMetadataLeaseFromBase(*base_ptr);
        }
      },
      entry.base());
  return metadata;
}

} // namespace orteaf::internal::kernel::core
