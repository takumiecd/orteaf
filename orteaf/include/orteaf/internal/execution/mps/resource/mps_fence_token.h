#pragma once

#if ORTEAF_ENABLE_MPS

#include <cstddef>
#include <utility>

#include <orteaf/internal/base/handle.h>
#include <orteaf/internal/base/small_vector.h>
#include <orteaf/internal/execution/mps/manager/mps_fence_manager.h>

namespace orteaf::internal::execution::mps::resource {

class MpsFenceToken {
public:
  using StrongFenceLease = ::orteaf::internal::execution::mps::manager::
      MpsFenceManager::StrongFenceLease;

  static constexpr std::size_t kInlineCapacity = 4;

  MpsFenceToken() = default;
  MpsFenceToken(const MpsFenceToken &) = default;
  MpsFenceToken &operator=(const MpsFenceToken &) = default;
  MpsFenceToken(MpsFenceToken &&) noexcept = default;
  MpsFenceToken &operator=(MpsFenceToken &&) noexcept = default;
  ~MpsFenceToken() = default;

  bool empty() const noexcept { return leases_.empty(); }
  std::size_t size() const noexcept { return leases_.size(); }

  // Add a lease, replacing any existing lease with the same command queue id.
  void addOrReplaceLease(StrongFenceLease &&lease) {
    auto *payload = lease.operator->();
    if (payload != nullptr) {
      const auto queue_handle = payload->commandQueueHandle();
      for (std::size_t i = 0; i < leases_.size(); ++i) {
        auto *existing_payload = leases_[i].operator->();
        if (existing_payload != nullptr &&
            existing_payload->commandQueueHandle() == queue_handle) {
          leases_[i] = std::move(lease);
          return;
        }
      }
    }
    leases_.pushBack(std::move(lease));
  }

  void clear() noexcept { leases_.clear(); }

  const StrongFenceLease &operator[](std::size_t index) const noexcept {
    return leases_[index];
  }
  StrongFenceLease &operator[](std::size_t index) noexcept {
    return leases_[index];
  }

  const StrongFenceLease *begin() const noexcept { return leases_.begin(); }
  const StrongFenceLease *end() const noexcept { return leases_.end(); }
  StrongFenceLease *begin() noexcept { return leases_.begin(); }
  StrongFenceLease *end() noexcept { return leases_.end(); }

private:
  ::orteaf::internal::base::SmallVector<StrongFenceLease, kInlineCapacity>
      leases_{};
};

} // namespace orteaf::internal::execution::mps::resource

#endif // ORTEAF_ENABLE_MPS
