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

  bool empty() const noexcept { 
    return !write_fence_ && read_fences_.empty(); 
  }

  std::size_t readFenceCount() const noexcept { 
    return read_fences_.size(); 
  }

  bool hasWriteFence() const noexcept {
    return static_cast<bool>(write_fence_);
  }

  // Add a read fence, replacing any existing fence with the same command queue id.
  void addReadFence(StrongFenceLease &&lease) {
    auto *payload = lease.operator->();
    if (payload != nullptr) {
      const auto queue_handle = payload->commandQueueHandle();
      for (std::size_t i = 0; i < read_fences_.size(); ++i) {
        auto *existing_payload = read_fences_[i].operator->();
        if (existing_payload != nullptr &&
            existing_payload->commandQueueHandle() == queue_handle) {
          read_fences_[i] = std::move(lease);
          return;
        }
      }
    }
    read_fences_.pushBack(std::move(lease));
  }

  // Set the write fence, replacing any existing write fence.
  void setWriteFence(StrongFenceLease &&lease) {
    write_fence_ = std::move(lease);
  }

  void clear() noexcept { 
    read_fences_.clear();
    write_fence_ = StrongFenceLease{};
  }

  // Access read fences
  const StrongFenceLease &readFence(std::size_t index) const noexcept {
    return read_fences_[index];
  }
  StrongFenceLease &readFence(std::size_t index) noexcept {
    return read_fences_[index];
  }

  // Access write fence
  const StrongFenceLease &writeFence() const noexcept {
    return write_fence_;
  }
  StrongFenceLease &writeFence() noexcept {
    return write_fence_;
  }

  // Iterators for read fences
  const StrongFenceLease *readBegin() const noexcept { return read_fences_.begin(); }
  const StrongFenceLease *readEnd() const noexcept { return read_fences_.end(); }
  StrongFenceLease *readBegin() noexcept { return read_fences_.begin(); }
  StrongFenceLease *readEnd() noexcept { return read_fences_.end(); }

private:
  // Multiple read fences (one per command queue that reads from this storage)
  ::orteaf::internal::base::SmallVector<StrongFenceLease, kInlineCapacity>
      read_fences_{};
  // Single write fence (only the last write matters for hazard tracking)
  StrongFenceLease write_fence_{};
};

} // namespace orteaf::internal::execution::mps::resource

#endif // ORTEAF_ENABLE_MPS
