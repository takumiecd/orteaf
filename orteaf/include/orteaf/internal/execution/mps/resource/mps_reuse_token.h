#pragma once

#if ORTEAF_ENABLE_MPS

#include <cstddef>
#include <utility>

#include <orteaf/internal/base/small_vector.h>
#include <orteaf/internal/execution/mps/resource/mps_fence_token.h>
#include <orteaf/internal/execution/mps/resource/mps_reuse_hazard.h>

namespace orteaf::internal::execution::mps::resource {

class MpsReuseToken {
public:
  using Hazard = MpsReuseHazard;
  static constexpr std::size_t kInlineCapacity = 4;

  static MpsReuseToken fromFenceToken(MpsFenceToken &&token) {
    MpsReuseToken reuse_token;
    for (const auto &lease : token) {
      auto *payload = lease.payloadPtr();
      Hazard hazard;
      if (payload != nullptr) {
        hazard.setCommandQueueHandle(payload->commandQueueHandle());
        hazard.setCommandBuffer(payload->commandBuffer());
      }
      reuse_token.addOrReplaceHazard(std::move(hazard));
    }
    token.clear();
    return reuse_token;
  }

  MpsReuseToken() = default;

  explicit MpsReuseToken(MpsFenceToken &&token)
      : MpsReuseToken(fromFenceToken(std::move(token))) {}

  MpsReuseToken(const MpsReuseToken &) = delete;
  MpsReuseToken &operator=(const MpsReuseToken &) = delete;
  MpsReuseToken(MpsReuseToken &&) noexcept = default;
  MpsReuseToken &operator=(MpsReuseToken &&) noexcept = default;
  ~MpsReuseToken() = default;

  bool empty() const noexcept { return hazards_.empty(); }
  std::size_t size() const noexcept { return hazards_.size(); }

  // Add a hazard, replacing any existing hazard with the same command queue id.
  void addOrReplaceHazard(Hazard &&hazard) {
    const auto queue_handle = hazard.commandQueueHandle();
    for (std::size_t i = 0; i < hazards_.size(); ++i) {
      if (hazards_[i].commandQueueHandle() == queue_handle) {
        hazards_[i] = std::move(hazard);
        return;
      }
    }
    hazards_.pushBack(std::move(hazard));
  }

  void clear() noexcept { hazards_.clear(); }

  const Hazard &operator[](std::size_t index) const noexcept {
    return hazards_[index];
  }
  Hazard &operator[](std::size_t index) noexcept { return hazards_[index]; }

  const Hazard *begin() const noexcept { return hazards_.begin(); }
  const Hazard *end() const noexcept { return hazards_.end(); }
  Hazard *begin() noexcept { return hazards_.begin(); }
  Hazard *end() noexcept { return hazards_.end(); }

private:
  ::orteaf::internal::base::SmallVector<Hazard, kInlineCapacity> hazards_{};
};

} // namespace orteaf::internal::execution::mps::resource

#endif // ORTEAF_ENABLE_MPS
