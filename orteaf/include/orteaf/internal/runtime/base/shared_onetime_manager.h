#pragma once

#include <atomic>
#include <cstddef>

#include "orteaf/internal/runtime/base/base_manager.h"

namespace orteaf::internal::runtime::base {

/**
 * @brief State for SharedOneTimeManager.
 * @tparam Resource The resource type being managed.
 */
template <typename Resource> struct SharedOneTimeState {
  std::atomic<std::size_t> ref_count{0};
  Resource resource{};
  bool alive{false};

  SharedOneTimeState() = default;
  SharedOneTimeState(const SharedOneTimeState &) = delete;
  SharedOneTimeState &operator=(const SharedOneTimeState &) = delete;
  SharedOneTimeState(SharedOneTimeState &&other) noexcept
      : ref_count(other.ref_count.load(std::memory_order_relaxed)),
        resource(other.resource), alive(other.alive) {
    other.resource = {};
    other.alive = false;
  }
  SharedOneTimeState &operator=(SharedOneTimeState &&other) noexcept {
    if (this != &other) {
      ref_count.store(other.ref_count.load(std::memory_order_relaxed),
                      std::memory_order_relaxed);
      resource = other.resource;
      alive = other.alive;
      other.resource = {};
      other.alive = false;
    }
    return *this;
  }
};

/**
 * @brief Base manager for non-reusable resources with shared access.
 *
 * Resources are destroyed when ref_count reaches zero. Not reused, not cached.
 *
 * @tparam Derived CRTP derived class.
 * @tparam Traits Traits class with OpsType, StateType, Name.
 */
template <typename Derived, typename Traits>
class SharedOneTimeManager : public BaseManager<Derived, Traits> {
public:
  using Base = BaseManager<Derived, Traits>;
  using Ops = typename Traits::OpsType;
  using State = typename Traits::StateType;

  using Base::ensureInitialized;
  using Base::growth_chunk_size_;
  using Base::initialized_;
  using Base::ops_;
  using Base::states_;

protected:
  // OneTime managers: resource destroyed when ref_count=0, slot recycled
  void releaseSlotAndDestroy(std::size_t index) {
    if (index < states_.size()) {
      State &state = states_[index];
      // Derived class should destroy the resource before calling this
      state.resource = {};
      state.alive = false;
      Base::free_list_.pushBack(index);
    }
  }
};

} // namespace orteaf::internal::runtime::base
