#pragma once

#include <cstddef>

#include "orteaf/internal/runtime/base/base_manager.h"

namespace orteaf::internal::runtime::base {

/**
 * @brief State for ExclusiveOneTimeManager.
 * @tparam Resource The resource type being managed.
 */
template <typename Resource> struct ExclusiveOneTimeState {
  Resource resource{};
  bool alive{false};
};

/**
 * @brief Base manager for non-reusable resources with exclusive access.
 *
 * Resources are destroyed when released. Not reused, not cached.
 *
 * @tparam Derived CRTP derived class.
 * @tparam Traits Traits class with OpsType, StateType, Name.
 */
template <typename Derived, typename Traits>
class ExclusiveOneTimeManager : public BaseManager<Derived, Traits> {
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
  // OneTime managers: resource destroyed on release, slot recycled
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
