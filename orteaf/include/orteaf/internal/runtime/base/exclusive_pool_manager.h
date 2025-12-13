#pragma once

#include <cstddef>

#include "orteaf/internal/runtime/base/base_manager.h"

namespace orteaf::internal::runtime::base {

/**
 * @brief State for ExclusivePoolManager.
 * @tparam Resource The resource type being managed.
 */
template <typename Resource> struct ExclusivePoolState {
  Resource resource{};
  bool alive{false};
  bool in_use{false};
};

/**
 * @brief Base manager for reusable resources with exclusive access.
 *
 * Resources are acquired exclusively (one user at a time) and returned
 * to the pool for reuse after release.
 *
 * @tparam Derived CRTP derived class.
 * @tparam Traits Traits class with OpsType, StateType, Name.
 */
template <typename Derived, typename Traits>
class ExclusivePoolManager : public BaseManager<Derived, Traits> {
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
  void releaseSlot(std::size_t index) {
    if (index < states_.size()) {
      states_[index].in_use = false;
      Base::free_list_.pushBack(index);
    }
  }
};

} // namespace orteaf::internal::runtime::base
