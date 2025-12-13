#pragma once

#include <cstddef>

#include "orteaf/internal/runtime/base/base_manager.h"

namespace orteaf::internal::runtime::base {

/**
 * @brief State for SharedCacheManager.
 * @tparam Resource The resource type being managed.
 */
template <typename Resource> struct SharedCacheState {
  Resource resource{};
  std::size_t use_count{0};
  bool alive{false};
};

/**
 * @brief Base manager for immutable resources with shared access.
 *
 * Resources are cached and persist until shutdown. Multiple users can
 * share access (use_count tracked). Key-based lookup in derived classes.
 *
 * @tparam Derived CRTP derived class.
 * @tparam Traits Traits class with OpsType, StateType, Name.
 */
template <typename Derived, typename Traits>
class SharedCacheManager : public BaseManager<Derived, Traits> {
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
  // Cache managers don't use free_list - resources persist until shutdown
  std::size_t allocateSlot() {
    std::size_t current_size = states_.size();
    states_.resize(current_size + 1);
    return current_size;
  }
};

} // namespace orteaf::internal::runtime::base
