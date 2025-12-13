#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "orteaf/internal/runtime/base/base_manager.h"

namespace orteaf::internal::runtime::base {

/**
 * @brief State for SharedPoolManager.
 * @tparam Resource The resource type being managed.
 * @tparam Generation The generation type for stale handle detection.
 */
template <typename Resource, typename Generation = std::uint32_t>
struct SharedPoolState {
  std::atomic<std::size_t> ref_count{0};
  Resource resource{};
  Generation generation{0};
  bool alive{false};
  bool in_use{false};

  SharedPoolState() = default;
  SharedPoolState(const SharedPoolState &) = delete;
  SharedPoolState &operator=(const SharedPoolState &) = delete;
  SharedPoolState(SharedPoolState &&other) noexcept
      : ref_count(other.ref_count.load(std::memory_order_relaxed)),
        resource(other.resource), generation(other.generation),
        alive(other.alive), in_use(other.in_use) {
    other.resource = {};
    other.alive = false;
    other.in_use = false;
  }
  SharedPoolState &operator=(SharedPoolState &&other) noexcept {
    if (this != &other) {
      ref_count.store(other.ref_count.load(std::memory_order_relaxed),
                      std::memory_order_relaxed);
      resource = other.resource;
      generation = other.generation;
      alive = other.alive;
      in_use = other.in_use;
      other.resource = {};
      other.alive = false;
      other.in_use = false;
    }
    return *this;
  }
};

/**
 * @brief Base manager for reusable resources with shared access.
 *
 * Resources can be acquired by multiple users (ref-counted) and returned
 * to the pool when ref_count reaches zero.
 *
 * @tparam Derived CRTP derived class.
 * @tparam Traits Traits class with OpsType, StateType, Name.
 */
template <typename Derived, typename Traits>
class SharedPoolManager : public BaseManager<Derived, Traits> {
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
      State &state = states_[index];
      state.in_use = false;
      ++state.generation;
      Base::free_list_.pushBack(index);
    }
  }
};

} // namespace orteaf::internal::runtime::base
