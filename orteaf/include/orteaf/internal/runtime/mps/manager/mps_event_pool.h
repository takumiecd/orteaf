#pragma once

#if ORTEAF_ENABLE_MPS

#include "orteaf/internal/runtime/base/resource_pool.h"
#include "orteaf/internal/runtime/mps/platform/mps_slow_ops.h"
#include "orteaf/internal/runtime/mps/platform/wrapper/mps_event.h"
#include "orteaf/internal/base/handle.h"
#include "orteaf/internal/base/shared_lease.h"
#include "orteaf/internal/runtime/base/base_manager.h"

#include <atomic>

namespace orteaf::internal::runtime::mps::manager {

struct EventPoolState {
    std::atomic<std::size_t> ref_count;
    ::orteaf::internal::runtime::mps::platform::wrapper::MPSEvent_t event;
    uint32_t generation;
    bool alive;
    bool in_use;
};

struct EventPoolTraits {
  using StateType = EventPoolState;
  using DeviceType =
      ::orteaf::internal::runtime::mps::platform::wrapper::MPSDevice_t;
  using OpsType = ::orteaf::internal::runtime::mps::platform::MpsSlowOps;

  static constexpr const char *Name = "MPS event pool";

};


class MpsEventPool
    : public ::orteaf::internal::runtime::base::BaseManager<MpsEventPool,
                                                             EventPoolTraits> {
public:
  using Base = ::orteaf::internal::runtime::base::BaseManager<MpsEventPool,
                                                               EventPoolTraits>;
  using SlowOps = Base::Ops;
  using EventHandle = ::orteaf::internal::base::EventHandle;
  using EventLease = ::orteaf::internal::base::SharedLease<EventHandle, ::orteaf::internal::runtime::mps::platform::wrapper::MPSEvent_t, MpsEventPool>;

  MpsEventPool() = default;
  MpsEventPool(const MpsEventPool &) = delete;
  MpsEventPool &operator=(const MpsEventPool &) = delete;
  MpsEventPool(MpsEventPool &&) = default;
  MpsEventPool &operator=(MpsEventPool &&) = default;
  ~MpsEventPool() = default;

  void initialize(::orteaf::internal::runtime::mps::platform::wrapper::MPSDevice_t device,
                  SlowOps *slow_ops, std::size_t capacity);

  void shutdown();

  EventLease acquire();
  EventLease acquire(EventHandle handle);

  void release(EventLease &lease) noexcept;

#if ORTEAF_ENABLE_TEST
  struct DebugState {
    bool alive{false};
    std::uint32_t generation{0};
    std::size_t growth_chunk_size{0};
  };

  DebugState debugState(::orteaf::internal::base::EventHandle handle) const;
#endif

private:
  void release(EventHandle &handle);
};

} // namespace orteaf::internal::runtime::mps::manager

#endif // ORTEAF_ENABLE_MPS
