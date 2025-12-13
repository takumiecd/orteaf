#pragma once

#if ORTEAF_ENABLE_MPS

#include "orteaf/internal/base/handle.h"
#include "orteaf/internal/base/shared_lease.h"
#include "orteaf/internal/runtime/base/shared_pool_manager.h"
#include "orteaf/internal/runtime/mps/platform/mps_slow_ops.h"
#include "orteaf/internal/runtime/mps/platform/wrapper/mps_event.h"

namespace orteaf::internal::runtime::mps::manager {

// Use the standard SharedPoolState template
using MpsEventManagerState = ::orteaf::internal::runtime::base::SharedPoolState<
    ::orteaf::internal::runtime::mps::platform::wrapper::MPSEvent_t>;

struct MpsEventManagerTraits {
  using OpsType = ::orteaf::internal::runtime::mps::platform::MpsSlowOps;
  using StateType = MpsEventManagerState;
  static constexpr const char *Name = "MPS event manager";
};

class MpsEventManager
    : public ::orteaf::internal::runtime::base::SharedPoolManager<
          MpsEventManager, MpsEventManagerTraits> {
public:
  using Base = ::orteaf::internal::runtime::base::SharedPoolManager<
      MpsEventManager, MpsEventManagerTraits>;
  using SlowOps = ::orteaf::internal::runtime::mps::platform::MpsSlowOps;
  using DeviceType =
      ::orteaf::internal::runtime::mps::platform::wrapper::MPSDevice_t;
  using EventHandle = ::orteaf::internal::base::EventHandle;
  using EventLease = ::orteaf::internal::base::SharedLease<
      EventHandle,
      ::orteaf::internal::runtime::mps::platform::wrapper::MPSEvent_t,
      MpsEventManager>;

  MpsEventManager() = default;
  MpsEventManager(const MpsEventManager &) = delete;
  MpsEventManager &operator=(const MpsEventManager &) = delete;
  MpsEventManager(MpsEventManager &&) = default;
  MpsEventManager &operator=(MpsEventManager &&) = default;
  ~MpsEventManager() = default;

  void initialize(DeviceType device, SlowOps *ops, std::size_t capacity);
  void shutdown();

  EventLease acquire();
  EventLease acquire(EventHandle handle);
  void release(EventLease &lease) noexcept;
  void release(EventHandle handle) noexcept;

private:
  DeviceType device_{nullptr};
};

} // namespace orteaf::internal::runtime::mps::manager

#endif // ORTEAF_ENABLE_MPS
