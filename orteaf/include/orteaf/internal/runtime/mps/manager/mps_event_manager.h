#pragma once

#if ORTEAF_ENABLE_MPS

#include "orteaf/internal/base/handle.h"
#include "orteaf/internal/base/shared_lease.h"
#include "orteaf/internal/runtime/base/resource_manager.h"
#include "orteaf/internal/runtime/mps/platform/mps_slow_ops.h"
#include "orteaf/internal/runtime/mps/platform/wrapper/mps_event.h"

#include <atomic>

namespace orteaf::internal::runtime::mps::manager {

struct EventPoolTraits {
  using ResourceType =
      ::orteaf::internal::runtime::mps::platform::wrapper::MPSEvent_t;
  using StateType =
      ::orteaf::internal::runtime::base::GenerationalPoolState<ResourceType>;
  using DeviceType =
      ::orteaf::internal::runtime::mps::platform::wrapper::MPSDevice_t;
  using OpsType = ::orteaf::internal::runtime::mps::platform::MpsSlowOps;
  using HandleType = ::orteaf::internal::base::EventHandle;

  static constexpr const char *Name = "MPS event pool";

  static ResourceType create(OpsType *ops, DeviceType device) {
    return ops->createEvent(device);
  }

  static void destroy(OpsType *ops, ResourceType resource) {
    if (resource != nullptr) {
      ops->destroyEvent(resource);
    }
  }
};

class MpsEventManager
    : public ::orteaf::internal::runtime::base::ResourceManager<
          MpsEventManager, EventPoolTraits> {
public:
  using Base =
      ::orteaf::internal::runtime::base::ResourceManager<MpsEventManager,
                                                         EventPoolTraits>;
  using SlowOps = Base::Ops;
  using DeviceType = Base::Device;
  using EventHandle = Base::ResourceHandle;
  using EventLease = Base::ResourceLease;

  MpsEventManager() = default;
  MpsEventManager(const MpsEventManager &) = delete;
  MpsEventManager &operator=(const MpsEventManager &) = delete;
  MpsEventManager(MpsEventManager &&) = default;
  MpsEventManager &operator=(MpsEventManager &&) = default;
  ~MpsEventManager() = default;

  // Base class provides initialize, shutdown, acquire(s), release(s)
  // and debugState.
};

} // namespace orteaf::internal::runtime::mps::manager

#endif // ORTEAF_ENABLE_MPS
