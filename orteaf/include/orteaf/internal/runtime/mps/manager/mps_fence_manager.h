#pragma once

#if ORTEAF_ENABLE_MPS

#include "orteaf/internal/base/handle.h"
#include "orteaf/internal/base/shared_lease.h"
#include "orteaf/internal/runtime/base/shared_pool_manager.h"
#include "orteaf/internal/runtime/mps/platform/mps_slow_ops.h"
#include "orteaf/internal/runtime/mps/platform/wrapper/mps_fence.h"

namespace orteaf::internal::runtime::mps::manager {

// Use the standard SharedPoolState template
using MpsFenceManagerState = ::orteaf::internal::runtime::base::SharedPoolState<
    ::orteaf::internal::runtime::mps::platform::wrapper::MPSFence_t>;

struct MpsFenceManagerTraits {
  using OpsType = ::orteaf::internal::runtime::mps::platform::MpsSlowOps;
  using StateType = MpsFenceManagerState;
  static constexpr const char *Name = "MPS fence manager";
};

class MpsFenceManager
    : public ::orteaf::internal::runtime::base::SharedPoolManager<
          MpsFenceManager, MpsFenceManagerTraits> {
public:
  using Base = ::orteaf::internal::runtime::base::SharedPoolManager<
      MpsFenceManager, MpsFenceManagerTraits>;
  using SlowOps = ::orteaf::internal::runtime::mps::platform::MpsSlowOps;
  using DeviceType =
      ::orteaf::internal::runtime::mps::platform::wrapper::MPSDevice_t;
  using FenceHandle = ::orteaf::internal::base::FenceHandle;
  using FenceLease = ::orteaf::internal::base::SharedLease<
      FenceHandle,
      ::orteaf::internal::runtime::mps::platform::wrapper::MPSFence_t,
      MpsFenceManager>;

  MpsFenceManager() = default;
  MpsFenceManager(const MpsFenceManager &) = delete;
  MpsFenceManager &operator=(const MpsFenceManager &) = delete;
  MpsFenceManager(MpsFenceManager &&) = default;
  MpsFenceManager &operator=(MpsFenceManager &&) = default;
  ~MpsFenceManager() = default;

  void initialize(DeviceType device, SlowOps *ops, std::size_t capacity);
  void shutdown();

  FenceLease acquire();
  FenceLease acquire(FenceHandle handle);
  void release(FenceLease &lease) noexcept;
  void release(FenceHandle handle) noexcept;

private:
  DeviceType device_{nullptr};
};

} // namespace orteaf::internal::runtime::mps::manager

#endif // ORTEAF_ENABLE_MPS
