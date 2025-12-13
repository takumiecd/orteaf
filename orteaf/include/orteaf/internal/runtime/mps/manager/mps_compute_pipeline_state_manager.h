#pragma once

#if ORTEAF_ENABLE_MPS

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>

#include "orteaf/internal/base/handle.h"
#include "orteaf/internal/base/lease.h"
#include "orteaf/internal/runtime/base/shared_cache_manager.h"
#include "orteaf/internal/runtime/mps/platform/mps_slow_ops.h"
#include "orteaf/internal/runtime/mps/platform/wrapper/mps_compute_pipeline_state.h"
#include "orteaf/internal/runtime/mps/platform/wrapper/mps_function.h"
#include "orteaf/internal/runtime/mps/platform/wrapper/mps_library.h"

namespace orteaf::internal::runtime::mps::manager {

enum class FunctionKeyKind : std::uint8_t {
  kNamed,
};

struct FunctionKey {
  FunctionKeyKind kind{FunctionKeyKind::kNamed};
  std::string identifier{};

  static FunctionKey Named(std::string identifier) {
    FunctionKey key{};
    key.kind = FunctionKeyKind::kNamed;
    key.identifier = std::move(identifier);
    return key;
  }

  friend bool operator==(const FunctionKey &lhs,
                         const FunctionKey &rhs) noexcept = default;
};

struct FunctionKeyHasher {
  std::size_t operator()(const FunctionKey &key) const noexcept {
    std::size_t seed = static_cast<std::size_t>(key.kind);
    constexpr std::size_t kMagic = 0x9e3779b97f4a7c15ull;
    seed ^= std::hash<std::string>{}(key.identifier) + kMagic + (seed << 6) +
            (seed >> 2);
    return seed;
  }
};

// Resource struct: holds function + pipeline_state
struct MpsPipelineResource {
  ::orteaf::internal::runtime::mps::platform::wrapper::MPSFunction_t function{
      nullptr};
  ::orteaf::internal::runtime::mps::platform::wrapper::MPSComputePipelineState_t
      pipeline_state{nullptr};
};

// Use SharedCacheState template
using MpsComputePipelineStateManagerState =
    ::orteaf::internal::runtime::base::SharedCacheState<MpsPipelineResource>;

struct MpsComputePipelineStateManagerTraits {
  using OpsType = ::orteaf::internal::runtime::mps::platform::MpsSlowOps;
  using StateType = MpsComputePipelineStateManagerState;
  static constexpr const char *Name = "MPS compute pipeline state manager";
};

class MpsComputePipelineStateManager
    : public ::orteaf::internal::runtime::base::SharedCacheManager<
          MpsComputePipelineStateManager,
          MpsComputePipelineStateManagerTraits> {
public:
  using Base = ::orteaf::internal::runtime::base::SharedCacheManager<
      MpsComputePipelineStateManager, MpsComputePipelineStateManagerTraits>;
  using SlowOps = ::orteaf::internal::runtime::mps::platform::MpsSlowOps;
  using DeviceType =
      ::orteaf::internal::runtime::mps::platform::wrapper::MPSDevice_t;
  using LibraryType =
      ::orteaf::internal::runtime::mps::platform::wrapper::MPSLibrary_t;
  using FunctionHandle = ::orteaf::internal::base::FunctionHandle;
  using PipelineLease = ::orteaf::internal::base::Lease<
      FunctionHandle,
      ::orteaf::internal::runtime::mps::platform::wrapper::
          MPSComputePipelineState_t,
      MpsComputePipelineStateManager>;

  MpsComputePipelineStateManager() = default;
  MpsComputePipelineStateManager(const MpsComputePipelineStateManager &) =
      delete;
  MpsComputePipelineStateManager &
  operator=(const MpsComputePipelineStateManager &) = delete;
  MpsComputePipelineStateManager(MpsComputePipelineStateManager &&) = default;
  MpsComputePipelineStateManager &
  operator=(MpsComputePipelineStateManager &&) = default;
  ~MpsComputePipelineStateManager() = default;

  void initialize(DeviceType device, LibraryType library, SlowOps *ops,
                  std::size_t capacity);
  void shutdown();

  PipelineLease acquire(const FunctionKey &key);
  void release(PipelineLease &lease) noexcept;

private:
  void validateKey(const FunctionKey &key) const;
  void destroyResource(MpsPipelineResource &resource);

  std::unordered_map<FunctionKey, std::size_t, FunctionKeyHasher>
      key_to_index_{};
  LibraryType library_{nullptr};
  DeviceType device_{nullptr};
};

} // namespace orteaf::internal::runtime::mps::manager

#endif // ORTEAF_ENABLE_MPS
