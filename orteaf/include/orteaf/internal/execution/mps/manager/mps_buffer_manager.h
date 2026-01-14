#pragma once

#if ORTEAF_ENABLE_MPS

#include <cstddef>
#include <string>
#include <utility>

#include "orteaf/internal/base/lease/control_block/strong.h"
#include "orteaf/internal/base/lease/strong_lease.h"
#include "orteaf/internal/base/manager/pool_manager.h"
#include "orteaf/internal/base/pool/slot_pool.h"
#include "orteaf/internal/diagnostics/error/error.h"
#include "orteaf/internal/execution/mps/resource/mps_buffer.h"
#include "orteaf/internal/execution/base/execution_traits.h"
#include "orteaf/internal/execution/execution.h"
#include "orteaf/internal/execution/mps/mps_handles.h"
#include "orteaf/internal/execution/mps/manager/mps_library_manager.h"
#include "orteaf/internal/execution/mps/platform/wrapper/mps_device.h"
#include "orteaf/internal/execution/mps/resource/mps_buffer_view.h"

namespace orteaf::internal::execution::mps::manager {

struct HeapPayloadPoolTraits;

using ::orteaf::internal::execution::Execution;

// Forward declaration
template <typename ResourceT> class MpsBufferManager;

// ============================================================================
// BufferPayloadPoolTraits - Defines Payload/Handle/Request/Context for SlotPool
// ============================================================================
template <typename ResourceT> struct BufferPayloadPoolTraitsT {
  using MpsBuffer =
      ::orteaf::internal::execution::mps::resource::MpsBuffer;
  using Payload = MpsBuffer;
  using Handle = ::orteaf::internal::execution::mps::MpsBufferHandle;
  using BufferViewHandle =
      ::orteaf::internal::execution::mps::MpsBufferViewHandle;
  using Resource = ResourceT;

  struct Request {
    std::size_t size{0};
    std::size_t alignment{0};
    Handle handle{Handle::invalid()};
  };

  struct Context {
    Resource *resource{nullptr};
  };

  static bool create(Payload &payload, const Request &request,
                     const Context &context) {
    if (context.resource == nullptr) {
      return false;
    }
    if (request.size == 0) {
      // Zero-size is valid but results in invalid buffer
      payload = Payload{};
      return true;
    }
    if (!request.handle.isValid()) {
      return false;
    }
    auto view = context.resource->allocate(request.size, request.alignment);
    if (!view) {
      return false;
    }
    payload = Payload{BufferViewHandle{request.handle.index}, std::move(view)};
    return true;
  }

  static void destroy(Payload &payload, const Request &request,
                      const Context &context) {
    if (!payload.valid()) {
      return;
    }
    if (context.resource != nullptr) {
      context.resource->deallocate(payload.view, request.size,
                                   request.alignment);
    }
    payload = Payload{};
  }
};

// ============================================================================
// PayloadPool type alias
// ============================================================================
template <typename ResourceT>
using BufferPayloadPoolT = ::orteaf::internal::base::pool::SlotPool<
    BufferPayloadPoolTraitsT<ResourceT>>;

// ============================================================================
// ControlBlock type using StrongControlBlock
// ============================================================================
template <typename ResourceT>
using MpsBuffer =
    ::orteaf::internal::execution::mps::resource::MpsBuffer;

template <typename ResourceT>
using BufferControlBlockT = ::orteaf::internal::base::StrongControlBlock<
    ::orteaf::internal::execution::mps::MpsBufferHandle,
    MpsBuffer<ResourceT>,
    BufferPayloadPoolT<ResourceT>>;

// ============================================================================
// Traits for PoolManager
// ============================================================================
template <typename ResourceT> struct MpsBufferManagerTraits {
  using PayloadPool = BufferPayloadPoolT<ResourceT>;
  using ControlBlock = BufferControlBlockT<ResourceT>;
  struct ControlBlockTag {};
  using PayloadHandle = ::orteaf::internal::execution::mps::MpsBufferHandle;
  static constexpr const char *Name = "MpsBufferManager";
};

// ============================================================================
// MpsBufferManager - Templated buffer manager using PoolManager
// ============================================================================
template <typename ResourceT> class MpsBufferManager {
public:
  using Traits = MpsBufferManagerTraits<ResourceT>;
  using Core = ::orteaf::internal::base::PoolManager<Traits>;
  using MpsBuffer = MpsBuffer<ResourceT>;
  using BufferHandle = ::orteaf::internal::execution::mps::MpsBufferHandle;
  using LaunchParams = ::orteaf::internal::execution::base::ExecutionTraits<
      Execution::Mps>::KernelLaunchParams;
  using Resource = ResourceT;
  using DeviceType =
      ::orteaf::internal::execution::mps::platform::wrapper::MpsDevice_t;

  using ControlBlock = typename Core::ControlBlock;
  using ControlBlockHandle = typename Core::ControlBlockHandle;
  using ControlBlockPool = typename Core::ControlBlockPool;
  using PayloadPool = typename Core::PayloadPool;

  // Lease types
  using StrongBufferLease = typename Core::StrongLeaseType;
  // Legacy alias for compatibility
  using BufferLease = StrongBufferLease;

public:
  // HeapType deduced from Resource::Config
  using HeapType = decltype(std::declval<typename Resource::Config>().heap);

  // =========================================================================
  // Config - Public settings only
  // =========================================================================
  struct Config {
    // Buffer config
    std::size_t chunk_size{16 * 1024 * 1024};
    std::size_t min_block_size{64};
    std::size_t max_block_size{16 * 1024 * 1024};
    ::orteaf::internal::execution::mps::platform::wrapper::MpsBufferUsage_t
        usage{::orteaf::internal::execution::mps::platform::wrapper::
                  kMPSDefaultBufferUsage};
    // PoolManager config
    std::size_t control_block_capacity{0};
    std::size_t control_block_block_size{0};
    std::size_t control_block_growth_chunk_size{1};
    std::size_t payload_capacity{0};
    std::size_t payload_block_size{0};
    std::size_t payload_growth_chunk_size{1};
  };

  // =========================================================================
  // Lifecycle
  // =========================================================================
  MpsBufferManager() = default;
  MpsBufferManager(const MpsBufferManager &) = delete;
  MpsBufferManager &operator=(const MpsBufferManager &) = delete;
  MpsBufferManager(MpsBufferManager &&) = default;
  MpsBufferManager &operator=(MpsBufferManager &&) = default;
  ~MpsBufferManager() = default;

private:
  struct InternalConfig {
    Config public_config{};
    DeviceType device{nullptr};
    ::orteaf::internal::execution::mps::MpsDeviceHandle device_handle{};
    HeapType heap{nullptr};
    MpsLibraryManager *library_manager{nullptr};
  };

  void configure(const InternalConfig &config) {
    shutdown();

    if (config.device == nullptr) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
          std::string(Traits::Name) + " requires a valid device");
    }
    if (config.heap == nullptr) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
          std::string(Traits::Name) + " requires a valid heap");
    }

    device_ = config.device;
    device_handle_ = config.device_handle;
    heap_ = config.heap;
    const auto &cfg = config.public_config;

    // Initialize resource
    typename Resource::Config res_cfg{};
    res_cfg.device = config.device;
    res_cfg.device_handle = config.device_handle;
    res_cfg.heap = config.heap;
    res_cfg.usage = cfg.usage;
    res_cfg.library_manager = config.library_manager;

    resource_ = Resource{};
    resource_.initialize(res_cfg);

    // Configure PoolManager + PayloadPool
    typename BufferPayloadPoolTraitsT<ResourceT>::Request request{};
    auto context = makePayloadContext();
    typename Core::template Builder<
        typename BufferPayloadPoolTraitsT<ResourceT>::Request,
        typename BufferPayloadPoolTraitsT<ResourceT>::Context>{}
        .withControlBlockCapacity(cfg.control_block_capacity)
        .withControlBlockBlockSize(cfg.control_block_block_size)
        .withControlBlockGrowthChunkSize(
            cfg.control_block_growth_chunk_size)
        .withPayloadCapacity(cfg.payload_capacity)
        .withPayloadBlockSize(cfg.payload_block_size)
        .withPayloadGrowthChunkSize(cfg.payload_growth_chunk_size)
        .withRequest(request)
        .withContext(context)
        .configure(core_);
  }

  friend struct HeapPayloadPoolTraits;

public:

  void shutdown() {
    if (!core_.isConfigured()) {
      return;
    }

    // Shutdown PoolManager (includes check and clear for both pools)
    auto context = makePayloadContext();
    typename BufferPayloadPoolTraitsT<ResourceT>::Request request{};
    core_.shutdown(request, context);

    resource_ = Resource{};
    device_ = nullptr;
    heap_ = nullptr;
  }

  // =========================================================================
  // Acquire (allocate new buffer)
  // =========================================================================
  StrongBufferLease acquire(std::size_t size, std::size_t alignment) {
    LaunchParams params{};
    return acquire(size, alignment, params);
  }

  StrongBufferLease acquire(std::size_t size, std::size_t alignment,
                            LaunchParams &params) {
    (void)params;
    core_.ensureConfigured();
    if (size == 0) {
      return {};
    }

    // Get or grow PayloadPool slot
    typename BufferPayloadPoolTraitsT<ResourceT>::Request request{size,
                                                                  alignment};
    auto context = makePayloadContext(); // Build lease for the buffer payload
    auto payload_handle = core_.acquirePayloadOrGrowAndCreate(request, context);
    if (!payload_handle.isValid()) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::OutOfRange,
          "MPS buffer manager has no available slots");
    }
    return core_.acquireStrongLease(payload_handle);
  }

#if ORTEAF_ENABLE_TEST
  void configureForTest(const Config &config, DeviceType device,
                        ::orteaf::internal::execution::mps::MpsDeviceHandle
                            device_handle,
                        HeapType heap, MpsLibraryManager *library_manager) {
    InternalConfig internal{};
    internal.public_config = config;
    internal.device = device;
    internal.device_handle = device_handle;
    internal.heap = heap;
    internal.library_manager = library_manager;
    configure(internal);
  }

  bool isConfiguredForTest() const noexcept { return core_.isConfigured(); }
  std::size_t payloadPoolSizeForTest() const noexcept {
    return core_.payloadPoolSizeForTest();
  }
  std::size_t payloadPoolCapacityForTest() const noexcept {
    return core_.payloadPoolCapacityForTest();
  }
  std::size_t payloadPoolAvailableForTest() const noexcept {
    return core_.payloadPoolAvailableForTest();
  }
  std::size_t controlBlockPoolSizeForTest() const noexcept {
    return core_.controlBlockPoolSizeForTest();
  }
  std::size_t controlBlockPoolCapacityForTest() const noexcept {
    return core_.controlBlockPoolCapacityForTest();
  }
  bool isAliveForTest(BufferHandle handle) const noexcept {
    return core_.isAlive(handle);
  }
  std::size_t payloadGrowthChunkSizeForTest() const noexcept {
    return core_.payloadGrowthChunkSize();
  }
  std::size_t controlBlockGrowthChunkSizeForTest() const noexcept {
    return core_.controlBlockGrowthChunkSize();
  }
  const ControlBlock *controlBlockForTest(ControlBlockHandle handle) const {
    return core_.controlBlockForTest(handle);
  }
#endif

private:
  typename BufferPayloadPoolTraitsT<ResourceT>::Context
  makePayloadContext() noexcept {
    typename BufferPayloadPoolTraitsT<ResourceT>::Context ctx{};
    ctx.resource = &resource_;
    return ctx;
  }

  // Runtime state
  Resource resource_{};
  DeviceType device_{nullptr};
  ::orteaf::internal::execution::mps::MpsDeviceHandle device_handle_{};
  HeapType heap_{nullptr};
  Core core_{};
};

} // namespace orteaf::internal::execution::mps::manager

// ============================================================================
// Default type alias (after namespace to avoid circular dependency)
// ============================================================================
#include "orteaf/internal/execution/allocator/resource/mps/mps_resource.h"

namespace orteaf::internal::execution::mps::manager {
using MpsResource =
    ::orteaf::internal::execution::allocator::resource::mps::MpsResource;
} // namespace orteaf::internal::execution::mps::manager

#endif // ORTEAF_ENABLE_MPS
