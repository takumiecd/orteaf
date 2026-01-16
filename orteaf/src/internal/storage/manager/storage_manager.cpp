#include "orteaf/internal/storage/manager/storage_manager.h"

#include <type_traits>
#include <utility>
#include <variant>

#include "orteaf/internal/diagnostics/error/error.h"

namespace orteaf::internal::storage::manager::detail {

void StoragePayloadPoolTraits::validateRequestOrThrow(const Request &request) {
  std::visit(
      [](const auto &req) {
        using RequestT = std::decay_t<decltype(req)>;
        if constexpr (std::is_same_v<RequestT, CpuStorageRequest>) {
          if (!req.device.isValid()) {
            ::orteaf::internal::diagnostics::error::throwError(
                ::orteaf::internal::diagnostics::error::OrteafErrc::
                    InvalidArgument,
                "CpuStorage request requires a valid device handle");
          }
          if (req.numel == 0) {
            ::orteaf::internal::diagnostics::error::throwError(
                ::orteaf::internal::diagnostics::error::OrteafErrc::
                    InvalidArgument,
                "CpuStorage request numel must be > 0");
          }
        }
#if ORTEAF_ENABLE_MPS
        else if constexpr (std::is_same_v<RequestT, MpsStorageRequest>) {
          if (!req.device.isValid()) {
            ::orteaf::internal::diagnostics::error::throwError(
                ::orteaf::internal::diagnostics::error::OrteafErrc::
                    InvalidArgument,
                "MpsStorage request requires a valid device handle");
          }
          if (req.numel == 0) {
            ::orteaf::internal::diagnostics::error::throwError(
                ::orteaf::internal::diagnostics::error::OrteafErrc::
                    InvalidArgument,
                "MpsStorage request numel must be > 0");
          }
        }
#endif // ORTEAF_ENABLE_MPS
      },
      request);
}

bool StoragePayloadPoolTraits::create(Payload &payload, const Request &request,
                                      const Context &) {
  return std::visit(
      [&](const auto &req) -> bool {
        using RequestT = std::decay_t<decltype(req)>;
        if constexpr (std::is_same_v<RequestT, CpuStorageRequest>) {
          auto storage = ::orteaf::internal::storage::cpu::CpuStorage::builder()
                             .withDeviceHandle(req.device)
                             .withDType(req.dtype)
                             .withNumElements(req.numel)
                             .withAlignment(req.alignment)
                             .withLayout(req.layout)
                             .build();
          payload = Payload::erase(std::move(storage));
          return true;
        }
#if ORTEAF_ENABLE_MPS
        else if constexpr (std::is_same_v<RequestT, MpsStorageRequest>) {
          auto storage = ::orteaf::internal::storage::mps::MpsStorage::builder()
                             .withDeviceHandle(req.device, req.heap_key)
                             .withDType(req.dtype)
                             .withNumElements(req.numel)
                             .withAlignment(req.alignment)
                             .withLayout(req.layout)
                             .build();
          payload = Payload::erase(std::move(storage));
          return true;
        }
#endif // ORTEAF_ENABLE_MPS
        else {
          return false;
        }
      },
      request);
}

void StoragePayloadPoolTraits::destroy(Payload &payload, const Request &,
                                       const Context &) {
  payload = Payload{};
}

} // namespace orteaf::internal::storage::manager::detail

namespace orteaf::internal::storage::manager {

void StorageManager::configure(const Config &config) {
  std::size_t payload_capacity = config.payload_capacity;
  if (payload_capacity == 0) {
    payload_capacity = 64;
  }
  std::size_t payload_block_size = config.payload_block_size;
  if (payload_block_size == 0) {
    payload_block_size = 16;
  }
  std::size_t control_block_capacity = config.control_block_capacity;
  if (control_block_capacity == 0) {
    control_block_capacity = 64;
  }
  std::size_t control_block_block_size = config.control_block_block_size;
  if (control_block_block_size == 0) {
    control_block_block_size = 16;
  }

  Request request{};
  Context context{};

  Core::Builder<Request, Context> builder{};
  builder.withControlBlockCapacity(control_block_capacity)
      .withControlBlockBlockSize(control_block_block_size)
      .withControlBlockGrowthChunkSize(config.control_block_growth_chunk_size)
      .withPayloadCapacity(payload_capacity)
      .withPayloadBlockSize(payload_block_size)
      .withPayloadGrowthChunkSize(config.payload_growth_chunk_size)
      .withRequest(request)
      .withContext(context)
      .configure(core_);
}

StorageManager::StorageLease StorageManager::acquire(const Request &request) {
  core_.ensureConfigured();
  detail::StoragePayloadPoolTraits::validateRequestOrThrow(request);

  Context context{};

  auto payload_handle = core_.reserveUncreatedPayloadOrGrow();
  if (!payload_handle.isValid()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::OutOfRange,
        "Storage manager has no available slots");
  }

  if (!core_.emplacePayload(payload_handle, request, context)) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "Storage manager failed to create storage");
  }

  return core_.acquireStrongLease(payload_handle);
}

void StorageManager::shutdown() {
  Request request{};
  Context context{};
  core_.shutdown(request, context);
}

bool StorageManager::isConfigured() const noexcept {
  return core_.isConfigured();
}

} // namespace orteaf::internal::storage::manager
