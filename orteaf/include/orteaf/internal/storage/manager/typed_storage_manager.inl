#pragma once

/**
 * @file typed_storage_manager.inl
 * @brief Implementation of TypedStorageManager template methods.
 */

#include <orteaf/internal/storage/manager/typed_storage_manager.h>

namespace orteaf::internal::storage::manager {

template <typename Storage>
  requires concepts::StorageConcept<Storage>
void TypedStorageManager<Storage>::configure(const Config &config) {
  detail::TypedStorageRequest<Storage> request{};
  detail::TypedStorageContext<Storage> context{};

  typename Core::template Builder<detail::TypedStorageRequest<Storage>,
                                  detail::TypedStorageContext<Storage>>
      builder{};
  builder.withControlBlockCapacity(config.control_block_capacity)
      .withControlBlockBlockSize(config.control_block_block_size)
      .withControlBlockGrowthChunkSize(config.control_block_growth_chunk_size)
      .withPayloadCapacity(config.payload_capacity)
      .withPayloadBlockSize(config.payload_block_size)
      .withPayloadGrowthChunkSize(config.payload_growth_chunk_size)
      .withRequest(request)
      .withContext(context)
      .configure(core_);
}

template <typename Storage>
  requires concepts::StorageConcept<Storage>
typename TypedStorageManager<Storage>::StorageLease
TypedStorageManager<Storage>::acquire(const Request &request) {
  detail::TypedStoragePoolTraits<Storage>::validateRequestOrThrow(request);
  core_.ensureConfigured();

  detail::TypedStorageContext<Storage> context{};
  auto payload_handle = core_.reserveUncreatedPayloadOrGrow();
  if (!payload_handle.isValid()) {
    ORTEAF_THROW(OutOfRange, "TypedStorageManager has no available slots");
  }

  if (!core_.emplacePayload(payload_handle, request, context)) {
    ORTEAF_THROW(InvalidState, "TypedStorageManager failed to create storage");
  }

  return core_.acquireStrongLease(payload_handle);
}

template <typename Storage>
  requires concepts::StorageConcept<Storage>
void TypedStorageManager<Storage>::shutdown() {
  detail::TypedStorageRequest<Storage> request{};
  detail::TypedStorageContext<Storage> context{};
  core_.shutdown(request, context);
}

template <typename Storage>
  requires concepts::StorageConcept<Storage>
bool TypedStorageManager<Storage>::isConfigured() const noexcept {
  return core_.isConfigured();
}

} // namespace orteaf::internal::storage::manager
