#pragma once

/**
 * @file tensor_impl_manager.inl
 * @brief Implementation of generic TensorImplManager template.
 *
 * This file contains the template implementation and must be included
 * after tensor_impl_manager.h.
 */

#include <orteaf/internal/tensor/manager/tensor_impl_manager.h>

namespace orteaf::internal::tensor {

namespace detail {

// =============================================================================
// Pool Traits Implementation
// =============================================================================

template <typename Impl>
void TensorImplPoolTraits<Impl>::validateRequestOrThrow(
    const Request &request) {
  std::visit(
      [](const auto &req) {
        using T = std::decay_t<decltype(req)>;
        if constexpr (std::is_same_v<T, TensorImplViewRequest<Impl>>) {
          if (!req.storage) {
            ::orteaf::internal::diagnostics::error::throwError(
                ::orteaf::internal::diagnostics::error::OrteafErrc::
                    InvalidArgument,
                "TensorImplViewRequest requires valid storage");
          }
        } else {
          ::orteaf::internal::tensor::registry::TensorImplTraits<
              Impl>::validateCreateRequest(req);
        }
      },
      request);
}

template <typename Impl>
bool TensorImplPoolTraits<Impl>::create(Payload &payload,
                                        const Request &request,
                                        const Context &context) {
  return std::visit(
      [&](const auto &req) -> bool {
        using T = std::decay_t<decltype(req)>;
        if constexpr (std::is_same_v<T, TensorImplViewRequest<Impl>>) {
          payload = Impl(req.layout, req.storage);
          return true;
        } else {
          return ::orteaf::internal::tensor::registry::TensorImplTraits<
              Impl>::createPayload(payload, req, context);
        }
      },
      request);
}

template <typename Impl>
void TensorImplPoolTraits<Impl>::destroy(Payload &payload, const Request &,
                                         const Context &context) {
  ::orteaf::internal::tensor::registry::TensorImplTraits<Impl>::destroyPayload(
      payload, context);
}

} // namespace detail

// =============================================================================
// TensorImplManager Implementation
// =============================================================================

template <typename Impl>
  requires TensorImplConcept<Impl>
void TensorImplManager<Impl>::configure(const Config &config,
                                        StorageRegistry &storage_registry) {
  storage_registry_ = &storage_registry;

  detail::TensorImplRequest<Impl> request{};
  detail::TensorImplContext context{storage_registry_};

  typename Core::template Builder<detail::TensorImplRequest<Impl>,
                                  detail::TensorImplContext>
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

template <typename Impl>
  requires TensorImplConcept<Impl>
void TensorImplManager<Impl>::shutdown() {
  detail::TensorImplRequest<Impl> request{};
  detail::TensorImplContext context{storage_registry_};
  core_.shutdown(request, context);
  storage_registry_ = nullptr;
}

template <typename Impl>
  requires TensorImplConcept<Impl>
bool TensorImplManager<Impl>::isConfigured() const noexcept {
  return core_.isConfigured();
}

template <typename Impl>
  requires TensorImplConcept<Impl>
typename TensorImplManager<Impl>::TensorImplLease
TensorImplManager<Impl>::create(const CreateRequest &create_request) {
  core_.ensureConfigured();

  detail::TensorImplRequest<Impl> request{create_request};
  detail::TensorImplContext context{storage_registry_};

  auto payload_handle = core_.reserveUncreatedPayloadOrGrow();
  if (!payload_handle.isValid()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::OutOfRange,
        "TensorImplManager has no available slots");
  }

  if (!core_.emplacePayload(payload_handle, request, context)) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "TensorImplManager failed to create tensor impl");
  }

  return core_.acquireStrongLease(payload_handle);
}

template <typename Impl>
  requires TensorImplConcept<Impl>
typename TensorImplManager<Impl>::TensorImplLease
TensorImplManager<Impl>::createView(Layout layout, StorageLease storage) {
  core_.ensureConfigured();

  detail::TensorImplViewRequest<Impl> req{};
  req.layout = std::move(layout);
  req.storage = std::move(storage);

  detail::TensorImplRequest<Impl> request{req};
  detail::TensorImplContext context{storage_registry_};

  auto payload_handle = core_.reserveUncreatedPayloadOrGrow();
  if (!payload_handle.isValid()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::OutOfRange,
        "TensorImplManager has no available slots");
  }

  if (!core_.emplacePayload(payload_handle, request, context)) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "TensorImplManager failed to create view");
  }

  return core_.acquireStrongLease(payload_handle);
}

// ===== View Operations =====

template <typename Impl>
  requires TensorImplConcept<Impl>
typename TensorImplManager<Impl>::TensorImplLease
TensorImplManager<Impl>::transpose(const TensorImplLease &src,
                                   std::span<const std::size_t> perm)
  requires HasTranspose<Impl>
{
  auto new_layout = src->layout().transpose(perm);
  return createView(std::move(new_layout), src->storageLease());
}

template <typename Impl>
  requires TensorImplConcept<Impl>
typename TensorImplManager<Impl>::TensorImplLease
TensorImplManager<Impl>::slice(const TensorImplLease &src,
                               std::span<const Dim> starts,
                               std::span<const Dim> sizes)
  requires HasSlice<Impl>
{
  auto new_layout = src->layout().slice(starts, sizes);
  return createView(std::move(new_layout), src->storageLease());
}

template <typename Impl>
  requires TensorImplConcept<Impl>
typename TensorImplManager<Impl>::TensorImplLease
TensorImplManager<Impl>::reshape(const TensorImplLease &src,
                                 std::span<const Dim> new_shape)
  requires HasReshape<Impl>
{
  auto new_layout = src->layout().reshape(new_shape);
  return createView(std::move(new_layout), src->storageLease());
}

template <typename Impl>
  requires TensorImplConcept<Impl>
typename TensorImplManager<Impl>::TensorImplLease
TensorImplManager<Impl>::squeeze(const TensorImplLease &src)
  requires HasSqueeze<Impl>
{
  auto new_layout = src->layout().squeeze();
  return createView(std::move(new_layout), src->storageLease());
}

template <typename Impl>
  requires TensorImplConcept<Impl>
typename TensorImplManager<Impl>::TensorImplLease
TensorImplManager<Impl>::unsqueeze(const TensorImplLease &src, std::size_t dim)
  requires HasUnsqueeze<Impl>
{
  auto new_layout = src->layout().unsqueeze(dim);
  return createView(std::move(new_layout), src->storageLease());
}

} // namespace orteaf::internal::tensor
