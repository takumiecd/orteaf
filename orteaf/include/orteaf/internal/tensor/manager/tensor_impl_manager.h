#pragma once

/**
 * @file tensor_impl_manager.h
 * @brief Generic template for TensorImpl management.
 *
 * This template provides automatic pool management for any TensorImpl type.
 * Contributors only need to define their TensorImpl and this manager will
 * be auto-generated with appropriate view operations based on concepts.
 */

#include <cstddef>
#include <span>
#include <utility>
#include <variant>

#include <orteaf/internal/base/handle.h>
#include <orteaf/internal/base/lease/control_block/strong.h>
#include <orteaf/internal/base/manager/pool_manager.h>
#include <orteaf/internal/base/pool/slot_pool.h>
#include <orteaf/internal/diagnostics/error/error.h>
#include <orteaf/internal/dtype/dtype.h>
#include <orteaf/internal/execution/execution.h>
#include <orteaf/internal/storage/registry/storage_types.h>
#include <orteaf/internal/storage/storage_lease.h>
#include <orteaf/internal/tensor/concepts/tensor_impl_concepts.h>
#include <orteaf/internal/tensor/traits/tensor_impl_traits.h>

namespace orteaf::internal::tensor {

// =============================================================================
// Handle for TensorImpl
// =============================================================================

template <typename Impl> struct TensorImplTag {};

template <typename Impl>
using TensorImplHandle =
    ::orteaf::internal::base::Handle<TensorImplTag<Impl>, uint32_t, uint32_t>;

// =============================================================================
// Pool Traits for TensorImpl
// =============================================================================

namespace detail {

/// @brief Request for creating a view (shares storage)
template <typename Impl> struct TensorImplViewRequest {
  using Layout = typename Impl::Layout;
  using StorageLease = ::orteaf::internal::storage::StorageLease;

  Layout layout{};
  StorageLease storage{};
};

/// @brief Combined request type
template <typename Impl>
using TensorImplRequest = std::variant<
    typename ::orteaf::internal::tensor::registry::TensorImplTraits<
        Impl>::CreateRequest,
    TensorImplViewRequest<Impl>>;

/// @brief Context for pool operations
struct TensorImplContext {
  ::orteaf::internal::storage::RegisteredStorages *storage_registry{nullptr};
};

/// @brief Pool traits for generic TensorImpl
template <typename Impl> struct TensorImplPoolTraits {
  using CreateRequest = typename ::orteaf::internal::tensor::registry::
      TensorImplTraits<Impl>::CreateRequest;
  using Payload = Impl;
  using Handle = TensorImplHandle<Impl>;
  using Request = TensorImplRequest<Impl>;
  using Context = TensorImplContext;

  static constexpr bool destroy_on_release = true;
  static constexpr const char *ManagerName = "TensorImpl manager";

  static void validateRequestOrThrow(const Request &request);
  static bool create(Payload &payload, const Request &request,
                     const Context &context);
  static void destroy(Payload &payload, const Request &request,
                      const Context &context);
};

} // namespace detail

// =============================================================================
// Generic TensorImplManager
// =============================================================================

/**
 * @brief Generic manager for TensorImpl types.
 *
 * Provides automatic pool management for any TensorImpl type.
 * View operations are conditionally enabled based on concepts.
 *
 * @tparam Impl The TensorImpl type (must satisfy TensorImplConcept)
 */
template <typename Impl>
  requires TensorImplConcept<Impl>
class TensorImplManager {
public:
  using PayloadPool = ::orteaf::internal::base::pool::SlotPool<
      detail::TensorImplPoolTraits<Impl>>;
  using ControlBlock =
      ::orteaf::internal::base::StrongControlBlock<TensorImplHandle<Impl>, Impl,
                                                   PayloadPool>;

  struct Traits {
    using PayloadPool = TensorImplManager::PayloadPool;
    using ControlBlock = TensorImplManager::ControlBlock;
    struct ControlBlockTag {};
    using PayloadHandle = TensorImplHandle<Impl>;
    static constexpr const char *Name =
        detail::TensorImplPoolTraits<Impl>::ManagerName;
  };

  using Core = ::orteaf::internal::base::PoolManager<Traits>;
  using TensorImplLease = typename Core::StrongLeaseType;
  using Layout = typename Impl::Layout;
  using Dims = typename Layout::Dims;
  using Dim = typename Layout::Dim;
  using CreateRequest = typename ::orteaf::internal::tensor::registry::
      TensorImplTraits<Impl>::CreateRequest;
  using StorageRegistry = ::orteaf::internal::storage::RegisteredStorages;
  using StorageLease = ::orteaf::internal::storage::StorageLease;

  struct Config {
    std::size_t control_block_capacity{64};
    std::size_t control_block_block_size{16};
    std::size_t control_block_growth_chunk_size{1};
    std::size_t payload_capacity{64};
    std::size_t payload_block_size{16};
    std::size_t payload_growth_chunk_size{1};
  };

  TensorImplManager() = default;
  TensorImplManager(const TensorImplManager &) = delete;
  TensorImplManager &operator=(const TensorImplManager &) = delete;
  TensorImplManager(TensorImplManager &&) = default;
  TensorImplManager &operator=(TensorImplManager &&) = default;
  ~TensorImplManager() = default;

  void configure(const Config &config, StorageRegistry &storage_registry);
  void shutdown();
  bool isConfigured() const noexcept;

  // ===== Creation =====

  TensorImplLease create(const CreateRequest &request);

  // ===== View Operations (conditionally enabled) =====

  TensorImplLease transpose(const TensorImplLease &src,
                            std::span<const std::size_t> perm)
    requires HasTranspose<Impl>;

  TensorImplLease slice(const TensorImplLease &src, std::span<const Dim> starts,
                        std::span<const Dim> sizes)
    requires HasSlice<Impl>;

  TensorImplLease reshape(const TensorImplLease &src,
                          std::span<const Dim> new_shape)
    requires HasReshape<Impl>;

  TensorImplLease squeeze(const TensorImplLease &src)
    requires HasSqueeze<Impl>;

  TensorImplLease unsqueeze(const TensorImplLease &src, std::size_t dim)
    requires HasUnsqueeze<Impl>;

private:
  TensorImplLease createView(Layout layout, StorageLease storage);

  Core core_{};
  StorageRegistry *storage_registry_{nullptr};
};

} // namespace orteaf::internal::tensor
