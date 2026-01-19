#pragma once

/**
 * @file tensor_api.h
 * @brief Internal API for tensor management.
 *
 * TensorApi provides operations that automatically dispatch to the correct
 * manager based on the tensor impl type. Managers are not exposed.
 */

#include <span>
#include <string_view>

#include <orteaf/extension/tensor/registry/tensor_impl_types.h>
#include <orteaf/internal/storage/registry/storage_types.h>

namespace orteaf::internal::tensor::api {

/**
 * @brief Internal API for tensor management.
 *
 * Operations automatically dispatch to the correct manager based on impl type.
 * Managers are not exposed - use the operations directly.
 */
class TensorApi {
public:
  using StorageRegistry = ::orteaf::internal::storage::RegisteredStorages;
  using Registry = ::orteaf::internal::tensor::registry::RegisteredImpls;
  using LeaseVariant = typename Registry::LeaseVariant;
  using DenseTensorImpl = ::orteaf::extension::tensor::DenseTensorImpl;
  using DType = ::orteaf::internal::DType;
  using Execution = ::orteaf::internal::execution::Execution;
  using Dim = ::orteaf::extension::tensor::DenseTensorLayout::Dim;

  struct Config {
    StorageRegistry::Config storage_config{};
    typename Registry::Config registry_config{};
  };

  TensorApi() = delete;

  static void configure(const Config &config);
  static void shutdown();
  static bool isConfigured() noexcept;

  static StorageRegistry &storage();
  static Registry &registry();

  // ===== Creation (typed) =====

  template <typename Impl>
  static auto create(std::span<const Dim> shape, DType dtype,
                     Execution execution, std::size_t alignment = 0) {
    return registry().template get<Impl>().create(shape, dtype, execution,
                                                  alignment);
  }

  // ===== Creation (by name) =====

  /// @brief Create tensor impl by type name (e.g., "dense", "coo").
  /// @throws std::system_error if name is not registered.
  static LeaseVariant createByName(std::string_view impl_name,
                                   std::span<const Dim> shape, DType dtype,
                                   Execution execution,
                                   std::size_t alignment = 0);

  /// @brief Check if an impl type name is registered.
  static bool hasImplName(std::string_view impl_name);

  // ===== Auto-dispatch Operations =====

  static LeaseVariant transpose(const LeaseVariant &src,
                                std::span<const std::size_t> perm);

  static LeaseVariant slice(const LeaseVariant &src,
                            std::span<const Dim> starts,
                            std::span<const Dim> sizes);

  static LeaseVariant reshape(const LeaseVariant &src,
                              std::span<const Dim> new_shape);

  static LeaseVariant squeeze(const LeaseVariant &src);

  static LeaseVariant unsqueeze(const LeaseVariant &src, std::size_t dim);
};

} // namespace orteaf::internal::tensor::api
