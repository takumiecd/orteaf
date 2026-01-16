#pragma once

/**
 * @file tensor_api.h
 * @brief Internal API for tensor management.
 *
 * TensorApi provides centralized access to StorageManager and
 * tensor impl managers, enabling tensor creation and manipulation.
 * This is internal infrastructure - users should use the Tensor class.
 */

#include <span>

#include <orteaf/extension/tensor/manager/dense_tensor_impl_manager.h>
#include <orteaf/internal/storage/manager/storage_manager.h>

namespace orteaf::internal::tensor::api {

/**
 * @brief Internal API for tensor management.
 *
 * Holds StorageManager and various tensor impl managers.
 * Must be configured before use and shutdown when done.
 *
 * @note This is internal infrastructure. Users should use Tensor class.
 */
class TensorApi {
public:
  using StorageManager = ::orteaf::internal::storage::manager::StorageManager;
  using DenseTensorImplManager =
      ::orteaf::extension::tensor::DenseTensorImplManager;
  using TensorImplLease = DenseTensorImplManager::TensorImplLease;
  using DType = ::orteaf::internal::DType;
  using Execution = ::orteaf::internal::execution::Execution;
  using Dim = ::orteaf::extension::tensor::DenseTensorLayout::Dim;

  struct Config {
    StorageManager::Config storage_config{};
    DenseTensorImplManager::Config dense_config{};
  };

  TensorApi() = delete;

  /// @brief Configure the API with all managers.
  static void configure(const Config &config);

  /// @brief Shutdown all managers.
  static void shutdown();

  /// @brief Check if configured.
  static bool isConfigured() noexcept;

  /// @brief Access the storage manager.
  static StorageManager &storage();

  /// @brief Access the dense tensor impl manager.
  static DenseTensorImplManager &dense();

  // ===== Convenience methods =====

  /// @brief Create a new dense tensor impl.
  static TensorImplLease create(std::span<const Dim> shape, DType dtype,
                                Execution execution, std::size_t alignment = 0);

  /// @brief Create a transposed view.
  static TensorImplLease transpose(const TensorImplLease &src,
                                   std::span<const std::size_t> perm);

  /// @brief Create a sliced view.
  static TensorImplLease slice(const TensorImplLease &src,
                               std::span<const Dim> starts,
                               std::span<const Dim> sizes);

  /// @brief Create a reshaped view.
  static TensorImplLease reshape(const TensorImplLease &src,
                                 std::span<const Dim> new_shape);

  /// @brief Create a squeezed view.
  static TensorImplLease squeeze(const TensorImplLease &src);

  /// @brief Create an unsqueezed view.
  static TensorImplLease unsqueeze(const TensorImplLease &src, std::size_t dim);

  // Future: Add more impl managers
  // static CooTensorImplManager& coo();
  // static CsrTensorImplManager& csr();
};

} // namespace orteaf::internal::tensor::api
