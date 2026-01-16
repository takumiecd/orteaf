#include "orteaf/internal/tensor/api/tensor_api.h"

#include "orteaf/internal/diagnostics/error/error.h"

namespace orteaf::internal::tensor::api {

namespace {

bool g_configured = false;

TensorApi::StorageManager &storageManagerSingleton() {
  static TensorApi::StorageManager instance;
  return instance;
}

TensorApi::DenseTensorImplManager &denseManagerSingleton() {
  static TensorApi::DenseTensorImplManager instance;
  return instance;
}

} // namespace

void TensorApi::configure(const Config &config) {
  if (g_configured) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "TensorApi is already configured");
  }

  storageManagerSingleton().configure(config.storage_config);
  denseManagerSingleton().configure(config.dense_config,
                                    storageManagerSingleton());
  g_configured = true;
}

void TensorApi::shutdown() {
  if (!g_configured) {
    return;
  }
  denseManagerSingleton().shutdown();
  storageManagerSingleton().shutdown();
  g_configured = false;
}

bool TensorApi::isConfigured() noexcept { return g_configured; }

TensorApi::StorageManager &TensorApi::storage() {
  if (!g_configured) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "TensorApi is not configured");
  }
  return storageManagerSingleton();
}

TensorApi::DenseTensorImplManager &TensorApi::dense() {
  if (!g_configured) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "TensorApi is not configured");
  }
  return denseManagerSingleton();
}

// ===== Convenience methods =====

TensorApi::TensorImplLease TensorApi::create(std::span<const Dim> shape,
                                             DType dtype, Execution execution,
                                             std::size_t alignment) {
  return dense().create(shape, dtype, execution, alignment);
}

TensorApi::TensorImplLease
TensorApi::transpose(const TensorImplLease &src,
                     std::span<const std::size_t> perm) {
  return dense().transpose(src, perm);
}

TensorApi::TensorImplLease TensorApi::slice(const TensorImplLease &src,
                                            std::span<const Dim> starts,
                                            std::span<const Dim> sizes) {
  return dense().slice(src, starts, sizes);
}

TensorApi::TensorImplLease TensorApi::reshape(const TensorImplLease &src,
                                              std::span<const Dim> new_shape) {
  return dense().reshape(src, new_shape);
}

TensorApi::TensorImplLease TensorApi::squeeze(const TensorImplLease &src) {
  return dense().squeeze(src);
}

TensorApi::TensorImplLease TensorApi::unsqueeze(const TensorImplLease &src,
                                                std::size_t dim) {
  return dense().unsqueeze(src, dim);
}

} // namespace orteaf::internal::tensor::api
