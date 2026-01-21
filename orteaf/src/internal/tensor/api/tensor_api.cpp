#include "orteaf/internal/tensor/api/tensor_api.h"

#include "orteaf/internal/diagnostics/error/error.h"

namespace orteaf::internal::tensor::api {

namespace {

bool g_configured = false;

TensorApi::StorageRegistry &storageRegistrySingleton() {
  static TensorApi::StorageRegistry instance;
  return instance;
}

TensorApi::Registry &registrySingleton() {
  static TensorApi::Registry instance;
  return instance;
}

void ensureConfigured() {
  if (!g_configured) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "TensorApi is not configured");
  }
}

void throwInvalidState(const char *op) {
  ::orteaf::internal::diagnostics::error::throwError(
      ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState, op);
}

} // namespace

void TensorApi::configure(const Config &config) {
  if (g_configured) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "TensorApi is already configured");
  }

  storageRegistrySingleton().configure(config.storage_config);
  registrySingleton().configure(config.registry_config,
                                storageRegistrySingleton());
  g_configured = true;
}

void TensorApi::shutdown() {
  if (!g_configured) {
    return;
  }
  registrySingleton().shutdown();
  storageRegistrySingleton().shutdown();
  g_configured = false;
}

bool TensorApi::isConfigured() noexcept { return g_configured; }

TensorApi::StorageRegistry &TensorApi::storage() {
  ensureConfigured();
  return storageRegistrySingleton();
}

TensorApi::Registry &TensorApi::registry() {
  ensureConfigured();
  return registrySingleton();
}

// ===== Creation by Name =====

TensorApi::LeaseVariant TensorApi::createByName(std::string_view impl_name,
                                                std::span<const Dim> shape,
                                                DType dtype,
                                                Execution execution,
                                                std::size_t alignment) {
  ensureConfigured();

  LeaseVariant result;
  bool found = registrySingleton().dispatchByName(
      impl_name, [&]<typename Impl>(auto &manager) {
        result = manager.create(shape, dtype, execution, alignment);
      });

  if (!found) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "Unknown tensor impl type name");
  }

  return result;
}

bool TensorApi::hasImplName(std::string_view impl_name) {
  return Registry::hasName(impl_name);
}

// ===== Auto-dispatch Operations =====

TensorApi::LeaseVariant
TensorApi::transpose(const LeaseVariant &src,
                     std::span<const std::size_t> perm) {
  return std::visit(
      [&](const auto &lease) -> LeaseVariant {
        using LeaseType = std::decay_t<decltype(lease)>;
        if constexpr (std::is_same_v<LeaseType, std::monostate>) {
          throwInvalidState("Cannot transpose invalid tensor");
          return std::monostate{};
        } else {
          return Registry::dispatch(lease, [&]<typename Impl>(const auto &l) {
            return registrySingleton().template get<Impl>().transpose(l, perm);
          });
        }
      },
      src);
}

TensorApi::LeaseVariant TensorApi::slice(const LeaseVariant &src,
                                         std::span<const Dim> starts,
                                         std::span<const Dim> sizes) {
  return std::visit(
      [&](const auto &lease) -> LeaseVariant {
        using LeaseType = std::decay_t<decltype(lease)>;
        if constexpr (std::is_same_v<LeaseType, std::monostate>) {
          throwInvalidState("Cannot slice invalid tensor");
          return std::monostate{};
        } else {
          return Registry::dispatch(lease, [&]<typename Impl>(const auto &l) {
            return registrySingleton().template get<Impl>().slice(l, starts,
                                                                  sizes);
          });
        }
      },
      src);
}

TensorApi::LeaseVariant TensorApi::reshape(const LeaseVariant &src,
                                           std::span<const Dim> new_shape) {
  return std::visit(
      [&](const auto &lease) -> LeaseVariant {
        using LeaseType = std::decay_t<decltype(lease)>;
        if constexpr (std::is_same_v<LeaseType, std::monostate>) {
          throwInvalidState("Cannot reshape invalid tensor");
          return std::monostate{};
        } else {
          return Registry::dispatch(lease, [&]<typename Impl>(const auto &l) {
            return registrySingleton().template get<Impl>().reshape(l,
                                                                    new_shape);
          });
        }
      },
      src);
}

TensorApi::LeaseVariant TensorApi::squeeze(const LeaseVariant &src) {
  return std::visit(
      [&](const auto &lease) -> LeaseVariant {
        using LeaseType = std::decay_t<decltype(lease)>;
        if constexpr (std::is_same_v<LeaseType, std::monostate>) {
          throwInvalidState("Cannot squeeze invalid tensor");
          return std::monostate{};
        } else {
          return Registry::dispatch(lease, [&]<typename Impl>(const auto &l) {
            return registrySingleton().template get<Impl>().squeeze(l);
          });
        }
      },
      src);
}

TensorApi::LeaseVariant TensorApi::unsqueeze(const LeaseVariant &src,
                                             std::size_t dim) {
  return std::visit(
      [&](const auto &lease) -> LeaseVariant {
        using LeaseType = std::decay_t<decltype(lease)>;
        if constexpr (std::is_same_v<LeaseType, std::monostate>) {
          throwInvalidState("Cannot unsqueeze invalid tensor");
          return std::monostate{};
        } else {
          return Registry::dispatch(lease, [&]<typename Impl>(const auto &l) {
            return registrySingleton().template get<Impl>().unsqueeze(l, dim);
          });
        }
      },
      src);
}

} // namespace orteaf::internal::tensor::api
