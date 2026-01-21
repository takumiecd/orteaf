#pragma once

/**
 * @file storage_registry.h
 * @brief Core registry template for storage implementations.
 */

#include <tuple>
#include <variant>

#include <orteaf/internal/storage/manager/typed_storage_manager.h>
#include <orteaf/internal/storage/manager/typed_storage_manager.inl>

namespace orteaf::internal::storage::registry {

// =============================================================================
// StorageTraits
// =============================================================================

template <typename Storage> struct StorageTraits {
  using Manager = manager::TypedStorageManager<Storage>;
  using Lease = typename Manager::StorageLease;
  static constexpr const char *name = "unknown";
};

// =============================================================================
// StorageRegistry
// =============================================================================

template <typename... Storages> class StorageRegistry {
public:
  using ManagerTuple = std::tuple<typename StorageTraits<Storages>::Manager...>;

  using LeaseVariant =
      std::variant<std::monostate, typename StorageTraits<Storages>::Lease...>;

  using StorageTypes = std::tuple<Storages...>;

  struct Config {
    std::tuple<typename StorageTraits<Storages>::Manager::Config...> configs{};

    template <typename Storage> auto &get() {
      return std::get<typename StorageTraits<Storage>::Manager::Config>(
          configs);
    }

    template <typename Storage> const auto &get() const {
      return std::get<typename StorageTraits<Storage>::Manager::Config>(
          configs);
    }
  };

  StorageRegistry() = default;
  StorageRegistry(const StorageRegistry &) = delete;
  StorageRegistry &operator=(const StorageRegistry &) = delete;
  StorageRegistry(StorageRegistry &&) = default;
  StorageRegistry &operator=(StorageRegistry &&) = default;

  void configure(const Config &config) { configureImpl<Storages...>(config); }

  void shutdown() { shutdownImpl<Storages...>(); }

  bool isConfigured() const noexcept { return isConfiguredImpl<Storages...>(); }

  template <typename Storage> auto &get() {
    return std::get<typename StorageTraits<Storage>::Manager>(managers_);
  }

  template <typename Storage> const auto &get() const {
    return std::get<typename StorageTraits<Storage>::Manager>(managers_);
  }

private:
  template <typename First, typename... Rest>
  void configureImpl(const Config &config) {
    std::get<typename StorageTraits<First>::Manager>(managers_).configure(
        std::get<typename StorageTraits<First>::Manager::Config>(
            config.configs));
    if constexpr (sizeof...(Rest) > 0) {
      configureImpl<Rest...>(config);
    }
  }

  template <typename First, typename... Rest> void shutdownImpl() {
    std::get<typename StorageTraits<First>::Manager>(managers_).shutdown();
    if constexpr (sizeof...(Rest) > 0) {
      shutdownImpl<Rest...>();
    }
  }

  template <typename First, typename... Rest>
  bool isConfiguredImpl() const noexcept {
    bool first = std::get<typename StorageTraits<First>::Manager>(managers_)
                     .isConfigured();
    if constexpr (sizeof...(Rest) > 0) {
      return first && isConfiguredImpl<Rest...>();
    }
    return first;
  }

  ManagerTuple managers_{};
};

} // namespace orteaf::internal::storage::registry
