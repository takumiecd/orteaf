#pragma once

/**
 * @file tensor_impl_registry.h
 * @brief Core registry template for tensor implementations.
 */

#include <cstring>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <variant>

#include <orteaf/internal/storage/registry/storage_types.h>
#include <orteaf/internal/tensor/manager/tensor_impl_manager.h>
#include <orteaf/internal/tensor/manager/tensor_impl_manager.inl>

namespace orteaf::internal::tensor::registry {

// =============================================================================
// TensorImplTraits
// =============================================================================

template <typename Impl> struct TensorImplTraits {
  using Manager = TensorImplManager<Impl>;
  using Lease = typename Manager::TensorImplLease;
  static constexpr const char *name = "unknown";
};

// =============================================================================
// TensorImplRegistry
// =============================================================================

template <typename... Impls> class TensorImplRegistry {
public:
  using StorageRegistry = ::orteaf::internal::storage::RegisteredStorages;
  using ManagerTuple = std::tuple<typename TensorImplTraits<Impls>::Manager...>;

  using LeaseVariant =
      std::variant<std::monostate, typename TensorImplTraits<Impls>::Lease...>;

  using ImplTypes = std::tuple<Impls...>;

  struct Config {
    std::tuple<typename TensorImplTraits<Impls>::Manager::Config...> configs{};

    template <typename Impl> auto &get() {
      return std::get<typename TensorImplTraits<Impl>::Manager::Config>(
          configs);
    }

    template <typename Impl> const auto &get() const {
      return std::get<typename TensorImplTraits<Impl>::Manager::Config>(
          configs);
    }
  };

  TensorImplRegistry() = default;
  TensorImplRegistry(const TensorImplRegistry &) = delete;
  TensorImplRegistry &operator=(const TensorImplRegistry &) = delete;
  TensorImplRegistry(TensorImplRegistry &&) = default;
  TensorImplRegistry &operator=(TensorImplRegistry &&) = default;

  void configure(const Config &config, StorageRegistry &storage_registry) {
    configureImpl<Impls...>(config, storage_registry);
  }

  void shutdown() { shutdownImpl<Impls...>(); }

  bool isConfigured() const noexcept { return isConfiguredImpl<Impls...>(); }

  template <typename Impl> auto &get() {
    return std::get<typename TensorImplTraits<Impl>::Manager>(managers_);
  }

  template <typename Impl> const auto &get() const {
    return std::get<typename TensorImplTraits<Impl>::Manager>(managers_);
  }

  /// @brief Dispatch an operation to the correct manager based on lease type.
  template <typename Lease, typename Func>
  static auto dispatch(const Lease &lease, Func &&func) {
    return dispatchImpl<Lease, Impls...>(lease, std::forward<Func>(func));
  }

  /// @brief Dispatch by name - calls func with the matching Impl type.
  /// @return true if name matched, false otherwise.
  template <typename Func>
  bool dispatchByName(std::string_view name, Func &&func) {
    return dispatchByNameImpl<Impls...>(name, std::forward<Func>(func));
  }

  /// @brief Check if a name is registered.
  static bool hasName(std::string_view name) {
    return hasNameImpl<Impls...>(name);
  }

private:
  template <typename First, typename... Rest>
  void configureImpl(const Config &config, StorageRegistry &storage_registry) {
    std::get<typename TensorImplTraits<First>::Manager>(managers_).configure(
        std::get<typename TensorImplTraits<First>::Manager::Config>(
            config.configs),
        storage_registry);
    if constexpr (sizeof...(Rest) > 0) {
      configureImpl<Rest...>(config, storage_registry);
    }
  }

  template <typename First, typename... Rest> void shutdownImpl() {
    std::get<typename TensorImplTraits<First>::Manager>(managers_).shutdown();
    if constexpr (sizeof...(Rest) > 0) {
      shutdownImpl<Rest...>();
    }
  }

  template <typename First, typename... Rest>
  bool isConfiguredImpl() const noexcept {
    bool first = std::get<typename TensorImplTraits<First>::Manager>(managers_)
                     .isConfigured();
    if constexpr (sizeof...(Rest) > 0) {
      return first && isConfiguredImpl<Rest...>();
    }
    return first;
  }

  // Dispatch by lease type
  template <typename Lease, typename First, typename... Rest, typename Func>
  static auto dispatchImpl(const Lease &lease, Func &&func) {
    if constexpr (std::is_same_v<Lease,
                                 typename TensorImplTraits<First>::Lease>) {
      return func.template operator()<First>(lease);
    } else if constexpr (sizeof...(Rest) > 0) {
      return dispatchImpl<Lease, Rest...>(lease, std::forward<Func>(func));
    }
  }

  // Dispatch by name
  template <typename First, typename... Rest, typename Func>
  bool dispatchByNameImpl(std::string_view name, Func &&func) {
    if (name == TensorImplTraits<First>::name) {
      func.template operator()<First>(
          std::get<typename TensorImplTraits<First>::Manager>(managers_));
      return true;
    }
    if constexpr (sizeof...(Rest) > 0) {
      return dispatchByNameImpl<Rest...>(name, std::forward<Func>(func));
    }
    return false;
  }

  // Check name
  template <typename First, typename... Rest>
  static bool hasNameImpl(std::string_view name) {
    if (name == TensorImplTraits<First>::name) {
      return true;
    }
    if constexpr (sizeof...(Rest) > 0) {
      return hasNameImpl<Rest...>(name);
    }
    return false;
  }

  ManagerTuple managers_{};
};

} // namespace orteaf::internal::tensor::registry
