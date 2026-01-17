#pragma once

/**
 * @file tensor_impl_registry.h
 * @brief Core registry template for tensor implementations.
 */

#include <tuple>
#include <variant>

#include <orteaf/internal/storage/manager/storage_manager.h>
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
  using StorageManager = ::orteaf::internal::storage::manager::StorageManager;
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

  void configure(const Config &config, StorageManager &storage_manager) {
    configureImpl<Impls...>(config, storage_manager);
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

private:
  template <typename First, typename... Rest>
  void configureImpl(const Config &config, StorageManager &storage_manager) {
    std::get<typename TensorImplTraits<First>::Manager>(managers_).configure(
        std::get<typename TensorImplTraits<First>::Manager::Config>(
            config.configs),
        storage_manager);
    if constexpr (sizeof...(Rest) > 0) {
      configureImpl<Rest...>(config, storage_manager);
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

  // Dispatch helper - find which Impl matches the Lease type
  template <typename Lease, typename First, typename... Rest, typename Func>
  static auto dispatchImpl(const Lease &lease, Func &&func) {
    if constexpr (std::is_same_v<Lease,
                                 typename TensorImplTraits<First>::Lease>) {
      return func.template operator()<First>(lease);
    } else if constexpr (sizeof...(Rest) > 0) {
      return dispatchImpl<Lease, Rest...>(lease, std::forward<Func>(func));
    }
  }

  ManagerTuple managers_{};
};

} // namespace orteaf::internal::tensor::registry
