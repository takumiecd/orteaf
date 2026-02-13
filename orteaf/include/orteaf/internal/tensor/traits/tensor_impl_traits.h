#pragma once

#include <type_traits>

namespace orteaf::internal::tensor::registry {

namespace detail {

template <typename> struct DependentFalse : std::false_type {};

} // namespace detail

template <typename Impl> struct TensorImplTraits {
  using CreateRequest = typename Impl::CreateRequest;

  static constexpr const char *name = "unknown";

  static void validateCreateRequest(const CreateRequest &) {
    static_assert(detail::DependentFalse<Impl>::value,
                  "TensorImplTraits specialization is required");
  }

  template <typename Context>
  static bool createPayload(Impl &, const CreateRequest &, const Context &) {
    static_assert(detail::DependentFalse<Impl>::value,
                  "TensorImplTraits specialization is required");
    return false;
  }

  template <typename Context>
  static void destroyPayload(Impl &, const Context &) {
    static_assert(detail::DependentFalse<Impl>::value,
                  "TensorImplTraits specialization is required");
  }
};

} // namespace orteaf::internal::tensor::registry
