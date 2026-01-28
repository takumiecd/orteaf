#pragma once

#include <type_traits>

namespace orteaf::internal::kernel {

template <typename From, typename To>
struct TransformImpl {
  static To apply(const From &value) { return To{value}; }
};

template <typename T> struct TransformImpl<T, T> {
  static T apply(const T &value) { return value; }
};

template <typename From, typename To>
To Transform(const From &value) {
  return TransformImpl<From, To>::apply(value);
}

} // namespace orteaf::internal::kernel
