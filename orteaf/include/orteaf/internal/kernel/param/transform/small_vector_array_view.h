#pragma once

#include <cstddef>

#include <orteaf/internal/base/array_view.h>
#include <orteaf/internal/base/small_vector.h>
#include <orteaf/internal/kernel/param/param_transform.h>

namespace orteaf::internal::kernel {

template <typename T, std::size_t InlineCapacity>
struct TransformImpl<::orteaf::internal::base::SmallVector<T, InlineCapacity>,
                     ::orteaf::internal::base::ArrayView<const T>> {
  static ::orteaf::internal::base::ArrayView<const T>
  apply(const ::orteaf::internal::base::SmallVector<T, InlineCapacity> &value) {
    return ::orteaf::internal::base::ArrayView<const T>(value.data(), value.size());
  }
};

} // namespace orteaf::internal::kernel
