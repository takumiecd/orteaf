#pragma once

#include <cstddef>
#include <cstdint>

#include <orteaf/internal/base/array_view.h>
#include <orteaf/internal/base/inline_vector.h>
#include <orteaf/internal/kernel/param/param_transform.h>

namespace orteaf::internal::kernel {

template <typename T, std::uint8_t N>
struct TransformImpl<::orteaf::internal::base::ArrayView<const T>,
                     ::orteaf::internal::base::InlineVector<T, N>> {
  static ::orteaf::internal::base::InlineVector<T, N>
  apply(const ::orteaf::internal::base::ArrayView<const T> &value) {
    ::orteaf::internal::base::InlineVector<T, N> result{};
    const std::size_t count = value.size();
    const std::size_t limit =
        count < static_cast<std::size_t>(N) ? count
                                            : static_cast<std::size_t>(N);
    result.size = static_cast<std::uint8_t>(limit);
    for (std::size_t i = 0; i < limit; ++i) {
      result.data[i] = value.data[i];
    }
    return result;
  }
};

} // namespace orteaf::internal::kernel
