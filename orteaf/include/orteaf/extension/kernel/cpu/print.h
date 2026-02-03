#pragma once

#include <cstdint>
#include <ostream>
#include <span>

#include <orteaf/internal/base/array_view.h>
#include <orteaf/internal/base/inline_vector.h>
#include <orteaf/internal/dtype/dtype.h>

namespace orteaf::extension::kernel::cpu {

void printTensor(std::span<const std::int64_t> shape, const void *buffer,
                 ::orteaf::internal::DType dtype, std::ostream &os);

inline void printTensor(
    ::orteaf::internal::base::ArrayView<const std::int64_t> shape,
    const void *buffer, ::orteaf::internal::DType dtype, std::ostream &os) {
  printTensor(std::span<const std::int64_t>(shape.data, shape.count), buffer,
              dtype, os);
}

template <std::uint8_t N>
inline void printTensor(
    const ::orteaf::internal::base::InlineVector<std::int64_t, N> &shape,
    const void *buffer, ::orteaf::internal::DType dtype, std::ostream &os) {
  printTensor(std::span<const std::int64_t>(shape.data, shape.size), buffer,
              dtype, os);
}

}  // namespace orteaf::extension::kernel::cpu
