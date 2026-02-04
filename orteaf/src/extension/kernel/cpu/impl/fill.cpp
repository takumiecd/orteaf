#include <cstddef>
#include <cstdint>
#include <span>

#include <orteaf/internal/diagnostics/error/error.h>
#include <orteaf/internal/dtype/dtype.h>
#include <orteaf/internal/dtype/float16.h>
#include <orteaf/internal/dtype/float8.h>

namespace orteaf::extension::kernel::cpu {

namespace {

namespace error = ::orteaf::internal::diagnostics::error;

template <typename T>
T castFillValue(double value) {
  return static_cast<T>(value);
}

template <>
bool castFillValue<bool>(double value) {
  return value != 0.0;
}

template <typename T>
void fillTyped(void *data, std::size_t count, double value) {
  if (count == 0) {
    return;
  }
  if (data == nullptr) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "Fill kernel received null output buffer");
  }
  const T casted = castFillValue<T>(value);
  auto *typed = static_cast<T *>(data);
  for (std::size_t i = 0; i < count; ++i) {
    typed[i] = casted;
  }
}

template <typename T>
void fillStridedRecursive(T *base, std::span<const std::int64_t> shape,
                          std::span<const std::int64_t> strides,
                          std::size_t dim, std::int64_t offset, T value) {
  if (dim == shape.size()) {
    base[offset] = value;
    return;
  }

  const auto extent = shape[dim];
  if (extent == 0) {
    return;
  }

  const auto stride = strides[dim];
  for (std::int64_t i = 0; i < extent; ++i) {
    fillStridedRecursive(base, shape, strides, dim + 1,
                         offset + i * stride, value);
  }
}

template <typename T>
void fillStridedTyped(void *data, std::span<const std::int64_t> shape,
                      std::span<const std::int64_t> strides,
                      std::int64_t offset, double value) {
  if (shape.empty()) {
    if (data == nullptr) {
      error::throwError(error::OrteafErrc::InvalidParameter,
                        "Fill kernel received null output buffer");
    }
    const T casted = castFillValue<T>(value);
    auto *typed = static_cast<T *>(data);
    typed[offset] = casted;
    return;
  }

  if (data == nullptr) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "Fill kernel received null output buffer");
  }

  const T casted = castFillValue<T>(value);
  auto *typed = static_cast<T *>(data);
  fillStridedRecursive(typed, shape, strides, 0, offset, casted);
}

}  // namespace

void fillTensor(void *data, std::size_t count, ::orteaf::internal::DType dtype,
                double value) {
  if (count == 0) {
    return;
  }
  switch (dtype) {
  case ::orteaf::internal::DType::Bool:
    fillTyped<bool>(data, count, value);
    break;
  case ::orteaf::internal::DType::I8:
    fillTyped<std::int8_t>(data, count, value);
    break;
  case ::orteaf::internal::DType::I16:
    fillTyped<std::int16_t>(data, count, value);
    break;
  case ::orteaf::internal::DType::I32:
    fillTyped<std::int32_t>(data, count, value);
    break;
  case ::orteaf::internal::DType::I64:
    fillTyped<std::int64_t>(data, count, value);
    break;
  case ::orteaf::internal::DType::U8:
    fillTyped<std::uint8_t>(data, count, value);
    break;
  case ::orteaf::internal::DType::U16:
    fillTyped<std::uint16_t>(data, count, value);
    break;
  case ::orteaf::internal::DType::U32:
    fillTyped<std::uint32_t>(data, count, value);
    break;
  case ::orteaf::internal::DType::U64:
    fillTyped<std::uint64_t>(data, count, value);
    break;
  case ::orteaf::internal::DType::F8E4M3:
    fillTyped<::orteaf::internal::Float8E4M3>(data, count, value);
    break;
  case ::orteaf::internal::DType::F8E5M2:
    fillTyped<::orteaf::internal::Float8E5M2>(data, count, value);
    break;
  case ::orteaf::internal::DType::F16:
    fillTyped<::orteaf::internal::Float16>(data, count, value);
    break;
  case ::orteaf::internal::DType::F32:
    fillTyped<float>(data, count, value);
    break;
  case ::orteaf::internal::DType::F64:
    fillTyped<double>(data, count, value);
    break;
  default:
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "Fill kernel received unsupported dtype");
  }
}

void fillTensorStrided(void *data, std::span<const std::int64_t> shape,
                       std::span<const std::int64_t> strides,
                       std::int64_t offset, ::orteaf::internal::DType dtype,
                       double value) {
  if (shape.size() != strides.size()) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "Fill kernel received mismatched shape/strides");
  }

  switch (dtype) {
  case ::orteaf::internal::DType::Bool:
    fillStridedTyped<bool>(data, shape, strides, offset, value);
    break;
  case ::orteaf::internal::DType::I8:
    fillStridedTyped<std::int8_t>(data, shape, strides, offset, value);
    break;
  case ::orteaf::internal::DType::I16:
    fillStridedTyped<std::int16_t>(data, shape, strides, offset, value);
    break;
  case ::orteaf::internal::DType::I32:
    fillStridedTyped<std::int32_t>(data, shape, strides, offset, value);
    break;
  case ::orteaf::internal::DType::I64:
    fillStridedTyped<std::int64_t>(data, shape, strides, offset, value);
    break;
  case ::orteaf::internal::DType::U8:
    fillStridedTyped<std::uint8_t>(data, shape, strides, offset, value);
    break;
  case ::orteaf::internal::DType::U16:
    fillStridedTyped<std::uint16_t>(data, shape, strides, offset, value);
    break;
  case ::orteaf::internal::DType::U32:
    fillStridedTyped<std::uint32_t>(data, shape, strides, offset, value);
    break;
  case ::orteaf::internal::DType::U64:
    fillStridedTyped<std::uint64_t>(data, shape, strides, offset, value);
    break;
  case ::orteaf::internal::DType::F8E4M3:
    fillStridedTyped<::orteaf::internal::Float8E4M3>(data, shape, strides,
                                                     offset, value);
    break;
  case ::orteaf::internal::DType::F8E5M2:
    fillStridedTyped<::orteaf::internal::Float8E5M2>(data, shape, strides,
                                                     offset, value);
    break;
  case ::orteaf::internal::DType::F16:
    fillStridedTyped<::orteaf::internal::Float16>(data, shape, strides, offset,
                                                  value);
    break;
  case ::orteaf::internal::DType::F32:
    fillStridedTyped<float>(data, shape, strides, offset, value);
    break;
  case ::orteaf::internal::DType::F64:
    fillStridedTyped<double>(data, shape, strides, offset, value);
    break;
  default:
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "Fill kernel received unsupported dtype");
  }
}

}  // namespace orteaf::extension::kernel::cpu
