#include <cstddef>
#include <cstdint>

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

}  // namespace orteaf::extension::kernel::cpu
