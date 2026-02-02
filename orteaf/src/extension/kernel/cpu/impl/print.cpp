#include "orteaf/extension/kernel/cpu/print.h"

#include <functional>
#include <iomanip>
#include <ostream>
#include <vector>

namespace orteaf::extension::kernel::cpu {

namespace {

class StreamStateGuard {
public:
  explicit StreamStateGuard(std::ostream &os)
      : os_(os), flags_(os.flags()), precision_(os.precision()) {}
  ~StreamStateGuard() {
    os_.flags(flags_);
    os_.precision(precision_);
  }

private:
  std::ostream &os_;
  std::ios::fmtflags flags_;
  std::streamsize precision_;
};

template <typename T> void printScalar(std::ostream &os, T value) {
  os << value;
}

template <> void printScalar<std::int8_t>(std::ostream &os, std::int8_t value) {
  os << static_cast<int>(value);
}

template <> void printScalar<std::uint8_t>(std::ostream &os, std::uint8_t value) {
  os << static_cast<unsigned int>(value);
}

template <> void printScalar<bool>(std::ostream &os, bool value) {
  os << (value ? "true" : "false");
}

template <>
void printScalar<::orteaf::internal::Float16>(std::ostream &os,
                                              ::orteaf::internal::Float16 value) {
  os << value.toFloat32();
}

template <>
void printScalar<::orteaf::internal::Float8E4M3>(
    std::ostream &os, ::orteaf::internal::Float8E4M3 value) {
  os << value.toFloat32();
}

template <>
void printScalar<::orteaf::internal::Float8E5M2>(
    std::ostream &os, ::orteaf::internal::Float8E5M2 value) {
  os << value.toFloat32();
}

template <typename T>
void printTensorImpl(std::span<const std::int64_t> shape, const void *buffer,
                     std::ostream &os) {
  if (buffer == nullptr) {
    os << "<null>";
    return;
  }

  for (const auto dim : shape) {
    if (dim < 0) {
      os << "<invalid shape>";
      return;
    }
  }

  if (shape.empty()) {
    const auto *typed = static_cast<const T *>(buffer);
    printScalar(os, typed[0]);
    return;
  }

  std::vector<std::size_t> dims(shape.size());
  for (std::size_t i = 0; i < shape.size(); ++i) {
    dims[i] = static_cast<std::size_t>(shape[i]);
  }

  std::vector<std::size_t> strides(shape.size(), 1);
  for (std::size_t i = shape.size(); i-- > 1;) {
    strides[i - 1] = strides[i] * dims[i];
  }

  const auto *typed = static_cast<const T *>(buffer);
  constexpr std::size_t kIndentStep = 2;

  auto indent = [&](std::size_t depth) {
    for (std::size_t i = 0; i < depth * kIndentStep; ++i) {
      os << ' ';
    }
  };

  std::function<void(std::size_t, std::size_t)> printDim;
  printDim = [&](std::size_t depth, std::size_t base_index) {
    if (dims[depth] == 0) {
      os << "[]";
      return;
    }

    if (depth + 1 == dims.size()) {
      os << '[';
      for (std::size_t i = 0; i < dims[depth]; ++i) {
        if (i > 0) {
          os << ", ";
        }
        printScalar(os, typed[base_index + i * strides[depth]]);
      }
      os << ']';
      return;
    }

    os << '[';
    os << '\n';
    for (std::size_t i = 0; i < dims[depth]; ++i) {
      if (i > 0) {
        os << ",\n";
      }
      indent(depth + 1);
      printDim(depth + 1, base_index + i * strides[depth]);
    }
    os << '\n';
    indent(depth);
    os << ']';
  };

  printDim(0, 0);
}

}  // namespace

void printTensor(std::span<const std::int64_t> shape, const void *buffer,
                 ::orteaf::internal::DType dtype, std::ostream &os) {
  StreamStateGuard guard(os);
  os << std::setprecision(6) << std::defaultfloat;

  switch (dtype) {
  case ::orteaf::internal::DType::Bool:
    printTensorImpl<bool>(shape, buffer, os);
    break;
  case ::orteaf::internal::DType::I8:
    printTensorImpl<std::int8_t>(shape, buffer, os);
    break;
  case ::orteaf::internal::DType::I16:
    printTensorImpl<std::int16_t>(shape, buffer, os);
    break;
  case ::orteaf::internal::DType::I32:
    printTensorImpl<std::int32_t>(shape, buffer, os);
    break;
  case ::orteaf::internal::DType::I64:
    printTensorImpl<std::int64_t>(shape, buffer, os);
    break;
  case ::orteaf::internal::DType::U8:
    printTensorImpl<std::uint8_t>(shape, buffer, os);
    break;
  case ::orteaf::internal::DType::U16:
    printTensorImpl<std::uint16_t>(shape, buffer, os);
    break;
  case ::orteaf::internal::DType::U32:
    printTensorImpl<std::uint32_t>(shape, buffer, os);
    break;
  case ::orteaf::internal::DType::U64:
    printTensorImpl<std::uint64_t>(shape, buffer, os);
    break;
  case ::orteaf::internal::DType::F8E4M3:
    printTensorImpl<::orteaf::internal::Float8E4M3>(shape, buffer, os);
    break;
  case ::orteaf::internal::DType::F8E5M2:
    printTensorImpl<::orteaf::internal::Float8E5M2>(shape, buffer, os);
    break;
  case ::orteaf::internal::DType::F16:
    printTensorImpl<::orteaf::internal::Float16>(shape, buffer, os);
    break;
  case ::orteaf::internal::DType::F32:
    printTensorImpl<float>(shape, buffer, os);
    break;
  case ::orteaf::internal::DType::F64:
    printTensorImpl<double>(shape, buffer, os);
    break;
  default:
    os << "<unsupported dtype>";
    break;
  }
}

}  // namespace orteaf::extension::kernel::cpu
