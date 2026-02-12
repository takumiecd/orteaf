#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include <orteaf/internal/base/checked_int.h>
#include <orteaf/internal/base/inline_vector.h>
#include <orteaf/internal/diagnostics/error/error.h>

namespace orteaf::extension::kernel::mps::detail {

namespace error = ::orteaf::internal::diagnostics::error;

inline constexpr std::uint8_t kTransferShapeInlineCapacity = 8;
using ShapeVector =
    ::orteaf::internal::base::InlineVector<std::int64_t,
                                           kTransferShapeInlineCapacity>;

struct LayoutInfo {
  std::size_t numel{};
  bool has_zero{};
  bool contiguous{};
  std::int64_t min_index{};
  std::int64_t max_index{};
};

struct TransferLayoutParams {
  std::uint32_t rank{};
  std::uint32_t shape[kTransferShapeInlineCapacity]{};
  std::int32_t src_strides[kTransferShapeInlineCapacity]{};
  std::int32_t dst_strides[kTransferShapeInlineCapacity]{};
};

inline LayoutInfo analyzeLayout(const ShapeVector &shape,
                                const ShapeVector &strides,
                                std::int64_t offset,
                                const char *op_name,
                                const char *tensor_name) {
  if (shape.size != strides.size) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      std::string(op_name) + ": " + tensor_name +
                          " has mismatched shape/strides");
  }

  LayoutInfo info{};
  info.numel = 1;
  info.has_zero = false;
  info.contiguous = true;
  info.min_index = offset;
  info.max_index = offset;

  if (shape.size == 0) {
    return info;
  }

  std::size_t expected_stride = 1;
  for (std::size_t i = shape.size; i-- > 0;) {
    const auto dim = shape.data[i];
    if (dim < 0) {
      error::throwError(
          error::OrteafErrc::InvalidParameter,
          std::string(op_name) + ": " + tensor_name +
              " has negative shape dimension");
    }
    if (dim == 0) {
      info.numel = 0;
      info.has_zero = true;
      return info;
    }

    const auto dim_size = static_cast<std::size_t>(dim);
    if (info.numel > std::numeric_limits<std::size_t>::max() / dim_size) {
      error::throwError(error::OrteafErrc::InvalidParameter,
                        std::string(op_name) + ": " + tensor_name +
                            " shape is too large");
    }
    info.numel *= dim_size;

    if (strides.data[i] != static_cast<std::int64_t>(expected_stride)) {
      info.contiguous = false;
    }
    if (expected_stride > std::numeric_limits<std::size_t>::max() / dim_size) {
      error::throwError(error::OrteafErrc::InvalidParameter,
                        std::string(op_name) + ": " + tensor_name +
                            " shape is too large");
    }
    expected_stride *= dim_size;
  }

  std::int64_t min_index = offset;
  std::int64_t max_index = offset;
  for (std::size_t i = 0; i < shape.size; ++i) {
    const auto dim = shape.data[i];
    if (dim <= 0) {
      continue;
    }
    const auto stride = strides.data[i];
    std::int64_t span = 0;
    if (::orteaf::internal::base::mulOverflowI64(stride, dim - 1, span)) {
      error::throwError(error::OrteafErrc::InvalidParameter,
                        std::string(op_name) + ": " + tensor_name +
                            " index range overflow");
    }
    if (stride >= 0) {
      if (::orteaf::internal::base::addOverflowI64(max_index, span,
                                                   max_index)) {
        error::throwError(error::OrteafErrc::InvalidParameter,
                          std::string(op_name) + ": " + tensor_name +
                              " index range overflow");
      }
    } else {
      if (::orteaf::internal::base::addOverflowI64(min_index, span,
                                                   min_index)) {
        error::throwError(error::OrteafErrc::InvalidParameter,
                          std::string(op_name) + ": " + tensor_name +
                              " index range overflow");
      }
    }
  }

  info.min_index = min_index;
  info.max_index = max_index;
  return info;
}

inline void ensureSameShape(const ShapeVector &lhs, const ShapeVector &rhs,
                            const char *op_name) {
  if (lhs.size != rhs.size) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      std::string(op_name) + ": input/output rank mismatch");
  }
  for (std::size_t i = 0; i < lhs.size; ++i) {
    if (lhs.data[i] != rhs.data[i]) {
      error::throwError(error::OrteafErrc::InvalidParameter,
                        std::string(op_name) + ": input/output shape mismatch");
    }
  }
}

inline void fillLayoutParams(TransferLayoutParams &params,
                             const ShapeVector &shape,
                             const ShapeVector &src_strides,
                             const ShapeVector &dst_strides,
                             const char *op_name) {
  if (shape.size != src_strides.size || shape.size != dst_strides.size) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      std::string(op_name) +
                          ": shape/stride rank mismatch in layout params");
  }
  params.rank = shape.size;
  for (std::size_t i = 0; i < shape.size; ++i) {
    const auto dim = shape.data[i];
    const auto src_stride = src_strides.data[i];
    const auto dst_stride = dst_strides.data[i];
    if (dim < 0 || dim > static_cast<std::int64_t>(
                            std::numeric_limits<std::uint32_t>::max())) {
      error::throwError(error::OrteafErrc::InvalidParameter,
                        std::string(op_name) +
                            ": shape dimension exceeds uint32 range");
    }
    if (src_stride < static_cast<std::int64_t>(
                         std::numeric_limits<std::int32_t>::min()) ||
        src_stride > static_cast<std::int64_t>(
                         std::numeric_limits<std::int32_t>::max()) ||
        dst_stride < static_cast<std::int64_t>(
                         std::numeric_limits<std::int32_t>::min()) ||
        dst_stride > static_cast<std::int64_t>(
                         std::numeric_limits<std::int32_t>::max())) {
      error::throwError(error::OrteafErrc::InvalidParameter,
                        std::string(op_name) +
                            ": stride exceeds int32 range");
    }
    params.shape[i] = static_cast<std::uint32_t>(dim);
    params.src_strides[i] = static_cast<std::int32_t>(src_stride);
    params.dst_strides[i] = static_cast<std::int32_t>(dst_stride);
  }
}

} // namespace orteaf::extension::kernel::mps::detail

