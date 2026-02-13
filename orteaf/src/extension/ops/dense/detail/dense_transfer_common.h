#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>

#include <orteaf/internal/base/checked_int.h>
#include <orteaf/internal/diagnostics/error/error.h>
#include <orteaf/internal/execution/execution.h>
#include <orteaf/internal/storage/registry/storage_types.h>

#include "dense_op_common.h"

namespace orteaf::extension::ops::dense::detail::transfer {

namespace error = ::orteaf::internal::diagnostics::error;
using DenseTensorImpl = ::orteaf::extension::tensor::DenseTensorImpl;
using Execution = ::orteaf::internal::execution::Execution;

inline constexpr std::uint8_t kTransferShapeInlineCapacity = 8;

struct LayoutStats {
  std::size_t numel{};
  bool has_zero{};
  bool contiguous{};
  std::int64_t min_index{};
  std::int64_t max_index{};
};

inline LayoutStats analyzeLayout(std::span<const std::int64_t> shape,
                                 std::span<const std::int64_t> strides,
                                 std::int64_t offset,
                                 const char *op_name,
                                 const char *tensor_name) {
  if (shape.size() != strides.size()) {
    error::throwError(
        error::OrteafErrc::InvalidParameter,
        std::string(op_name) + ": " + tensor_name +
            " has mismatched shape/strides");
  }

  LayoutStats stats{};
  stats.numel = 1;
  stats.has_zero = false;
  stats.contiguous = true;
  stats.min_index = offset;
  stats.max_index = offset;

  if (shape.empty()) {
    return stats;
  }

  std::size_t expected_stride = 1;
  for (std::size_t i = shape.size(); i-- > 0;) {
    const auto dim = shape[i];
    if (dim < 0) {
      error::throwError(
          error::OrteafErrc::InvalidParameter,
          std::string(op_name) + ": " + tensor_name +
              " has negative shape dimension");
    }
    if (dim == 0) {
      stats.numel = 0;
      stats.has_zero = true;
      return stats;
    }

    const auto dim_size = static_cast<std::size_t>(dim);
    if (stats.numel > std::numeric_limits<std::size_t>::max() / dim_size) {
      error::throwError(
          error::OrteafErrc::InvalidParameter,
          std::string(op_name) + ": " + tensor_name + " shape is too large");
    }
    stats.numel *= dim_size;

    if (strides[i] != static_cast<std::int64_t>(expected_stride)) {
      stats.contiguous = false;
    }
    if (expected_stride > std::numeric_limits<std::size_t>::max() / dim_size) {
      error::throwError(
          error::OrteafErrc::InvalidParameter,
          std::string(op_name) + ": " + tensor_name + " shape is too large");
    }
    expected_stride *= dim_size;
  }

  std::int64_t min_index = offset;
  std::int64_t max_index = offset;
  for (std::size_t i = 0; i < shape.size(); ++i) {
    const auto dim = shape[i];
    if (dim <= 0) {
      continue;
    }
    const auto stride = strides[i];
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

  stats.min_index = min_index;
  stats.max_index = max_index;
  return stats;
}

inline LayoutStats validateViewBounds(const DenseTensorImpl *impl,
                                      const char *op_name,
                                      const char *tensor_name) {
  const auto storage_numel_raw = impl->storageLease().numel();
  if (storage_numel_raw >
      static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      std::string(op_name) + ": " + tensor_name +
                          " storage is too large");
  }
  const auto storage_numel = static_cast<std::int64_t>(storage_numel_raw);
  const auto offset = impl->offset();
  if (offset < 0) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      std::string(op_name) + ": " + tensor_name +
                          " has negative offset");
  }

  const auto &shape = impl->shape();
  const auto &strides = impl->strides();
  const auto stats = analyzeLayout(
      std::span<const std::int64_t>(shape.data(), shape.size()),
      std::span<const std::int64_t>(strides.data(), strides.size()), offset,
      op_name, tensor_name);

  if (!stats.has_zero &&
      (stats.min_index < 0 || stats.max_index < 0 ||
       stats.max_index >= storage_numel)) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      std::string(op_name) + ": " + tensor_name +
                          " view exceeds storage bounds");
  }

  return stats;
}

inline void requireExecution(const DenseTensorImpl *impl, Execution expected,
                             const char *op_name, const char *tensor_name) {
  if (impl->execution() != expected) {
    error::throwError(error::OrteafErrc::ExecutionUnavailable,
                      std::string(op_name) + ": " + tensor_name +
                          " has unsupported execution");
  }
}

inline void requireMatchingShapeAndDType(const DenseTensorImpl *output,
                                         const DenseTensorImpl *input,
                                         const char *op_name) {
  if (output->dtype() != input->dtype()) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      std::string(op_name) +
                          ": input/output dtype must match");
  }
  if (output->rank() != input->rank()) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      std::string(op_name) +
                          ": input/output rank must match");
  }
  const auto &out_shape = output->shape();
  const auto &in_shape = input->shape();
  for (std::size_t i = 0; i < out_shape.size(); ++i) {
    if (out_shape[i] != in_shape[i]) {
      error::throwError(error::OrteafErrc::InvalidParameter,
                        std::string(op_name) +
                            ": input/output shape must match");
    }
  }
}

inline void ensureRankSupported(const DenseTensorImpl *impl,
                                const char *op_name) {
  if (impl->rank() > kTransferShapeInlineCapacity) {
    error::throwError(error::OrteafErrc::Unsupported,
                      std::string(op_name) + ": rank > 8 is unsupported on MPS");
  }
}

} // namespace orteaf::extension::ops::dense::detail::transfer
