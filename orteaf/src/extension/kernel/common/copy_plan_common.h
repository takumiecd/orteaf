#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include <orteaf/internal/diagnostics/error/error.h>
#include <orteaf/internal/dtype/dtype.h>

#include "layout_common.h"

namespace orteaf::extension::kernel::common::copy_plan {

namespace error = ::orteaf::internal::diagnostics::error;
namespace common_layout = ::orteaf::extension::kernel::common::layout;

struct CopyValidation {
  common_layout::LayoutInfo input_layout{};
  common_layout::LayoutInfo output_layout{};
  std::size_t numel{};
  std::size_t elem_size{};
  std::size_t bytes{};
  bool has_zero{};
  bool input_contiguous{};
  bool output_contiguous{};
};

inline CopyValidation validateCopyLayouts(
    const common_layout::ShapeVector &input_shape,
    const common_layout::ShapeVector &input_strides, std::int64_t input_offset,
    std::int64_t input_storage_numel,
    const common_layout::ShapeVector &output_shape,
    const common_layout::ShapeVector &output_strides, std::int64_t output_offset,
    std::int64_t output_storage_numel, ::orteaf::internal::DType dtype,
    const char *op_name) {
  if (input_offset < 0 || output_offset < 0) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      std::string(op_name) + " received negative offset");
  }

  common_layout::ensureSameShape(input_shape, output_shape, op_name);

  CopyValidation validation{};
  validation.input_layout = common_layout::analyzeLayout(
      input_shape, input_strides, input_offset, op_name, "input");
  validation.output_layout = common_layout::analyzeLayout(
      output_shape, output_strides, output_offset, op_name, "output");
  validation.has_zero = validation.input_layout.has_zero;
  validation.input_contiguous = validation.input_layout.contiguous;
  validation.output_contiguous = validation.output_layout.contiguous;
  validation.numel = validation.input_layout.numel;
  validation.elem_size = ::orteaf::internal::sizeOf(dtype);

  if (validation.has_zero) {
    validation.bytes = 0;
    return validation;
  }

  if (validation.input_layout.numel != validation.output_layout.numel) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      std::string(op_name) + " numel mismatch");
  }

  if (validation.input_layout.min_index < 0 || validation.input_layout.max_index < 0 ||
      validation.input_layout.max_index >= input_storage_numel ||
      validation.output_layout.min_index < 0 ||
      validation.output_layout.max_index < 0 ||
      validation.output_layout.max_index >= output_storage_numel) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      std::string(op_name) + " view exceeds storage bounds");
  }

  if (validation.numel >
      std::numeric_limits<std::size_t>::max() / validation.elem_size) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      std::string(op_name) + " byte size overflow");
  }
  validation.bytes = validation.numel * validation.elem_size;
  return validation;
}

} // namespace orteaf::extension::kernel::common::copy_plan
