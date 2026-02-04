#pragma once

#include <orteaf/user/tensor/tensor.h>

namespace orteaf::extension::ops {

/// @brief Fill the output tensor with a scalar value (in-place).
void fill(::orteaf::user::tensor::Tensor &output, double value);

}  // namespace orteaf::extension::ops
