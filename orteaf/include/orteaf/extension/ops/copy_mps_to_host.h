#pragma once

#include <orteaf/user/tensor/tensor.h>

namespace orteaf::extension::ops {

/// @brief Copy tensor data from MPS storage to host (CPU) storage.
void copyMpsToHost(::orteaf::user::tensor::Tensor &output,
                   const ::orteaf::user::tensor::Tensor &input);

} // namespace orteaf::extension::ops

