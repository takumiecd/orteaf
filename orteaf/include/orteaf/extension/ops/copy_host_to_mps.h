#pragma once

#include <orteaf/user/tensor/tensor.h>

namespace orteaf::extension::ops {

/// @brief Copy tensor data from host (CPU) storage to MPS storage.
void copyHostToMps(::orteaf::user::tensor::Tensor &output,
                   const ::orteaf::user::tensor::Tensor &input);

} // namespace orteaf::extension::ops

