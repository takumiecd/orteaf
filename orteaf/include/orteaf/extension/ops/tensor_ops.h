#pragma once

#include <orteaf/user/tensor/tensor.h>

namespace orteaf::extension::ops {

class TensorOps {
public:
  using Tensor = ::orteaf::user::tensor::Tensor;

  static void fill(Tensor &output, double value);
  static void print(const Tensor &input);
  static void copyHostToMps(Tensor &output, const Tensor &input);
  static void copyMpsToHost(Tensor &output, const Tensor &input);
};

} // namespace orteaf::extension::ops
