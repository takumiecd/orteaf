#pragma once

#include <concepts>

#include <orteaf/extension/ops/dense/dense_tensor_ops.h>
#include <orteaf/extension/tensor/dense_tensor_impl.h>
#include <orteaf/user/tensor/tensor.h>

namespace orteaf::extension::ops::detail {

using Tensor = ::orteaf::user::tensor::Tensor;
using DenseTensorImpl = ::orteaf::extension::tensor::DenseTensorImpl;

template <typename Backend>
concept TensorKindBackendOps = requires(Tensor &output, const Tensor &input,
                                        double value) {
  { Backend::fill(output, value) } -> std::same_as<void>;
  { Backend::print(input) } -> std::same_as<void>;
  { Backend::copyHostToMps(output, input) } -> std::same_as<void>;
  { Backend::copyMpsToHost(output, input) } -> std::same_as<void>;
};

template <typename Impl> struct KindOps;

template <> struct KindOps<DenseTensorImpl> {
  using Backend = ::orteaf::extension::ops::dense::DenseTensorOps;
};

template <typename Impl>
concept HasKindOpsBackend = requires { typename KindOps<Impl>::Backend; } &&
                            TensorKindBackendOps<typename KindOps<Impl>::Backend>;

template <typename Impl>
inline void kindFill(Tensor &output, double value) {
  static_assert(HasKindOpsBackend<Impl>,
                "KindOps backend is missing required fill/print/copy methods");
  KindOps<Impl>::Backend::fill(output, value);
}

template <typename Impl>
inline void kindPrint(const Tensor &input) {
  static_assert(HasKindOpsBackend<Impl>,
                "KindOps backend is missing required fill/print/copy methods");
  KindOps<Impl>::Backend::print(input);
}

template <typename Impl>
inline void kindCopyHostToMps(Tensor &output, const Tensor &input) {
  static_assert(HasKindOpsBackend<Impl>,
                "KindOps backend is missing required fill/print/copy methods");
  KindOps<Impl>::Backend::copyHostToMps(output, input);
}

template <typename Impl>
inline void kindCopyMpsToHost(Tensor &output, const Tensor &input) {
  static_assert(HasKindOpsBackend<Impl>,
                "KindOps backend is missing required fill/print/copy methods");
  KindOps<Impl>::Backend::copyMpsToHost(output, input);
}

} // namespace orteaf::extension::ops::detail
