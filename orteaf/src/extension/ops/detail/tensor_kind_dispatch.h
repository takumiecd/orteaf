#pragma once

#include <string>
#include <utility>

#include <orteaf/extension/tensor/dense_tensor_impl.h>
#include <orteaf/internal/diagnostics/error/error.h>
#include <orteaf/user/tensor/tensor.h>

namespace orteaf::extension::ops::detail {

using Tensor = ::orteaf::user::tensor::Tensor;
using DenseTensorImpl = ::orteaf::extension::tensor::DenseTensorImpl;
namespace error = ::orteaf::internal::diagnostics::error;

template <typename Impl> struct ImplTag {
  using type = Impl;
};

[[noreturn]] inline void throwUnsupportedTensorKind(const char *op_name) {
  error::throwError(error::OrteafErrc::Unsupported,
                    std::string(op_name) +
                        ": unsupported tensor implementation");
}

inline void ensureValidTensor(const Tensor &tensor, const char *op_name) {
  if (!tensor.valid()) {
    error::throwError(error::OrteafErrc::InvalidState,
                      std::string(op_name) + ": tensor is not valid");
  }
}

#define ORTEAF_EXTENSION_OPS_FOR_EACH_IMPL_KIND(M)                           \
  M(::orteaf::extension::tensor::DenseTensorImpl)

template <typename Router>
inline bool dispatchOutputKind(Tensor &output, Router &&router) {
#define ORTEAF_EXTENSION_OPS_MATCH_OUTPUT_KIND(ImplType)                      \
  if (output.is<ImplType>()) {                                                \
    std::forward<Router>(router)(ImplTag<ImplType>{});                        \
    return true;                                                               \
  }
  ORTEAF_EXTENSION_OPS_FOR_EACH_IMPL_KIND(ORTEAF_EXTENSION_OPS_MATCH_OUTPUT_KIND)
#undef ORTEAF_EXTENSION_OPS_MATCH_OUTPUT_KIND
  return false;
}

template <typename Router>
inline bool dispatchInputKind(const Tensor &input, Router &&router) {
#define ORTEAF_EXTENSION_OPS_MATCH_INPUT_KIND(ImplType)                       \
  if (input.is<ImplType>()) {                                                 \
    std::forward<Router>(router)(ImplTag<ImplType>{});                        \
    return true;                                                               \
  }
  ORTEAF_EXTENSION_OPS_FOR_EACH_IMPL_KIND(ORTEAF_EXTENSION_OPS_MATCH_INPUT_KIND)
#undef ORTEAF_EXTENSION_OPS_MATCH_INPUT_KIND
  return false;
}

template <typename Router>
inline bool dispatchSameKind(Tensor &output, const Tensor &input,
                             Router &&router) {
#define ORTEAF_EXTENSION_OPS_MATCH_SAME_KIND(ImplType)                        \
  if (output.is<ImplType>() && input.is<ImplType>()) {                        \
    std::forward<Router>(router)(ImplTag<ImplType>{});                        \
    return true;                                                               \
  }
  ORTEAF_EXTENSION_OPS_FOR_EACH_IMPL_KIND(ORTEAF_EXTENSION_OPS_MATCH_SAME_KIND)
#undef ORTEAF_EXTENSION_OPS_MATCH_SAME_KIND
  return false;
}

template <typename Router>
inline void dispatch_out(Tensor &output, const char *op_name, Router &&router) {
  ensureValidTensor(output, op_name);
  if (dispatchOutputKind(output, std::forward<Router>(router))) {
    return;
  }
  throwUnsupportedTensorKind(op_name);
}

template <typename Router>
inline void dispatch_in(const Tensor &input, const char *op_name,
                        Router &&router) {
  ensureValidTensor(input, op_name);
  if (dispatchInputKind(input, std::forward<Router>(router))) {
    return;
  }
  throwUnsupportedTensorKind(op_name);
}

template <typename Router>
inline void dispatch_out_in(Tensor &output, const Tensor &input,
                            const char *op_name, Router &&router) {
  ensureValidTensor(output, op_name);
  ensureValidTensor(input, op_name);
  if (dispatchSameKind(output, input, std::forward<Router>(router))) {
    return;
  }
  throwUnsupportedTensorKind(op_name);
}

#undef ORTEAF_EXTENSION_OPS_FOR_EACH_IMPL_KIND

} // namespace orteaf::extension::ops::detail
