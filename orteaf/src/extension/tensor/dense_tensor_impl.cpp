#include "orteaf/extension/tensor/dense_tensor_impl.h"

#include "orteaf/internal/diagnostics/error/error.h"
#include "orteaf/internal/tensor/api/tensor_api.h"
#include "orteaf/user/tensor/tensor.h"

namespace orteaf::extension::tensor {

DenseTensorImpl::Builder DenseTensorImpl::builder() { return Builder{}; }

DenseTensorImpl::Builder &
DenseTensorImpl::Builder::withShape(std::span<const Dim> shape) {
  request_.shape.assign(shape.begin(), shape.end());
  shape_set_ = true;
  return *this;
}

DenseTensorImpl::Builder &
DenseTensorImpl::Builder::withDType(DType dtype) noexcept {
  request_.dtype = dtype;
  return *this;
}

DenseTensorImpl::Builder &
DenseTensorImpl::Builder::withExecution(Execution execution) noexcept {
  request_.execution = execution;
  execution_set_ = true;
  return *this;
}

DenseTensorImpl::Builder &
DenseTensorImpl::Builder::withAlignment(std::size_t alignment) noexcept {
  request_.alignment = alignment;
  return *this;
}

::orteaf::user::tensor::Tensor DenseTensorImpl::Builder::build() const {
  if (!shape_set_) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
        "DenseTensorImpl::Builder requires withShape() before build()");
  }
  if (!execution_set_) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
        "DenseTensorImpl::Builder requires withExecution() before build()");
  }

  auto impl =
      ::orteaf::internal::tensor::api::TensorApi::create<DenseTensorImpl>(
          request_);
  return ::orteaf::user::tensor::Tensor(std::move(impl));
}

} // namespace orteaf::extension::tensor
