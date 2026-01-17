#include "orteaf/user/tensor/tensor.h"

#include "orteaf/internal/diagnostics/error/error.h"
#include "orteaf/internal/tensor/api/tensor_api.h"

namespace orteaf::user::tensor {

namespace {

using TensorApi = ::orteaf::internal::tensor::api::TensorApi;
using DenseTensorImpl = ::orteaf::extension::tensor::DenseTensorImpl;

void ensureValid(const Tensor &t) {
  if (!t.valid()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "Tensor is not valid");
  }
}

} // namespace

Tensor Tensor::dense(std::span<const Dim> shape, DType dtype,
                     Execution execution, std::size_t alignment) {
  auto impl =
      TensorApi::create<DenseTensorImpl>(shape, dtype, execution, alignment);
  return Tensor(std::move(impl));
}

bool Tensor::valid() const noexcept {
  return !std::holds_alternative<std::monostate>(impl_);
}

Tensor::DType Tensor::dtype() const {
  ensureValid(*this);
  return std::visit(
      [](const auto &impl) -> DType {
        using T = std::decay_t<decltype(impl)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          return DType::F32; // Unreachable
        } else {
          return impl->dtype();
        }
      },
      impl_);
}

Tensor::Execution Tensor::execution() const {
  ensureValid(*this);
  return std::visit(
      [](const auto &impl) -> Execution {
        using T = std::decay_t<decltype(impl)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          return Execution::Cpu; // Unreachable
        } else {
          return impl->execution();
        }
      },
      impl_);
}

Tensor::Dims Tensor::shape() const {
  ensureValid(*this);
  return std::visit(
      [](const auto &impl) -> Dims {
        using T = std::decay_t<decltype(impl)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          return {}; // Unreachable
        } else {
          return impl->shape();
        }
      },
      impl_);
}

Tensor::Dims Tensor::strides() const {
  ensureValid(*this);
  return std::visit(
      [](const auto &impl) -> Dims {
        using T = std::decay_t<decltype(impl)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          return {}; // Unreachable
        } else {
          return impl->strides();
        }
      },
      impl_);
}

Tensor::Dim Tensor::numel() const {
  ensureValid(*this);
  return std::visit(
      [](const auto &impl) -> Dim {
        using T = std::decay_t<decltype(impl)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          return 0; // Unreachable
        } else {
          return impl->numel();
        }
      },
      impl_);
}

std::size_t Tensor::rank() const {
  ensureValid(*this);
  return std::visit(
      [](const auto &impl) -> std::size_t {
        using T = std::decay_t<decltype(impl)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          return 0; // Unreachable
        } else {
          return impl->rank();
        }
      },
      impl_);
}

bool Tensor::isContiguous() const {
  ensureValid(*this);
  return std::visit(
      [](const auto &impl) -> bool {
        using T = std::decay_t<decltype(impl)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          return true; // Unreachable
        } else {
          return impl->isContiguous();
        }
      },
      impl_);
}

// ===== View operations - auto-dispatch via TensorApi =====

Tensor Tensor::transpose(std::span<const std::size_t> perm) const {
  ensureValid(*this);
  auto new_impl = TensorApi::transpose(impl_, perm);
  Tensor result;
  result.impl_ = std::move(new_impl);
  return result;
}

Tensor Tensor::slice(std::span<const Dim> starts,
                     std::span<const Dim> sizes) const {
  ensureValid(*this);
  auto new_impl = TensorApi::slice(impl_, starts, sizes);
  Tensor result;
  result.impl_ = std::move(new_impl);
  return result;
}

Tensor Tensor::reshape(std::span<const Dim> new_shape) const {
  ensureValid(*this);
  auto new_impl = TensorApi::reshape(impl_, new_shape);
  Tensor result;
  result.impl_ = std::move(new_impl);
  return result;
}

Tensor Tensor::squeeze() const {
  ensureValid(*this);
  auto new_impl = TensorApi::squeeze(impl_);
  Tensor result;
  result.impl_ = std::move(new_impl);
  return result;
}

Tensor Tensor::unsqueeze(std::size_t dim) const {
  ensureValid(*this);
  auto new_impl = TensorApi::unsqueeze(impl_, dim);
  Tensor result;
  result.impl_ = std::move(new_impl);
  return result;
}

} // namespace orteaf::user::tensor
