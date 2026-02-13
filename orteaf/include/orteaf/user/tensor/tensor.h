#pragma once

/**
 * @file tensor.h
 * @brief User-facing Tensor class with type-erased impl.
 *
 * Tensor automatically supports all impls registered in tensor_impl_types.h.
 * Adding a new impl to RegisteredImpls makes it usable in Tensor automatically.
 *
 * NO MANUAL EDITING REQUIRED - just add your impl to RegisteredImpls.
 */

#include <span>

#include <orteaf/extension/tensor/layout/dense_tensor_layout.h>
#include <orteaf/extension/tensor/registry/tensor_impl_types.h>
#include <orteaf/internal/dtype/dtype.h>
#include <orteaf/internal/execution/execution.h>

namespace orteaf::user::tensor {

// Use the auto-generated variant from registry
using TensorImplVariant =
    ::orteaf::internal::tensor::registry::RegisteredImpls::LeaseVariant;

/**
 * @brief User-facing tensor class.
 *
 * Automatically supports all impls registered in RegisteredImpls.
 *
 * @par Example
 * @code
 * std::array<Tensor::Dim, 2> shape{3, 4};
 * auto a = Tensor::denseBuilder()
 *              .withShape(shape)
 *              .withDType(DType::F32)
 *              .withExecution(Execution::Cpu)
 *              .build();
 * auto b = a.transpose({1, 0});
 * @endcode
 */
class Tensor {
public:
  using Layout = ::orteaf::extension::tensor::DenseTensorLayout;
  using Dims = Layout::Dims;
  using Dim = Layout::Dim;
  using DType = ::orteaf::internal::DType;
  using Execution = ::orteaf::internal::execution::Execution;
  using DenseBuilder = ::orteaf::extension::tensor::DenseTensorImpl::Builder;

  Tensor() = default;

  /// @brief Construct from any impl lease (via variant)
  template <typename ImplLease>
  explicit Tensor(ImplLease impl) : impl_(std::move(impl)) {}

  Tensor(const Tensor &) = default;
  Tensor &operator=(const Tensor &) = default;
  Tensor(Tensor &&) = default;
  Tensor &operator=(Tensor &&) = default;
  ~Tensor() = default;

  // ===== Factory methods =====

  static DenseBuilder denseBuilder();

  // Future: Tensor::coo(), Tensor::csr() generated automatically
  // by adding to RegisteredImpls

  // ===== Type queries =====

  bool valid() const noexcept;

  /// @brief Check if tensor holds a specific impl type
  template <typename Impl> bool is() const noexcept {
    using Manager = ::orteaf::internal::tensor::TensorImplManager<Impl>;
    using Lease = typename Manager::TensorImplLease;
    return std::holds_alternative<Lease>(impl_);
  }

  // ===== Accessors =====

  DType dtype() const;
  Execution execution() const;
  Dims shape() const;
  Dims strides() const;
  Dim numel() const;
  std::size_t rank() const;
  bool isContiguous() const;

  // ===== View operations =====

  Tensor transpose(std::span<const std::size_t> perm) const;
  Tensor slice(std::span<const Dim> starts, std::span<const Dim> sizes) const;
  Tensor reshape(std::span<const Dim> new_shape) const;
  Tensor squeeze() const;
  Tensor unsqueeze(std::size_t dim) const;

  // ===== Access to underlying impl =====

  const TensorImplVariant &implVariant() const noexcept { return impl_; }

  /// @brief Try to get as a specific impl type
  template <typename Impl> auto *tryAs() const noexcept {
    using Manager = ::orteaf::internal::tensor::TensorImplManager<Impl>;
    using Lease = typename Manager::TensorImplLease;
    return std::get_if<Lease>(&impl_);
  }

private:
  TensorImplVariant impl_{};
};

} // namespace orteaf::user::tensor
