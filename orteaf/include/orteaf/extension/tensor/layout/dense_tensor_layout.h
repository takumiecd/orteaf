#pragma once

/**
 * @file dense_tensor_layout.h
 * @brief Dense/strided tensor layout description for view operations.
 *
 * Stores shape, strides, and element offset. All values are expressed in
 * element units so the layout stays dtype-agnostic.
 */

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

#include <orteaf/internal/base/small_vector.h>
#include <orteaf/internal/diagnostics/error/error.h>

namespace orteaf::extension::tensor {

class DenseTensorLayout {
public:
  using Dim = std::int64_t;
  using Dims = ::orteaf::internal::base::SmallVector<Dim, 4>;
  using size_type = std::size_t;

  DenseTensorLayout() = default;

  DenseTensorLayout(Dims shape, Dims strides, Dim offset = 0)
      : shape_(std::move(shape)), strides_(std::move(strides)),
        offset_(offset) {
    validateShape(shape_);
    validateRankMatch(shape_, strides_);
  }

  static DenseTensorLayout contiguous(std::span<const Dim> shape) {
    validateShape(shape);

    Dims shape_dims;
    shape_dims.assign(shape.begin(), shape.end());
    Dims strides = makeContiguousStrides(shape);
    return DenseTensorLayout(std::move(shape_dims), std::move(strides), 0);
  }

  static DenseTensorLayout contiguous(const Dims &shape) {
    return contiguous(std::span<const Dim>(shape.data(), shape.size()));
  }

  size_type rank() const noexcept { return shape_.size(); }
  const Dims &shape() const noexcept { return shape_; }
  const Dims &strides() const noexcept { return strides_; }
  Dim offset() const noexcept { return offset_; }

  Dim numel() const noexcept { return numelOf(shape_); }

  bool isContiguous() const noexcept {
    if (shape_.empty()) {
      return true;
    }
    Dim expected = 1;
    for (size_type idx = shape_.size(); idx > 0; --idx) {
      const size_type dim = idx - 1;
      if (strides_[dim] != expected) {
        return false;
      }
      expected *= shape_[dim];
    }
    return true;
  }

  DenseTensorLayout transpose(std::span<const size_type> perm) const {
    if (perm.size() != rank()) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
          "DenseTensorLayout transpose requires perm size to match rank");
    }

    Dims new_shape;
    Dims new_strides;
    new_shape.resize(rank());
    new_strides.resize(rank());

    for (size_type i = 0; i < perm.size(); ++i) {
      const size_type dim = perm[i];
      if (dim >= rank()) {
        ::orteaf::internal::diagnostics::error::throwError(
            ::orteaf::internal::diagnostics::error::OrteafErrc::OutOfRange,
            "DenseTensorLayout transpose perm entry out of range");
      }
      for (size_type j = 0; j < i; ++j) {
        if (perm[j] == dim) {
          ::orteaf::internal::diagnostics::error::throwError(
              ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
              "DenseTensorLayout transpose perm contains duplicates");
        }
      }
      new_shape[i] = shape_[dim];
      new_strides[i] = strides_[dim];
    }

    return DenseTensorLayout(std::move(new_shape), std::move(new_strides),
                             offset_);
  }

  DenseTensorLayout slice(std::span<const Dim> starts,
                          std::span<const Dim> sizes) const {
    return slice(starts, sizes, {});
  }

  DenseTensorLayout slice(std::span<const Dim> starts,
                          std::span<const Dim> sizes,
                          std::span<const Dim> steps) const {
    const size_type dims = rank();
    if (starts.size() != dims || sizes.size() != dims ||
        (!steps.empty() && steps.size() != dims)) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
          "DenseTensorLayout slice requires starts/sizes/steps to match rank");
    }

    Dims new_shape = shape_;
    Dims new_strides = strides_;
    Dim new_offset = offset_;

    for (size_type i = 0; i < dims; ++i) {
      const Dim dim = shape_[i];
      const Dim start = starts[i];
      const Dim size = sizes[i];
      const Dim step = steps.empty() ? Dim{1} : steps[i];

      if (start < 0 || start > dim) {
        ::orteaf::internal::diagnostics::error::throwError(
            ::orteaf::internal::diagnostics::error::OrteafErrc::OutOfRange,
            "DenseTensorLayout slice start out of range");
      }
      if (size < 0) {
        ::orteaf::internal::diagnostics::error::throwError(
            ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
            "DenseTensorLayout slice size must be non-negative");
      }
      if (step == 0) {
        ::orteaf::internal::diagnostics::error::throwError(
            ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
            "DenseTensorLayout slice step cannot be zero");
      }

      if (size == 0) {
        new_shape[i] = 0;
        new_strides[i] = strides_[i] * step;
        new_offset += start * strides_[i];
        continue;
      }

      if (start >= dim) {
        ::orteaf::internal::diagnostics::error::throwError(
            ::orteaf::internal::diagnostics::error::OrteafErrc::OutOfRange,
            "DenseTensorLayout slice start exceeds dimension");
      }

      const Dim last = start + (size - 1) * step;
      if (step > 0) {
        if (last >= dim) {
          ::orteaf::internal::diagnostics::error::throwError(
              ::orteaf::internal::diagnostics::error::OrteafErrc::OutOfRange,
              "DenseTensorLayout slice exceeds dimension");
        }
      } else {
        if (last < 0) {
          ::orteaf::internal::diagnostics::error::throwError(
              ::orteaf::internal::diagnostics::error::OrteafErrc::OutOfRange,
              "DenseTensorLayout slice exceeds dimension");
        }
      }

      new_shape[i] = size;
      new_strides[i] = strides_[i] * step;
      new_offset += start * strides_[i];
    }

    return DenseTensorLayout(std::move(new_shape), std::move(new_strides),
                             new_offset);
  }

  DenseTensorLayout reshape(std::span<const Dim> new_shape) const {
    if (!isContiguous()) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
          "DenseTensorLayout reshape requires contiguous layout");
    }

    Dim inferred = -1;
    size_type inferred_index = 0;
    Dim known_product = 1;
    bool has_zero = false;

    for (size_type i = 0; i < new_shape.size(); ++i) {
      const Dim dim = new_shape[i];
      if (dim == -1) {
        if (inferred != -1) {
          ::orteaf::internal::diagnostics::error::throwError(
              ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
              "DenseTensorLayout reshape only allows one inferred dimension");
        }
        inferred = dim;
        inferred_index = i;
        continue;
      }
      if (dim < 0) {
        ::orteaf::internal::diagnostics::error::throwError(
            ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
            "DenseTensorLayout reshape dimensions must be non-negative");
      }
      if (dim == 0) {
        has_zero = true;
      } else {
        known_product *= dim;
      }
    }

    const Dim current = numel();
    if (inferred == -1) {
      const Dim expected = has_zero ? Dim{0} : known_product;
      if (current != expected) {
        ::orteaf::internal::diagnostics::error::throwError(
            ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
            "DenseTensorLayout reshape element count mismatch");
      }
    } else {
      if (has_zero) {
        ::orteaf::internal::diagnostics::error::throwError(
            ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
            "DenseTensorLayout reshape cannot infer with zero dimensions");
      }
      if (current == 0) {
        ::orteaf::internal::diagnostics::error::throwError(
            ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
            "DenseTensorLayout reshape cannot infer with zero elements");
      }
      if (known_product == 0 || current % known_product != 0) {
        ::orteaf::internal::diagnostics::error::throwError(
            ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
            "DenseTensorLayout reshape cannot infer dimension");
      }
    }

    Dims resolved_shape;
    resolved_shape.assign(new_shape.begin(), new_shape.end());
    if (inferred != -1) {
      resolved_shape[inferred_index] = current / known_product;
    }

    Dims strides = makeContiguousStrides(resolved_shape);
    return DenseTensorLayout(std::move(resolved_shape), std::move(strides),
                             offset_);
  }

  DenseTensorLayout broadcastTo(std::span<const Dim> new_shape) const {
    validateShape(new_shape);
    if (new_shape.size() < rank()) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
          "DenseTensorLayout broadcast cannot reduce rank");
    }

    Dims new_shape_dims;
    new_shape_dims.assign(new_shape.begin(), new_shape.end());
    Dims new_strides;
    new_strides.resize(new_shape_dims.size());

    size_type old_index = rank();
    for (size_type idx = new_shape_dims.size(); idx > 0; --idx) {
      const size_type new_dim_index = idx - 1;
      const Dim new_dim = new_shape_dims[new_dim_index];
      const Dim old_dim =
          (old_index > 0) ? shape_[old_index - 1] : Dim{1};
      const Dim old_stride =
          (old_index > 0) ? strides_[old_index - 1] : Dim{0};
      if (old_index > 0) {
        --old_index;
      }

      if (old_dim == new_dim) {
        new_strides[new_dim_index] = old_stride;
      } else if (old_dim == 1) {
        new_strides[new_dim_index] = 0;
      } else {
        ::orteaf::internal::diagnostics::error::throwError(
            ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
            "DenseTensorLayout broadcast dimension mismatch");
      }
    }

    return DenseTensorLayout(std::move(new_shape_dims), std::move(new_strides),
                             offset_);
  }

  DenseTensorLayout squeeze() const {
    Dims new_shape;
    Dims new_strides;
    for (size_type i = 0; i < rank(); ++i) {
      if (shape_[i] == 1) {
        continue;
      }
      new_shape.pushBack(shape_[i]);
      new_strides.pushBack(strides_[i]);
    }
    return DenseTensorLayout(std::move(new_shape), std::move(new_strides),
                             offset_);
  }

  DenseTensorLayout squeeze(std::span<const size_type> dims) const {
    Dims new_shape;
    Dims new_strides;
    ::orteaf::internal::base::SmallVector<uint8_t, 4> remove;
    remove.resize(rank());

    for (size_type dim : dims) {
      if (dim >= rank()) {
        ::orteaf::internal::diagnostics::error::throwError(
            ::orteaf::internal::diagnostics::error::OrteafErrc::OutOfRange,
            "DenseTensorLayout squeeze dim out of range");
      }
      if (remove[dim] != 0) {
        ::orteaf::internal::diagnostics::error::throwError(
            ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
            "DenseTensorLayout squeeze dims contain duplicates");
      }
      if (shape_[dim] != 1) {
        ::orteaf::internal::diagnostics::error::throwError(
            ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
            "DenseTensorLayout squeeze dim must be size 1");
      }
      remove[dim] = 1;
    }

    for (size_type i = 0; i < rank(); ++i) {
      if (remove[i] != 0) {
        continue;
      }
      new_shape.pushBack(shape_[i]);
      new_strides.pushBack(strides_[i]);
    }

    return DenseTensorLayout(std::move(new_shape), std::move(new_strides),
                             offset_);
  }

  DenseTensorLayout unsqueeze(size_type dim) const {
    if (dim > rank()) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::OutOfRange,
          "DenseTensorLayout unsqueeze dim out of range");
    }

    Dims new_shape;
    Dims new_strides;
    new_shape.resize(rank() + 1);
    new_strides.resize(rank() + 1);

    size_type old_index = 0;
    for (size_type i = 0; i < new_shape.size(); ++i) {
      if (i == dim) {
        new_shape[i] = 1;
        if (old_index < rank()) {
          new_strides[i] = strides_[old_index] * shape_[old_index];
        } else {
          new_strides[i] = 1;
        }
        continue;
      }
      new_shape[i] = shape_[old_index];
      new_strides[i] = strides_[old_index];
      ++old_index;
    }

    return DenseTensorLayout(std::move(new_shape), std::move(new_strides),
                             offset_);
  }

private:
  static void validateRankMatch(const Dims &shape, const Dims &strides) {
    if (shape.size() != strides.size()) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
          "DenseTensorLayout requires shape and strides to match rank");
    }
  }

  static void validateShape(std::span<const Dim> shape) {
    for (Dim dim : shape) {
      if (dim < 0) {
        ::orteaf::internal::diagnostics::error::throwError(
            ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
            "DenseTensorLayout shape must be non-negative");
      }
    }
  }

  static Dim numelOf(std::span<const Dim> shape) noexcept {
    Dim total = 1;
    for (Dim dim : shape) {
      if (dim == 0) {
        return 0;
      }
      total *= dim;
    }
    return total;
  }

  static Dim numelOf(const Dims &shape) noexcept {
    return numelOf(std::span<const Dim>(shape.data(), shape.size()));
  }

  static Dims makeContiguousStrides(std::span<const Dim> shape) {
    Dims strides;
    strides.resize(shape.size());

    Dim stride = 1;
    for (size_type idx = shape.size(); idx > 0; --idx) {
      strides[idx - 1] = stride;
      stride *= shape[idx - 1];
    }
    return strides;
  }

  static Dims makeContiguousStrides(const Dims &shape) {
    return makeContiguousStrides(
        std::span<const Dim>(shape.data(), shape.size()));
  }

  Dims shape_{};
  Dims strides_{};
  Dim offset_{0};
};

} // namespace orteaf::extension::tensor
