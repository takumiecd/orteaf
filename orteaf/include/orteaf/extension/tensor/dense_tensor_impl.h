#pragma once

/**
 * @file dense_tensor_impl.h
 * @brief Dense tensor implementation holding layout and storage.
 *
 * DenseTensorImpl combines a DenseTensorLayout (shape, strides, offset)
 * with a StorageLease (type-erased backend storage lease) to represent
 * dense tensor data.
 */

#include <cstddef>
#include <utility>

#include <orteaf/extension/tensor/layout/dense_tensor_layout.h>
#include <orteaf/internal/dtype/dtype.h>
#include <orteaf/internal/execution/execution.h>
#include <orteaf/internal/kernel/core/kernel_arg_slots.h>
#include <orteaf/internal/kernel/storage/operand_id.h>
#include <orteaf/internal/storage/storage_lease.h>

namespace orteaf::extension::tensor {

/**
 * @brief Dense tensor implementation.
 *
 * Holds a DenseTensorLayout describing the logical view (shape, strides,
 * offset) and a StorageLease providing access to the underlying data buffer.
 *
 * Multiple DenseTensorImpl instances can share the same storage (for views).
 * The layout's numel() represents the logical element count, while
 * storage's numel() represents the physical buffer capacity.
 *
 * Invariant: layout_.numel() <= storage_.lease().numel() (for valid views)
 */
class DenseTensorImpl {
public:
  using Layout = DenseTensorLayout;
  using Dims = Layout::Dims;
  using Dim = Layout::Dim;
  using StorageLease = ::orteaf::internal::storage::StorageLease;
  using StorageSlot = ::orteaf::internal::kernel::StorageSlot<
      ::orteaf::internal::kernel::Role::Data>;
  using DType = ::orteaf::internal::DType;
  using Execution = ::orteaf::internal::execution::Execution;

  /**
   * @brief Default constructor. Creates an uninitialized impl.
   */
  DenseTensorImpl() = default;

  /**
   * @brief Construct from layout and storage lease.
   *
   * @param layout The tensor layout (shape, strides, offset).
   * @param storage The storage lease holding the data buffer.
   */
  DenseTensorImpl(Layout layout, StorageLease storage)
      : layout_(std::move(layout)), storage_(StorageSlot(std::move(storage))) {}

  DenseTensorImpl(Layout layout, StorageSlot storage)
      : layout_(std::move(layout)), storage_(std::move(storage)) {}

  DenseTensorImpl(const DenseTensorImpl &) = default;
  DenseTensorImpl &operator=(const DenseTensorImpl &) = default;
  DenseTensorImpl(DenseTensorImpl &&) = default;
  DenseTensorImpl &operator=(DenseTensorImpl &&) = default;
  ~DenseTensorImpl() = default;

  // ===== Accessors =====

  /// @brief Return the tensor layout.
  const Layout &layout() const noexcept { return layout_; }

  /// @brief Return the storage lease.
  const StorageLease &storageLease() const noexcept { return storage_.lease(); }

  /// @brief Return the storage slot.
  const StorageSlot &storageSlot() const noexcept { return storage_; }
  StorageSlot &storageSlot() noexcept { return storage_; }

  /// @brief Check if this impl is valid (has storage).
  bool valid() const noexcept { return static_cast<bool>(storage_.lease()); }

  // ===== Forwarding from StorageLease =====

  /// @brief Return the data type.
  DType dtype() const { return storage_.lease().dtype(); }

  /// @brief Return the execution backend.
  Execution execution() const { return storage_.lease().execution(); }

  /// @brief Return the storage size in bytes.
  std::size_t storageSizeInBytes() const { return storage_.lease().sizeInBytes(); }

  // ===== Forwarding from Layout =====

  /// @brief Return the tensor shape.
  const Dims &shape() const noexcept { return layout_.shape(); }

  /// @brief Return the tensor strides.
  const Dims &strides() const noexcept { return layout_.strides(); }

  /// @brief Return the element offset.
  Dim offset() const noexcept { return layout_.offset(); }

  /// @brief Return the number of elements (logical, based on shape).
  Dim numel() const noexcept { return layout_.numel(); }

  /// @brief Return the rank (number of dimensions).
  std::size_t rank() const noexcept { return layout_.rank(); }

  /// @brief Check if the layout is contiguous.
  bool isContiguous() const noexcept { return layout_.isContiguous(); }

  void bindAllArgs(::orteaf::internal::kernel::KernelArgs &args,
                   ::orteaf::internal::kernel::OperandId operand_id) const {
    storage_.bind(args, operand_id);
    layout_.bindParams(args, operand_id);
  }

private:
  Layout layout_{};
  StorageSlot storage_{};
};

} // namespace orteaf::extension::tensor
