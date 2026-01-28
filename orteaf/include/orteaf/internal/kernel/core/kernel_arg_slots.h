#pragma once

#include <type_traits>
#include <utility>

#include <orteaf/internal/kernel/core/kernel_args.h>
#include <orteaf/internal/kernel/param/param.h>
#include <orteaf/internal/kernel/param/param_transform.h>
#include <orteaf/internal/kernel/storage/operand_key.h>
#include <orteaf/internal/kernel/storage/role.h>
#include <orteaf/internal/kernel/storage/operand_id.h>
#include <orteaf/internal/storage/storage_lease.h>

namespace orteaf::internal::kernel {

/**
 * @brief Typed parameter slot with an optional role.
 *
 * The operand ID is provided later when binding into KernelArgs.
 * This keeps TensorImpl/Layout-side definitions stable even as the
 * operand layout evolves.
 */
template <typename T, ParamId Id, Role RoleValue = Role::Data>
struct ParamSlot {
  using Target =
      ::orteaf::generated::param_id_tables::ParamValueTypeT<Id>;
  static_assert(std::is_constructible_v<Param, ParamKey, Target>,
                "ParamSlot target type must be supported by Param");
  static constexpr ParamId kId = Id;
  static constexpr Role kRole = RoleValue;

  T value_{};

  constexpr ParamSlot() = default;
  constexpr explicit ParamSlot(T value) : value_(std::move(value)) {}

  static constexpr ParamKey globalKey() noexcept {
    return ParamKey::global(Id);
  }

  static constexpr ParamKey scopedKey(OperandId operand_id) noexcept {
    return ParamKey::scoped(Id, makeOperandKey(operand_id, RoleValue));
  }

  void bindGlobal(KernelArgs &args) const {
    args.addParam(Param(globalKey(), Transform<T, Target>(value_)));
  }

  void bindScoped(KernelArgs &args, OperandId operand_id) const {
    args.addParam(
        Param(scopedKey(operand_id), Transform<T, Target>(value_)));
  }

  Param toGlobalParam() const {
    return Param(globalKey(), Transform<T, Target>(value_));
  }

  Param toScopedParam(OperandId operand_id) const {
    return Param(scopedKey(operand_id), Transform<T, Target>(value_));
  }
};

/**
 * @brief Storage slot with a fixed role.
 *
 * The operand ID is provided later when binding into KernelArgs.
 */
template <Role RoleValue = Role::Data>
struct StorageSlot {
  static constexpr Role kRole = RoleValue;

  ::orteaf::internal::storage::StorageLease storage_{};

  StorageSlot() = default;
  explicit StorageSlot(::orteaf::internal::storage::StorageLease storage)
      : storage_(std::move(storage)) {}

  const ::orteaf::internal::storage::StorageLease &lease() const {
    return storage_;
  }
  ::orteaf::internal::storage::StorageLease &lease() { return storage_; }

  void bind(KernelArgs &args, OperandId operand_id) const {
    args.addStorage(makeOperandKey(operand_id, RoleValue), storage_);
  }

  void bind(KernelArgs &args, OperandId operand_id) {
    args.addStorage(makeOperandKey(operand_id, RoleValue), std::move(storage_));
  }
};

} // namespace orteaf::internal::kernel
