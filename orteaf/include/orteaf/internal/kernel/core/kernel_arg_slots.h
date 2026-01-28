#pragma once

#include <type_traits>
#include <utility>

#include <orteaf/internal/kernel/core/kernel_args.h>
#include <orteaf/internal/kernel/param/param.h>
#include <orteaf/internal/kernel/storage/storage_key.h>
#include <orteaf/internal/kernel/storage/storage_role.h>
#include <orteaf/internal/kernel/storage/storage_id.h>
#include <orteaf/internal/storage/storage_lease.h>

namespace orteaf::internal::kernel {

/**
 * @brief Typed parameter slot with an optional storage role.
 *
 * The storage ID is provided later when binding into KernelArgs.
 * This keeps TensorImpl/Layout-side definitions stable even as the
 * storage layout evolves.
 */
template <typename T, ParamId Id, StorageRole Role = StorageRole::Data>
struct ParamSlot {
  static_assert(std::is_constructible_v<Param, ParamKey, T>,
                "ParamSlot value type must be supported by Param");
  static constexpr ParamId kId = Id;
  static constexpr StorageRole kRole = Role;

  T value_{};

  constexpr ParamSlot() = default;
  constexpr explicit ParamSlot(T value) : value_(std::move(value)) {}

  static constexpr ParamKey globalKey() noexcept {
    return ParamKey::global(Id);
  }

  static constexpr ParamKey scopedKey(StorageId storage_id) noexcept {
    return ParamKey::scoped(Id, makeStorageKey(storage_id, Role));
  }

  void bindGlobal(KernelArgs &args) const {
    args.addParam(Param(globalKey(), value_));
  }

  void bindScoped(KernelArgs &args, StorageId storage_id) const {
    args.addParam(Param(scopedKey(storage_id), value_));
  }

  Param toGlobalParam() const { return Param(globalKey(), value_); }

  Param toScopedParam(StorageId storage_id) const {
    return Param(scopedKey(storage_id), value_);
  }
};

/**
 * @brief Storage slot with a fixed role.
 *
 * The storage ID is provided later when binding into KernelArgs.
 */
template <StorageRole Role = StorageRole::Data>
struct StorageSlot {
  static constexpr StorageRole kRole = Role;

  ::orteaf::internal::storage::StorageLease storage_{};

  StorageSlot() = default;
  explicit StorageSlot(::orteaf::internal::storage::StorageLease storage)
      : storage_(std::move(storage)) {}

  const ::orteaf::internal::storage::StorageLease &lease() const {
    return storage_;
  }
  ::orteaf::internal::storage::StorageLease &lease() { return storage_; }

  void bind(KernelArgs &args, StorageId storage_id) const {
    args.addStorage(makeStorageKey(storage_id, Role), storage_);
  }

  void bind(KernelArgs &args, StorageId storage_id) {
    args.addStorage(makeStorageKey(storage_id, Role), std::move(storage_));
  }
};

} // namespace orteaf::internal::kernel
