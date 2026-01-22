#pragma once

#include <array>
#include <cstddef>
#include <cstring>
#include <utility>

#include <orteaf/internal/base/small_vector.h>
#include <orteaf/internal/kernel/param.h>

#include <orteaf/internal/kernel/access.h>
#include <orteaf/internal/kernel/kernel_key.h>
#include <orteaf/internal/storage/registry/storage_types.h>

namespace orteaf::internal::kernel::cpu {

/**
 * @brief CPU kernel arguments container.
 */
class CpuKernelArgs {
public:
  using StorageLease = ::orteaf::internal::storage::CpuStorageLease;

  static constexpr std::size_t kMaxBindings = 16;
  static constexpr std::size_t kMaxParams = 16;

  void addStorageLease(StorageLease lease, Access access) {
    if (storage_count_ >= kMaxBindings) {
      return;
    }
    storage_leases_[storage_count_] = std::move(lease);
    storage_accesses_[storage_count_] = access;
    ++storage_count_;
  }

  std::size_t storageCount() const { return storage_count_; }

  std::size_t storageCapacity() const { return kMaxBindings; }

  const StorageLease &storageLeaseAt(std::size_t index) const {
    return storage_leases_[index];
  }

  Access storageAccessAt(std::size_t index) const {
    return storage_accesses_[index];
  }

  void clearStorages() {
    for (std::size_t i = 0; i < storage_count_; ++i) {
      storage_leases_[i] = StorageLease{};
      storage_accesses_[i] = Access::None;
    }
    storage_count_ = 0;
  }

  void addParam(Param param) { params.pushBack(std::move(param)); }

  const Param *findParam(ParamId id) const {
    for (const auto &p : params) {
      if (p.id() == id) {
        return &p;
      }
    }
    return nullptr;
  }

  Param *findParam(ParamId id) {
    for (auto &p : params) {
      if (p.id() == id) {
        return &p;
      }
    }
    return nullptr;
  }

  const auto &paramList() const { return params; }

  void clearParams() { params.clear(); }

private:
  std::array<StorageLease, kMaxBindings> storage_leases_{};
  std::array<Access, kMaxBindings> storage_accesses_{};
  std::size_t storage_count_{0};
  ::orteaf::internal::base::SmallVector<Param, kMaxParams> params{};
};

} // namespace orteaf::internal::kernel::cpu
