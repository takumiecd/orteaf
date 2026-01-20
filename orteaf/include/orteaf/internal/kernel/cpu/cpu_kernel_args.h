#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include <orteaf/internal/execution/cpu/resource/cpu_buffer_view.h>
#include <orteaf/internal/execution_context/cpu/context.h>
#include <orteaf/internal/execution_context/cpu/current_context.h>
#include <orteaf/internal/kernel/access.h>
#include <orteaf/internal/storage/cpu/cpu_storage_layout.h>
#include <orteaf/internal/storage/registry/storage_types.h>

namespace orteaf::internal::kernel::cpu {

/**
 * @brief CPU kernel arguments container with Host and Device nested types.
 *
 * @tparam MaxBindings Maximum number of storage bindings.
 * @tparam Params User-defined POD parameter struct.
 *
 * Common: Manages StorageLease lifetime and access metadata.
 * Host: Non-POD, manages Context.
 * Device: Kernel-visible arguments container (views/layouts + params).
 */
template <std::size_t MaxBindings, typename Params> class CpuKernelArgs {
public:
  using BufferView =
      ::orteaf::internal::execution::cpu::resource::CpuBufferView;
  using StorageLayout = ::orteaf::internal::storage::cpu::CpuStorageLayout;
  using StorageLease = ::orteaf::internal::storage::CpuStorageLease;
  using Context = ::orteaf::internal::execution_context::cpu::Context;

  class Common {
  public:
    void addStorageLease(StorageLease lease, Access access) {
      if (storage_count_ < MaxBindings) {
        storage_leases_[storage_count_] = std::move(lease);
        storage_accesses_[storage_count_] = access;
        ++storage_count_;
      }
    }

    std::size_t storageCount() const { return storage_count_; }

    const StorageLease &storageLeaseAt(std::size_t index) const {
      return storage_leases_[index];
    }

    Access storageAccessAt(std::size_t index) const {
      return storage_accesses_[index];
    }

  protected:
    std::array<StorageLease, MaxBindings> storage_leases_{};
    std::array<Access, MaxBindings> storage_accesses_{};
    std::size_t storage_count_{0};
  };

  // ---- Device側（Kernelに渡す） ----
  struct Device {
    Params params{};
  };

  // ---- Host側（non-POD、ライフタイム管理） ----
  class Host {
  public:
    Host() = default;
    explicit Host(Context ctx) : context_(std::move(ctx)) {}

    /// @brief Create Host from current thread-local context.
    static Host fromCurrentContext() {
      return Host(::orteaf::internal::execution_context::cpu::currentContext());
    }

    Host(const Host &) = default;
    Host &operator=(const Host &) = default;
    Host(Host &&) = default;
    Host &operator=(Host &&) = default;
    ~Host() = default;

    const Context &context() const { return context_; }
    Context &context() { return context_; }

  private:
    Context context_{};
  };

private:
  Common common_{};
  Device device_{};
  Host host_{};
};

} // namespace orteaf::internal::kernel::cpu
