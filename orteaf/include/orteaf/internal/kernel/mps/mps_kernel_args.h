#pragma once

#if ORTEAF_ENABLE_MPS

#include <array>
#include <cstddef>
#include <utility>

#include <orteaf/internal/execution/mps/resource/mps_buffer_view.h>
#include <orteaf/internal/execution_context/mps/context.h>
#include <orteaf/internal/execution_context/mps/current_context.h>
#include <orteaf/internal/kernel/access.h>
#include <orteaf/internal/storage/mps/mps_storage_layout.h>
#include <orteaf/internal/storage/registry/storage_types.h>

namespace orteaf::internal::kernel::mps {

/**
 * @brief MPS kernel arguments container with Host and Device nested types.
 *
 * @tparam MaxBindings Maximum number of storage bindings.
 * @tparam Params User-defined POD parameter struct.
 *
 * Common: Manages StorageLease lifetime and access metadata.
 * Host: Non-POD, manages Context.
 * Device: Kernel-visible arguments container (views/layouts + params).
 */
template <std::size_t MaxBindings, typename Params> class MpsKernelArgs {
public:
  using BufferView =
      ::orteaf::internal::execution::mps::resource::MpsBufferView;
  using StorageLayout = ::orteaf::internal::storage::mps::MpsStorageLayout;
  using StorageLease = ::orteaf::internal::storage::MpsStorageLease;
  using Context = ::orteaf::internal::execution_context::mps::Context;

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
      return Host(::orteaf::internal::execution_context::mps::currentContext());
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

} // namespace orteaf::internal::kernel::mps

#endif // ORTEAF_ENABLE_MPS
