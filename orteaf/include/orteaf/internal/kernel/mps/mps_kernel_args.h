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
 * @tparam MaxBindings Maximum number of buffer bindings.
 * @tparam Params User-defined POD parameter struct.
 *
 * Host: Non-POD, manages StorageLease lifetime and Context.
 * Device: POD, contains buffer views and params for kernel execution.
 */
template <std::size_t MaxBindings, typename Params> class MpsKernelArgs {
public:
  using BufferView =
      ::orteaf::internal::execution::mps::resource::MpsBufferView;
  using StorageLayout = ::orteaf::internal::storage::mps::MpsStorageLayout;
  using StorageLease = ::orteaf::internal::storage::MpsStorageLease;
  using Context = ::orteaf::internal::execution_context::mps::Context;

  // ---- Device側（POD、Kernelに渡す） ----
  struct Device {
    struct Binding {
      BufferView view{};
      StorageLayout layout{};
      Access access{Access::None};
    };

    std::array<Binding, MaxBindings> bindings{};
    std::size_t binding_count{0};
    Params params{};
  };

  // ---- Host側（non-POD、ライフタイム管理） ----
  class Host {
  public:
    struct StorageEntry {
      StorageLease lease;
      Access access;
    };

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

    void addStorageLease(StorageLease lease, Access access) {
      if (count_ < MaxBindings) {
        storages_[count_] = StorageEntry{std::move(lease), access};
        ++count_;
      }
    }

    const Context &context() const { return context_; }
    Context &context() { return context_; }

    std::size_t storageCount() const { return count_; }

    const StorageEntry &storageAt(std::size_t index) const {
      return storages_[index];
    }

  private:
    Context context_{};
    std::array<StorageEntry, MaxBindings> storages_{};
    std::size_t count_{0};
  };

  // Host から Device への変換
  static Device toDevice(const Host &host, Params params) {
    Device device{};
    device.binding_count = host.storageCount();
    device.params = std::move(params);
    // TODO: StorageLease から BufferView/Layout を抽出して bindings に設定
    for (std::size_t i = 0; i < host.storageCount(); ++i) {
      const auto &entry = host.storageAt(i);
      device.bindings[i].access = entry.access;
    }
    return device;
  }
};

} // namespace orteaf::internal::kernel::mps

#endif // ORTEAF_ENABLE_MPS
