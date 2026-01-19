#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include <orteaf/internal/execution/cpu/resource/cpu_buffer_view.h>
#include <orteaf/internal/execution_context/cpu/context.h>
#include <orteaf/internal/execution_context/cpu/current_context.h>
#include <orteaf/internal/kernel/access.h>
#include <orteaf/internal/storage/cpu/cpu_storage.h>
#include <orteaf/internal/storage/cpu/cpu_storage_layout.h>

namespace orteaf::internal::kernel::cpu {

/**
 * @brief CPU kernel arguments container with Host and Device nested types.
 *
 * @tparam MaxBindings Maximum number of buffer bindings.
 * @tparam Params User-defined POD parameter struct.
 *
 * Host: Non-POD, manages Storage lifetime and Context.
 * Device: POD, contains buffer views and params for kernel execution.
 */
template <std::size_t MaxBindings, typename Params> class CpuKernelArgs {
public:
  using BufferView =
      ::orteaf::internal::execution::cpu::resource::CpuBufferView;
  using StorageLayout = ::orteaf::internal::storage::cpu::CpuStorageLayout;
  using Storage = ::orteaf::internal::storage::cpu::CpuStorage;
  using Context = ::orteaf::internal::execution_context::cpu::Context;

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
      Storage storage;
      Access access;
    };

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

    void addStorage(Storage storage, Access access) {
      if (count_ < MaxBindings) {
        storages_[count_] = StorageEntry{std::move(storage), access};
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
  // 注: Storage から BufferView への変換は、Storage クラスに
  //     bufferView() メソッドが必要になる想定。現時点ではプレースホルダー。
  static Device toDevice(const Host &host, Params params) {
    Device device{};
    device.binding_count = host.storageCount();
    device.params = std::move(params);
    // TODO: Storage から BufferView/Layout を抽出して bindings に設定
    for (std::size_t i = 0; i < host.storageCount(); ++i) {
      const auto &entry = host.storageAt(i);
      device.bindings[i].access = entry.access;
      // device.bindings[i].view = entry.storage.bufferView(); // 要実装
      // device.bindings[i].layout = entry.storage.layout();   // 要実装
    }
    return device;
  }
};

} // namespace orteaf::internal::kernel::cpu
