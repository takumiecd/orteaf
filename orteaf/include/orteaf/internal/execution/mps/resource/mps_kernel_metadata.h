#pragma once

#if ORTEAF_ENABLE_MPS

#include <utility>

#include "orteaf/internal/base/heap_vector.h"
#include "orteaf/internal/execution/mps/manager/mps_compute_pipeline_state_manager.h"
#include "orteaf/internal/execution/mps/manager/mps_library_manager.h"
#include "orteaf/internal/kernel/kernel_entry.h"

namespace orteaf::internal::execution::mps::resource {

/**
 * @brief Kernel metadata resource for MPS.
 *
 * Stores library/function keys and execute function for kernel reconstruction.
 */
struct MpsKernelMetadata {
  using LibraryKey = ::orteaf::internal::execution::mps::manager::LibraryKey;
  using FunctionKey = ::orteaf::internal::execution::mps::manager::FunctionKey;
  using Key = std::pair<LibraryKey, FunctionKey>;
  using ExecuteFunc = ::orteaf::internal::kernel::KernelEntry::ExecuteFunc;

  bool initialize(const ::orteaf::internal::base::HeapVector<Key> &keys,
                  ExecuteFunc execute) {
    reset();
    execute_ = execute;
    keys_.reserve(keys.size());
    for (const auto &key : keys) {
      keys_.pushBack(key);
    }
    return true;
  }

  void reset() noexcept {
    keys_.clear();
    execute_ = nullptr;
  }

  const ::orteaf::internal::base::HeapVector<Key> &keys() const noexcept {
    return keys_;
  }

  ExecuteFunc execute() const noexcept { return execute_; }

private:
  ::orteaf::internal::base::HeapVector<Key> keys_{};
  ExecuteFunc execute_{nullptr};
};

} // namespace orteaf::internal::execution::mps::resource

#endif // ORTEAF_ENABLE_MPS
