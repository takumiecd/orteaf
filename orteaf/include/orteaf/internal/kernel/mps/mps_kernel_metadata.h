#pragma once

#if ORTEAF_ENABLE_MPS

#include <utility>

#include "orteaf/internal/base/heap_vector.h"
#include "orteaf/internal/execution/mps/manager/mps_compute_pipeline_state_manager.h"
#include "orteaf/internal/kernel/mps/mps_kernel_args.h"
#include "orteaf/internal/kernel/mps/mps_kernel_base.h"
#include "orteaf/internal/kernel/mps/mps_kernel_entry.h"
#include "orteaf/internal/kernel/registry/kernel_entry_traits.h"

namespace orteaf::internal::kernel::registry {

/**
 * @brief Lightweight kernel metadata for Secondary Storage tier.
 *
 * Contains only the minimal information needed to reconstruct
 * an MpsKernelEntry after its PipelineLeases have been evicted.
 */
struct MpsKernelMetadata {
  using LibraryKey = ::orteaf::internal::execution::mps::manager::LibraryKey;
  using FunctionKey = ::orteaf::internal::execution::mps::manager::FunctionKey;
  using Key = std::pair<LibraryKey, FunctionKey>;
  using ExecuteFunc =
      ::orteaf::internal::kernel::mps::MpsKernelEntry::ExecuteFunc;

  /// Library/function pairs for kernel reconstruction
  ::orteaf::internal::base::HeapVector<Key> keys;

  /// Execution function pointer (stateless, safe to store)
  ExecuteFunc execute{nullptr};

  /**
   * @brief Reconstruct an MpsKernelEntry from this metadata.
   *
   * Creates a new entry with the stored library/function keys and
   * execute function. The returned entry will need to be configured
   * for a device before use.
   *
   * @return Reconstructed MpsKernelEntry (unconfigured)
   */
  [[nodiscard]] ::orteaf::internal::kernel::mps::MpsKernelEntry
  rebuild() const {
    ::orteaf::internal::kernel::mps::MpsKernelEntry entry;
    entry.base.reserveKeys(keys.size());
    for (const auto &key : keys) {
      entry.base.addKey(key.first.identifier.c_str(),
                        key.second.identifier.c_str());
    }
    entry.execute = execute;
    return entry;
  }
};

/**
 * @brief Traits for MPS kernel entries.
 *
 * Satisfies KernelEntryTraitsConcept for use with KernelRegistry.
 */
struct MpsKernelEntryTraits {
  using Entry = ::orteaf::internal::kernel::mps::MpsKernelEntry;
  using Metadata = MpsKernelMetadata;

  /**
   * @brief Create metadata from an existing MpsKernelEntry.
   *
   * Extracts the library/function keys and execute function.
   * Used when demoting an entry from Main Memory to Secondary Storage.
   *
   * @param entry Entry to extract metadata from
   * @return Metadata containing reconstruction information
   */
  static Metadata toMetadata(const Entry &entry) {
    Metadata metadata;
    metadata.execute = entry.execute;

    const auto &keys = entry.base.keys();
    metadata.keys.reserve(keys.size());
    for (const auto &key : keys) {
      metadata.keys.pushBack(key);
    }

    return metadata;
  }
};

// Static assert to verify traits satisfy the concept
static_assert(KernelEntryTraitsConcept<MpsKernelEntryTraits>,
              "MpsKernelEntryTraits must satisfy KernelEntryTraitsConcept");

} // namespace orteaf::internal::kernel::registry

#endif // ORTEAF_ENABLE_MPS
