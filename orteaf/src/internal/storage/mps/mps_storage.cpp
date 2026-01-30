#include "orteaf/internal/storage/mps/mps_storage.h"

#include "orteaf/internal/execution/mps/api/mps_execution_api.h"

namespace orteaf::internal::storage::mps {

MpsStorage MpsStorage::Builder::build() {
  if (!heap_lease_) {
    if (heap_handle_.isValid()) {
      heap_lease_ = ::orteaf::internal::execution::mps::api::MpsExecutionApi::
          acquireHeap(heap_handle_);
    } else if (has_heap_key_) {
      heap_lease_ = ::orteaf::internal::execution::mps::api::MpsExecutionApi::
          acquireHeap(heap_key_);
    }
  }
  if (!heap_lease_) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "MpsStorage requires a valid heap lease");
  }
  const std::size_t size_in_bytes = numel_ * ::orteaf::internal::sizeOf(dtype_);
  BufferLease lease = heap_lease_->acquireBuffer(size_in_bytes, alignment_);
  return MpsStorage(std::move(lease), std::move(fence_token_),
                    std::move(layout_), dtype_, numel_);
}

} // namespace orteaf::internal::storage::mps
