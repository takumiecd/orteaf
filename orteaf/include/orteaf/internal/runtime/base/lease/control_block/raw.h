#pragma once

#include <orteaf/internal/runtime/base/lease/category.h>
#include <orteaf/internal/runtime/base/lease/concepts.h>

namespace orteaf::internal::base {

/// @brief Raw control block - no reference counting, only payload
/// @details Used for resources that don't need lifecycle management.
/// All operations are no-op. This control block is optimized away in most
/// cases due to empty base optimization.
template <typename PayloadT>
  requires PayloadConcept<PayloadT>
struct RawControlBlock {
  using Category = lease_category::Raw;
  using Payload = PayloadT;

  PayloadT payload{};

  /// @brief Always succeeds (no actual acquire needed)
  constexpr bool tryAcquire() noexcept { return true; }

  /// @brief No-op release
  constexpr void release() noexcept {}

  /// @brief Always alive (no tracking)
  constexpr bool isAlive() const noexcept { return true; }
};

// Verify concept satisfaction
static_assert(ControlBlockConcept<RawControlBlock<int>>);

} // namespace orteaf::internal::base
