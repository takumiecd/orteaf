#pragma once

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace orteaf::internal::base {

// =============================================================================
// Base ControlBlock Concept
// =============================================================================

/// @brief Base concept for all control blocks
/// @details Minimal interface required by all control block types:
/// - canShutdown: check if the control block can be safely shutdown
/// - canTeardown: check if teardown is allowed (no strong refs blocking)
/// - canBindPayload: check if payload binding is allowed
/// - tryBindPayload: attempt to bind payload metadata
/// - payloadPtr: get pointer to payload
/// - payloadHandle: get handle to payload
template <typename CB>
concept BaseControlBlockConcept = requires(CB cb, const CB ccb) {
  typename CB::Category;
  typename CB::Handle;
  typename CB::Payload;
  typename CB::Pool;
  { ccb.canShutdown() } -> std::same_as<bool>;
  { ccb.canTeardown() } -> std::same_as<bool>;
  { ccb.canBindPayload() } -> std::same_as<bool>;
  // Note: tryBindPayload is a template so can't be fully checked here
  { ccb.payloadHandle() };
  { cb.payloadPtr() };
};

// =============================================================================
// Strong ControlBlock Concept
// =============================================================================

/// @brief Concept for strong-only control blocks
/// @details Control blocks that manage strong ownership with reference
/// counting:
/// - acquireStrong: increment strong reference count
/// - releaseStrong: decrement strong reference count, returns true on 0
/// transition
/// - strongCount: get current strong reference count
template <typename CB>
concept StrongControlBlockConcept =
    BaseControlBlockConcept<CB> && requires(CB cb, const CB ccb) {
      { cb.acquireStrong() };
      { cb.releaseStrong() } -> std::same_as<bool>;
      { ccb.strongCount() } -> std::convertible_to<std::size_t>;
    };

// =============================================================================
// Weak ControlBlock Concept
// =============================================================================

/// @brief Concept for weak-only control blocks
/// @details Control blocks that manage only weak references:
/// - acquireWeak: increment weak reference count
/// - releaseWeak: decrement weak reference count, returns true on 0 transition
/// - weakCount: get current weak reference count
template <typename CB>
concept WeakControlBlockConcept =
    BaseControlBlockConcept<CB> && requires(CB cb, const CB ccb) {
      { cb.acquireWeak() };
      { cb.releaseWeak() } -> std::same_as<bool>;
      { ccb.weakCount() } -> std::convertible_to<std::size_t>;
    };

// =============================================================================
// Shared ControlBlock Concept
// =============================================================================

/// @brief Concept for shared control blocks (both strong and weak references)
/// @details Control blocks that support both strong and weak references,
/// with the ability to promote weak references to strong:
/// - All StrongControlBlockConcept requirements
/// - All WeakControlBlockConcept requirements
/// - tryPromoteWeakToStrong: attempt to promote weak to strong reference
template <typename CB>
concept SharedControlBlockConcept =
    StrongControlBlockConcept<CB> && WeakControlBlockConcept<CB> &&
    requires(CB cb) {
      { cb.tryPromoteWeakToStrong() } -> std::same_as<bool>;
    };

} // namespace orteaf::internal::base
