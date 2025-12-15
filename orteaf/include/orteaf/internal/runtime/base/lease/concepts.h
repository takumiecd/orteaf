#pragma once

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace orteaf::internal::runtime::base {

// =============================================================================
// Base ControlBlock Concept
// =============================================================================

/// @brief Base concept for all control blocks
/// @details Minimal interface - essential lifecycle operations:
/// - acquire: take ownership, returns bool (true if succeeded)
/// - release: release ownership, returns bool (true if fully released)
/// - isAlive: check if currently owned/active
/// - validate: mark resource as initialized/valid
/// - invalidate: mark resource as uninitialized/invalid
template <typename CB>
concept ControlBlockConcept = requires(CB cb, const CB ccb) {
  typename CB::Category;
  typename CB::Slot;
  { cb.acquire() } -> std::same_as<bool>;
  { cb.release() } -> std::same_as<bool>;
  { ccb.isAlive() } -> std::same_as<bool>;
  { cb.validate() };
  { cb.invalidate() };
};

// =============================================================================
// Specialized ControlBlock Concepts
// =============================================================================

/// @brief Concept for shared control blocks (reference counted)
template <typename CB>
concept SharedControlBlockConcept =
    ControlBlockConcept<CB> && requires(CB cb, const CB ccb) {
      { cb.acquire() };
      { ccb.count() } -> std::convertible_to<std::size_t>;
    };

/// @brief Concept for control blocks supporting weak references
template <typename CB>
concept WeakableControlBlockConcept =
    ControlBlockConcept<CB> && requires(CB cb) {
      { cb.acquireWeak() };
      { cb.releaseWeak() } -> std::same_as<bool>;
    };

/// @brief Concept for weakable control blocks that can promote weak to strong
template <typename CB>
concept PromotableControlBlockConcept =
    WeakableControlBlockConcept<CB> && requires(CB cb) {
      { cb.tryPromote() } -> std::same_as<bool>;
    };

// =============================================================================
// Lease/ControlBlock Compatibility Concept
// =============================================================================

/// @brief Concept to check if a Lease type is compatible with a ControlBlock
/// type
/// @details Checks if CompatibleCategory of Lease matches Category of
/// ControlBlock
template <typename LeaseT, typename ControlBlockT>
concept CompatibleLeaseControlBlock =
    requires {
      typename LeaseT::CompatibleCategory;
      typename ControlBlockT::Category;
    } && std::same_as<typename LeaseT::CompatibleCategory,
                      typename ControlBlockT::Category>;

// =============================================================================
// Payload Concept
// =============================================================================

/// @brief Concept for payload types that can be stored in slots
template <typename P>
concept PayloadConcept =
    std::is_default_constructible_v<P> && std::is_move_constructible_v<P> &&
    std::is_move_assignable_v<P>;

} // namespace orteaf::internal::runtime::base
