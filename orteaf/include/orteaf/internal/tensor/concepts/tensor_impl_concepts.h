#pragma once

/**
 * @file tensor_impl_concepts.h
 * @brief Concepts for TensorImpl feature detection.
 *
 * These concepts are used to detect which view operations are
 * supported by a TensorImpl at compile time.
 */

#include <concepts>
#include <cstddef>
#include <span>
#include <type_traits>

namespace orteaf::internal::tensor {

// =============================================================================
// TensorImpl Concepts
// =============================================================================

/// @brief Concept for basic TensorImpl requirements
template <typename Impl>
concept TensorImplConcept = requires {
  typename Impl::Layout;
  typename Impl::StorageLease;
} && requires(const Impl &impl) {
  { impl.layout() } -> std::convertible_to<const typename Impl::Layout &>;
  {
    impl.storageLease()
  } -> std::convertible_to<const typename Impl::StorageLease &>;
  { impl.valid() } -> std::convertible_to<bool>;
};

/// @brief Concept for TensorImpl that supports transpose
template <typename Impl>
concept HasTranspose =
    TensorImplConcept<Impl> && requires(const typename Impl::Layout &layout,
                                        std::span<const std::size_t> perm) {
      { layout.transpose(perm) } -> std::same_as<typename Impl::Layout>;
    };

/// @brief Concept for TensorImpl that supports slice
template <typename Impl>
concept HasSlice = TensorImplConcept<Impl> &&
                   requires(const typename Impl::Layout &layout,
                            std::span<const typename Impl::Layout::Dim> starts,
                            std::span<const typename Impl::Layout::Dim> sizes) {
                     {
                       layout.slice(starts, sizes)
                     } -> std::same_as<typename Impl::Layout>;
                   };

/// @brief Concept for TensorImpl that supports reshape
template <typename Impl>
concept HasReshape =
    TensorImplConcept<Impl> &&
    requires(const typename Impl::Layout &layout,
             std::span<const typename Impl::Layout::Dim> new_shape) {
      { layout.reshape(new_shape) } -> std::same_as<typename Impl::Layout>;
    };

/// @brief Concept for TensorImpl that supports squeeze
template <typename Impl>
concept HasSqueeze =
    TensorImplConcept<Impl> && requires(const typename Impl::Layout &layout) {
      { layout.squeeze() } -> std::same_as<typename Impl::Layout>;
    };

/// @brief Concept for TensorImpl that supports unsqueeze
template <typename Impl>
concept HasUnsqueeze =
    TensorImplConcept<Impl> &&
    requires(const typename Impl::Layout &layout, std::size_t dim) {
      { layout.unsqueeze(dim) } -> std::same_as<typename Impl::Layout>;
    };

} // namespace orteaf::internal::tensor
