#pragma once

#include <string_view>

struct MPSString_st; using MPSString_t = MPSString_st*;

static_assert(sizeof(MPSString_t) == sizeof(void*), "MPSString must be pointer-sized.");

namespace orteaf::internal::backend::mps {

/// Convert std::string_view to NSString*.
/// Tries UTF-8 encoding first, falls back to ISO Latin-1 if UTF-8 fails.
[[nodiscard]] MPSString_t to_ns_string(std::string_view view);

} // namespace orteaf::internal::backend::mps
