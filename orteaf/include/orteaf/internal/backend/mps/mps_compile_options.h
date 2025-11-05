#pragma once

struct MPSCompileOptions_st; using MPSCompileOptions_t = MPSCompileOptions_st*;

static_assert(sizeof(MPSCompileOptions_t) == sizeof(void*), "MPSCompileOptions must be pointer-sized.");

namespace orteaf::internal::backend::mps {

/// Create a new MTLCompileOptions object.
/// Returns nullptr if allocation fails.
[[nodiscard]] MPSCompileOptions_t create_compile_options();

/// Destroy a compile options object.
void destroy_compile_options(MPSCompileOptions_t options);

/// Set math mode for fast math.
/// Only available on macOS 15.0 SDK or later.
void set_compile_options_math_mode(MPSCompileOptions_t options, bool fast_math_enabled);

/// Set preserve invariance flag.
void set_compile_options_preserve_invariance(MPSCompileOptions_t options, bool preserve_invariance);

/// Set preprocessor macros dictionary.
/// The dictionary should be created by the upper layer.
void set_compile_options_preprocessor_macros(MPSCompileOptions_t options, void* macros_dictionary);

} // namespace orteaf::internal::backend::mps
