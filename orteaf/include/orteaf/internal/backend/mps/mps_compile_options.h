/**
 * @file mps_compile_options.h
 * @brief MPS/Metal compile options creation and configuration.
 */
#pragma once

namespace orteaf::internal::backend::mps {

struct MPSCompileOptions_st; using MPSCompileOptions_t = MPSCompileOptions_st*;

static_assert(sizeof(MPSCompileOptions_t) == sizeof(void*), "MPSCompileOptions must be pointer-sized.");

/** Create a new `MTLCompileOptions` object (opaque). */
[[nodiscard]] MPSCompileOptions_t create_compile_options();

/** Destroy a compile options object; ignores nullptr. */
void destroy_compile_options(MPSCompileOptions_t options);

/** Set math mode (fast/safe). Requires macOS 15.0 SDK+. */
void set_compile_options_math_mode(MPSCompileOptions_t options, bool fast_math_enabled);

/** Set preserve invariance flag. */
void set_compile_options_preserve_invariance(MPSCompileOptions_t options, bool preserve_invariance);

/** Set preprocessor macros (expects NSDictionary*). */
void set_compile_options_preprocessor_macros(MPSCompileOptions_t options, void* macros_dictionary);

} // namespace orteaf::internal::backend::mps
