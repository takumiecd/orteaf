#pragma once

#include <cstddef>

#include "orteaf/internal/backend/mps/mps_device.h"
#include "orteaf/internal/backend/mps/mps_string.h"
#include "orteaf/internal/backend/mps/mps_compile_options.h"
#include "orteaf/internal/backend/mps/mps_error.h"

struct MPSLibrary_st; using MPSLibrary_t = MPSLibrary_st*;

static_assert(sizeof(MPSLibrary_t) == sizeof(void*), "MPSLibrary must be pointer-sized.");

namespace orteaf::internal::backend::mps {

/// Create a library from a name.
/// The name string should be converted to NSString* by the upper layer.
[[nodiscard]] MPSLibrary_t create_library(MPSDevice_t device, MPSString_t name, MPSError_t* error = nullptr);

/// Create a library from source code.
/// The source string and compile options should be prepared by the upper layer.
[[nodiscard]] MPSLibrary_t create_library_with_source(MPSDevice_t device,
                                                      MPSString_t source,
                                                      MPSCompileOptions_t compile_options,
                                                      MPSError_t* error = nullptr);

/// Create a library from binary data.
[[nodiscard]] MPSLibrary_t create_library_with_data(MPSDevice_t device,
                                                    const void* data,
                                                    std::size_t size,
                                                    MPSError_t* error = nullptr);

/// Destroy a library.
void destroy_library(MPSLibrary_t library);

} // namespace orteaf::internal::backend::mps
