#pragma once

#include <string_view>

struct MPSFunction_st; using MPSFunction_t = MPSFunction_st*;

#include "orteaf/internal/backend/mps/mps_library.h"

static_assert(sizeof(MPSFunction_t) == sizeof(void*), "MPSFunction must be pointer-sized.");

namespace orteaf::internal::backend::mps {

MPSFunction_t create_function(MPSLibrary_t library, std::string_view name);
//destroy
void destroy_function(MPSFunction_t function);

} // namespace orteaf::internal::backend::mps