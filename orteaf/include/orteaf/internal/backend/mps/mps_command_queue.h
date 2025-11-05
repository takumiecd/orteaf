#pragma once

#include "orteaf/internal/backend/mps/mps_device.h"

struct MPSCommandQueue_st; using MPSCommandQueue_t = MPSCommandQueue_st*;

static_assert(sizeof(MPSCommandQueue_t) == sizeof(void*), "MPSCommandQueue must be pointer-sized.");

namespace orteaf::internal::backend::mps {

MPSCommandQueue_t create_command_queue(MPSDevice_t device);
void destroy_command_queue(MPSCommandQueue_t command_queue);

} // namespace orteaf::internal::backend::mps