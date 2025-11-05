#pragma once

#include "orteaf/internal/backend/mps/mps_function.h"
#include "orteaf/internal/backend/mps/mps_device.h"
#include "orteaf/internal/backend/mps/mps_error.h"

struct MPSPipelineState_st; using MPSPipelineState_t = MPSPipelineState_st*;

static_assert(sizeof(MPSPipelineState_t) == sizeof(void*), "MPSPipelineState must be pointer-sized.");

namespace orteaf::internal::backend::mps {

MPSPipelineState_t create_pipeline_state(MPSDevice_t device, MPSFunction_t function, MPSError_t* error = nullptr);

void destroy_pipeline_state(MPSPipelineState_t pipeline_state);

} // namespace orteaf::internal::backend::mps