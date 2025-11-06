/**
 * @file mps_pipeline_state.h
 * @brief MPS/Metal compute pipeline state creation and destruction.
 */
#pragma once

#include "orteaf/internal/backend/mps/mps_function.h"
#include "orteaf/internal/backend/mps/mps_device.h"
#include "orteaf/internal/backend/mps/mps_error.h"

namespace orteaf::internal::backend::mps {

struct MPSPipelineState_st; using MPSPipelineState_t = MPSPipelineState_st*;

static_assert(sizeof(MPSPipelineState_t) == sizeof(void*), "MPSPipelineState must be pointer-sized.");

/**
 * @brief Create a compute pipeline state from a function.
 * @param device Opaque Metal device
 * @param function Opaque Metal function
 * @param error Optional error object out (bridged to NSError**)
 * @return Opaque pipeline state, or nullptr when unavailable/disabled.
 */
MPSPipelineState_t create_pipeline_state(MPSDevice_t device, MPSFunction_t function, MPSError_t* error = nullptr);

/**
 * @brief Destroy a pipeline state; ignores nullptr.
 */
void destroy_pipeline_state(MPSPipelineState_t pipeline_state);

} // namespace orteaf::internal::backend::mps