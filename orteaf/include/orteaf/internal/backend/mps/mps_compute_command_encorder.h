#pragma once

#include "orteaf/internal/backend/mps/mps_command_buffer.h"
#include "orteaf/internal/backend/mps/mps_buffer.h"
#include "orteaf/internal/backend/mps/mps_pipeline_state.h"
#include "orteaf/internal/backend/mps/mps_size.h"

#include <cstddef>

struct MPSComputeCommandEncoder_st; using MPSComputeCommandEncoder_t = MPSComputeCommandEncoder_st*;

static_assert(sizeof(MPSComputeCommandEncoder_t) == sizeof(void*), "MPSComputeCommandEncoder must be pointer-sized.");

namespace orteaf::internal::backend::mps {

MPSComputeCommandEncoder_t create_compute_command_encoder(MPSCommandBuffer_t command_buffer);
void destroy_compute_command_encoder(MPSComputeCommandEncoder_t compute_command_encoder);
void end_encoding(MPSComputeCommandEncoder_t compute_command_encoder);

void set_pipeline_state(MPSComputeCommandEncoder_t compute_command_encoder, MPSPipelineState_t pipeline_state);
void set_buffer(MPSComputeCommandEncoder_t compute_command_encoder, MPSBuffer_t buffer, size_t offset, size_t index);
void set_bytes(MPSComputeCommandEncoder_t compute_command_encoder, const void* bytes, size_t length, size_t index);
void set_threadgroups(MPSComputeCommandEncoder_t compute_command_encoder,
                      MPSSize_t threadgroups,
                      MPSSize_t threads_per_threadgroup);
} // namespace orteaf::internal::backend::mps
