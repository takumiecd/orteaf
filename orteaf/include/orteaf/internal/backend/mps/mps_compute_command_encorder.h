/**
 * @file mps_compute_command_encorder.h
 * @brief MPS/Metal compute command encoder helpers.
 */
#pragma once

#include "orteaf/internal/backend/mps/mps_command_buffer.h"
#include "orteaf/internal/backend/mps/mps_buffer.h"
#include "orteaf/internal/backend/mps/mps_pipeline_state.h"
#include "orteaf/internal/backend/mps/mps_size.h"

#include <cstddef>

namespace orteaf::internal::backend::mps {

struct MPSComputeCommandEncoder_st; using MPSComputeCommandEncoder_t = MPSComputeCommandEncoder_st*;

static_assert(sizeof(MPSComputeCommandEncoder_t) == sizeof(void*), "MPSComputeCommandEncoder must be pointer-sized.");

/** Create a compute command encoder from a command buffer. */
MPSComputeCommandEncoder_t create_compute_command_encoder(MPSCommandBuffer_t command_buffer);
/** Destroy a compute command encoder; ignores nullptr. */
void destroy_compute_command_encoder(MPSComputeCommandEncoder_t compute_command_encoder);
/** End encoding on the compute command encoder. */
void end_encoding(MPSComputeCommandEncoder_t compute_command_encoder);

/** Bind a compute pipeline state. */
void set_pipeline_state(MPSComputeCommandEncoder_t compute_command_encoder, MPSPipelineState_t pipeline_state);
/** Bind a buffer at the given index with an offset. */
void set_buffer(MPSComputeCommandEncoder_t compute_command_encoder, MPSBuffer_t buffer, size_t offset, size_t index);
/** Bind raw bytes at the given index. */
void set_bytes(MPSComputeCommandEncoder_t compute_command_encoder, const void* bytes, size_t length, size_t index);
/** Dispatch threadgroups with the specified sizes. */
void set_threadgroups(MPSComputeCommandEncoder_t compute_command_encoder,
                      MPSSize_t threadgroups,
                      MPSSize_t threads_per_threadgroup);
} // namespace orteaf::internal::backend::mps
