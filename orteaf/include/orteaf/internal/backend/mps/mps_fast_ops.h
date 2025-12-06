#pragma once

#include "orteaf/internal/backend/mps/wrapper/mps_command_buffer.h"
#include "orteaf/internal/backend/mps/wrapper/mps_compute_command_encorder.h"

namespace orteaf::internal::runtime::backend_ops::mps {

struct MpsFastOps {
  // Fast-path wrapper for command buffer creation.
  static inline ::orteaf::internal::backend::mps::MPSCommandBuffer_t createCommandBuffer(
      ::orteaf::internal::backend::mps::MPSCommandQueue_t command_queue) {
    return ::orteaf::internal::backend::mps::createCommandBuffer(command_queue);
  }

  // Fast-path wrapper for compute encoder creation and configuration primitives.
  static inline ::orteaf::internal::backend::mps::MPSComputeCommandEncoder_t createComputeCommandEncoder(
      ::orteaf::internal::backend::mps::MPSCommandBuffer_t command_buffer) {
    return ::orteaf::internal::backend::mps::createComputeCommandEncoder(command_buffer);
  }

  static inline void setPipelineState(
      ::orteaf::internal::backend::mps::MPSComputeCommandEncoder_t encoder,
      ::orteaf::internal::backend::mps::MPSComputePipelineState_t pipeline) {
    ::orteaf::internal::backend::mps::setPipelineState(encoder, pipeline);
  }
};

} // namespace orteaf::internal::runtime::backend_ops::mps
