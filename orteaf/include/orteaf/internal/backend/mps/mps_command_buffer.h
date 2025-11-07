/**
 * @file mps_command_buffer.h
 * @brief MPS/Metal command buffer creation and submission helpers.
 */
#pragma once

#include "orteaf/internal/backend/mps/mps_command_queue.h"
#include "orteaf/internal/backend/mps/mps_event.h"

namespace orteaf::internal::backend::mps {

/**
 * @brief Create a command buffer from a command queue.
 * @return Opaque command buffer handle, or nullptr when unavailable/disabled.
 */
MPSCommandBuffer_t create_command_buffer(MPSCommandQueue_t command_queue);

/**
 * @brief Destroy a command buffer.
 */
void destroy_command_buffer(MPSCommandBuffer_t command_buffer);

/**
 * @brief Encode a signal to an event with the given value.
 */
void encode_signal_event(MPSCommandBuffer_t command_buffer, MPSEvent_t event, uint32_t value);

/**
 * @brief Encode a wait on an event until it reaches the given value.
 */
void encode_wait(MPSCommandBuffer_t command_buffer, MPSEvent_t event, uint32_t value);

/**
 * @brief Commit the command buffer for execution.
 */
void commit(MPSCommandBuffer_t command_buffer);

/**
 * @brief Block until the command buffer has completed execution.
 */
void wait_until_completed(MPSCommandBuffer_t command_buffer);

} // namespace orteaf::internal::backend::mps