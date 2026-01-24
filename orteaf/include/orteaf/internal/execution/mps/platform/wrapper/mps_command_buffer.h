/**
 * @file mps_command_buffer.h
 * @brief MPS/Metal command buffer creation and submission helpers.
 */
#pragma once

#if ORTEAF_ENABLE_MPS

#include <cstdint>

#include "orteaf/internal/execution/mps/platform/wrapper/mps_types.h"

namespace orteaf::internal::execution::mps::platform::wrapper {

/**
 * @brief Create a command buffer from a command queue.
 * @return Opaque command buffer handle, or nullptr when unavailable/disabled.
 */
MpsCommandBuffer_t createCommandBuffer(MpsCommandQueue_t command_queue);

/**
 * @brief Destroy a command buffer.
 */
void destroyCommandBuffer(MpsCommandBuffer_t command_buffer);

/**
 * @brief Encode a signal to an event with the given value.
 */
void encodeSignalEvent(MpsCommandBuffer_t command_buffer, MpsEvent_t event,
                       uint32_t value);

/**
 * @brief Encode a wait on an event until it reaches the given value.
 */
void encodeWait(MpsCommandBuffer_t command_buffer, MpsEvent_t event,
                uint32_t value);

/**
 * @brief Commit the command buffer for execution.
 */
void commit(MpsCommandBuffer_t command_buffer);

/**
 * @brief Check if the command buffer has completed execution.
 */
bool isCompleted(MpsCommandBuffer_t command_buffer);

/**
 * @brief Block until the command buffer has completed execution.
 */
void waitUntilCompleted(MpsCommandBuffer_t command_buffer);

/**
 * @brief Get the GPU start time of the command buffer (in seconds).
 * 
 * Returns the time when the GPU started executing this command buffer.
 * Only valid after the command buffer has been scheduled.
 * 
 * @return GPU start time in seconds, or 0.0 if not available
 */
double getGPUStartTime(MpsCommandBuffer_t command_buffer);

/**
 * @brief Get the GPU end time of the command buffer (in seconds).
 * 
 * Returns the time when the GPU finished executing this command buffer.
 * Only valid after the command buffer has completed.
 * 
 * @return GPU end time in seconds, or 0.0 if not available
 */
double getGPUEndTime(MpsCommandBuffer_t command_buffer);

/**
 * @brief Get the GPU execution duration of the command buffer (in seconds).
 * 
 * Returns the elapsed time between GPU start and end.
 * Only valid after the command buffer has completed.
 * 
 * @return GPU execution duration in seconds, or 0.0 if not available
 */
double getGPUDuration(MpsCommandBuffer_t command_buffer);

} // namespace orteaf::internal::execution::mps::platform::wrapper

#endif  // ORTEAF_ENABLE_MPS
