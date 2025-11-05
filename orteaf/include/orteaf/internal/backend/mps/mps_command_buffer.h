#pragma once

#include "orteaf/internal/backend/mps/mps_command_queue.h"
#include "orteaf/internal/backend/mps/mps_event.h"

namespace orteaf::internal::backend::mps {

MPSCommandBuffer_t create_command_buffer(MPSCommandQueue_t command_queue);
void destroy_command_buffer(MPSCommandBuffer_t command_buffer);

void encode_signal_event(MPSCommandBuffer_t command_buffer, MPSEvent_t event, uint32_t value);
void encode_wait(MPSCommandBuffer_t command_buffer, MPSEvent_t event, uint32_t value);

void commit(MPSCommandBuffer_t command_buffer);
void wait_until_completed(MPSCommandBuffer_t command_buffer);

} // namespace orteaf::internal::backend::mps