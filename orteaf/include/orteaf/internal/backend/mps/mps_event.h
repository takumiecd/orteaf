#pragma once

#include "orteaf/internal/backend/mps/mps_device.h"
#include "orteaf/internal/backend/mps/mps_command_queue.h"

struct MPSEvent_st;
struct MPSCommandBuffer_st;
using MPSEvent_t = MPSEvent_st*;
using MPSCommandBuffer_t = MPSCommandBuffer_st*;

static_assert(sizeof(MPSCommandBuffer_t) == sizeof(void*), "MPSCommandBuffer must be pointer-sized.");
static_assert(sizeof(MPSEvent_t) == sizeof(void*), "MPSEvent_t must be pointer-sized.");

namespace orteaf::internal::backend::mps {

MPSEvent_t create_event(MPSDevice_t device);
void destroy_event(MPSEvent_t event);
void record_event(MPSEvent_t event, MPSCommandBuffer_t command_buffer, uint64_t value = 1);
bool query_event(MPSEvent_t event, uint64_t expected_value = 1);
uint64_t event_value(MPSEvent_t event);
void write_event(MPSCommandQueue_t command_queue, MPSEvent_t event, uint64_t value = 1);
void wait_event(MPSCommandBuffer_t command_buffer, MPSEvent_t event, uint64_t value = 1);
void write_event_queue(MPSCommandQueue_t command_queue, MPSEvent_t event, uint64_t value = 1);
void wait_event_queue(MPSCommandQueue_t command_queue, MPSEvent_t event, uint64_t value = 1);


} // namespace orteaf::internal::backend::mps

