#pragma once

#include "orteaf/internal/backend/mps/mps_device.h"

#include <cstddef>

struct MPSBuffer_st; using MPSBuffer_t = MPSBuffer_st*;
using MPSBufferUsage_t = unsigned long;
inline constexpr MPSBufferUsage_t kMPSDefaultBufferUsage = 0;

static_assert(sizeof(MPSBuffer_t) == sizeof(void*), "MPSBuffer must be pointer-sized.");

namespace orteaf::internal::backend::mps {

MPSBuffer_t create_buffer(MPSDevice_t device, size_t size, MPSBufferUsage_t usage = kMPSDefaultBufferUsage);
void destroy_buffer(MPSBuffer_t buffer);
const void* get_buffer_contents_const(MPSBuffer_t buffer);
void* get_buffer_contents(MPSBuffer_t buffer);

} // namespace orteaf::internal::backend::mps
