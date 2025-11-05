#pragma once

#include "orteaf/internal/backend/mps/mps_size.h"

struct MPSDevice_st; using MPSDevice_t = MPSDevice_st*;
struct MPSDeviceArray_st; using MPSDeviceArray_t = MPSDeviceArray_st*;

static_assert(sizeof(MPSDevice_t) == sizeof(void*), "MPSDevice must be pointer-sized.");

namespace orteaf::internal::backend::mps {

MPSDevice_t get_device();
MPSDevice_t get_device(MPSInt_t device_id);
int get_device_count();
void device_retain(MPSDevice_t device);
void device_release(MPSDevice_t device);
MPSDeviceArray_t get_device_array();


} // namespace orteaf::internal::backend::mps
