#pragma once

// TODO: ARCH type definition needed
// #include "orteaf/internal/backend/arch_register.h"
#include "orteaf/internal/backend/mps/mps_device.h"

namespace orteaf::internal::backend::mps {

// MPS detection: detect GPU family from Metal device
ARCH detect_mps_arch(MPSDevice_t device);

} // namespace orteaf::internal::backend::mps
