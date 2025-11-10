#pragma once

#include "orteaf/internal/architecture/architecture.h"

namespace orteaf::internal::architecture {

inline Architecture detect_cpu_architecture() {
    return Architecture::cpu_generic;
}

} // namespace orteaf::internal::architecture
