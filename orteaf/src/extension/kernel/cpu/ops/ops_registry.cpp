#include "orteaf/extension/kernel/cpu/ops/print_kernel.h"

#include <orteaf/internal/kernel/registry/kernel_auto_registry.h>

ORTEAF_REGISTER_KERNEL(
    ::orteaf::extension::kernel::cpu::ops::registerPrintKernel);
