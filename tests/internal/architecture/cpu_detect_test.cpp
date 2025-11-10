#include <gtest/gtest.h>

#include "orteaf/internal/architecture/architecture.h"
#include "orteaf/internal/backend/backend.h"
#include "orteaf/internal/architecture/cpu_detect.h"

namespace architecture = orteaf::internal::architecture;
namespace backend = orteaf::internal::backend;

TEST(CpuDetect, ReportsCpuBackendArchitecture) {
    const auto arch = architecture::detect_cpu_architecture();
    EXPECT_EQ(architecture::BackendOf(arch), backend::Backend::cpu);
    EXPECT_GE(static_cast<std::uint16_t>(arch), 0);
}

TEST(CpuDetect, GenericIsAlwaysValid) {
    constexpr auto generic = architecture::Architecture::cpu_generic;
    EXPECT_TRUE(architecture::IsValidIndex(static_cast<std::size_t>(generic)));
}
