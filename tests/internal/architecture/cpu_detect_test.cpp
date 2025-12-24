#include "orteaf/internal/architecture/architecture.h"
#include "orteaf/internal/execution/execution.h"
#include "orteaf/internal/architecture/cpu_detect.h"

#include <cstdlib>
#include <iostream>
#include <gtest/gtest.h>
namespace architecture = orteaf::internal::architecture;
namespace execution = orteaf::internal::execution;

/// Manual test hook: set ORTEAF_EXPECT_CPU_ARCH=Zen4 (or other ID) to assert your environment.
TEST(CpuDetect, ManualEnvironmentCheck) {
    const char* expected_env = std::getenv("ORTEAF_EXPECT_CPU_ARCH");
    if (!expected_env) {
        GTEST_SKIP() << "Set ORTEAF_EXPECT_CPU_ARCH to run this test on your environment.";
    }
    std::cout << "expected_env: " << expected_env << std::endl;
    const auto arch = architecture::detectCpuArchitecture();
    std::cout << "arch: " << architecture::idOf(arch).data() << std::endl;
    EXPECT_STREQ(expected_env, architecture::idOf(arch).data());
}

TEST(CpuDetect, ReportsCpuExecutionArchitecture) {
    const auto arch = architecture::detectCpuArchitecture();
    EXPECT_EQ(architecture::executionOf(arch), execution::Execution::Cpu);
    EXPECT_GE(static_cast<std::uint16_t>(arch), 0);
}

TEST(CpuDetect, GenericIsAlwaysValid) {
    constexpr auto generic = architecture::Architecture::CpuGeneric;
    EXPECT_TRUE(architecture::isValidIndex(static_cast<std::size_t>(generic)));
}
