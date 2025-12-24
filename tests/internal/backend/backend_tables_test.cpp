#include <gtest/gtest.h>

#include <string_view>

#include <orteaf/internal/execution/execution.h>

namespace execution = orteaf::internal::execution;

TEST(BackendTablesTest, BasicEnumerationProperties) {
    EXPECT_EQ(execution::kBackendCount, execution::allBackends().size());
    EXPECT_TRUE(execution::isValidIndex(0));
    EXPECT_FALSE(execution::isValidIndex(execution::kBackendCount));

    EXPECT_EQ(execution::fromIndex(0), execution::allBackends().front());
    EXPECT_EQ(execution::idOf(execution::fromIndex(0)), std::string_view("Cuda"));
}

TEST(BackendTablesTest, MetadataMatchesCatalog) {
    constexpr auto cuda = execution::Execution::Cuda;
    EXPECT_EQ(execution::displayNameOf(cuda), "CUDA");
    EXPECT_EQ(execution::modulePathOf(cuda), "@orteaf/internal/execution/cuda");
    EXPECT_EQ(execution::descriptionOf(cuda), "NVIDIA CUDA 実装");

    constexpr auto mps = execution::Execution::Mps;
    EXPECT_EQ(execution::displayNameOf(mps), "MPS");
    EXPECT_EQ(execution::modulePathOf(mps), "@orteaf/internal/execution/mps");
    EXPECT_EQ(execution::descriptionOf(mps), "macOS/iOS 向け Metal Performance Shaders 実装");

    constexpr auto cpu = execution::Execution::Cpu;
    EXPECT_EQ(execution::displayNameOf(cpu), "CPU");
    EXPECT_EQ(execution::modulePathOf(cpu), "@orteaf/internal/execution/cpu");
    EXPECT_EQ(execution::descriptionOf(cpu), "汎用 CPU 実装");
}
