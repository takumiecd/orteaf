#include "orteaf/internal/architecture/architecture.h"

#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

namespace arch = orteaf::internal::architecture;
namespace execution = orteaf::internal::execution;

TEST(ArchitectureBasic, GenericLocalIndexIsZero) {
  EXPECT_EQ(arch::localIndexOf(arch::Architecture::CudaGeneric), 0);
  EXPECT_EQ(arch::localIndexOf(arch::Architecture::MpsGeneric), 0);
  EXPECT_EQ(arch::localIndexOf(arch::Architecture::CpuGeneric), 0);

  EXPECT_TRUE(arch::isGeneric(arch::Architecture::CudaGeneric));
  EXPECT_TRUE(arch::isGeneric(arch::Architecture::MpsGeneric));
  EXPECT_TRUE(arch::isGeneric(arch::Architecture::CpuGeneric));
}

TEST(ArchitectureBasic, LocalIndicesIncrementPerExecution) {
  EXPECT_EQ(arch::localIndexOf(arch::Architecture::CudaSm80), 1);
  EXPECT_EQ(arch::localIndexOf(arch::Architecture::CudaSm86), 2);
  EXPECT_EQ(arch::localIndexOf(arch::Architecture::CudaSm90), 3);
  EXPECT_EQ(arch::localIndexOf(arch::Architecture::MpsM2), 1);
  EXPECT_EQ(arch::localIndexOf(arch::Architecture::MpsM3), 2);
  EXPECT_EQ(arch::localIndexOf(arch::Architecture::MpsM4), 3);
}

TEST(ArchitectureMetadata, ExecutionAssociationMatches) {
  EXPECT_EQ(arch::executionOf(arch::Architecture::CudaSm80),
            execution::Execution::Cuda);
  EXPECT_EQ(arch::executionOf(arch::Architecture::MpsM3),
            execution::Execution::Mps);
  EXPECT_EQ(arch::executionOf(arch::Architecture::CpuZen4),
            execution::Execution::Cpu);
}

TEST(ArchitectureMetadata, IdAndDisplayNameMatchYaml) {
  EXPECT_EQ(arch::idOf(arch::Architecture::CudaSm80), std::string_view("Sm80"));
  EXPECT_EQ(arch::displayNameOf(arch::Architecture::CudaSm80),
            std::string_view("CUDA SM80"));
  EXPECT_EQ(arch::descriptionOf(arch::Architecture::CudaSm80),
            std::string_view("Ampere 世代 GPU (A100 など) 向け最適化"));

  EXPECT_EQ(arch::idOf(arch::Architecture::CpuSkylake),
            std::string_view("Skylake"));
  EXPECT_EQ(arch::displayNameOf(arch::Architecture::CpuSkylake),
            std::string_view("Skylake AVX512"));
}

TEST(ArchitectureLookup, ExecutionCountsIncludeGeneric) {
  auto verify_execution = [](execution::Execution execution_id) {
    const auto count = arch::countForExecution(execution_id);
    const auto span = arch::architecturesOf(execution_id);
    EXPECT_EQ(count, span.size());
    ASSERT_GE(count, 1u);
    EXPECT_TRUE(arch::isGeneric(span.front()));
  };
  verify_execution(execution::Execution::Cuda);
  verify_execution(execution::Execution::Mps);
  verify_execution(execution::Execution::Cpu);
}

TEST(ArchitectureLookup, ArchitecturesOfReturnsContiguousSpan) {
  const auto cuda_archs = arch::architecturesOf(execution::Execution::Cuda);
  ASSERT_GE(cuda_archs.size(), 4u);
  EXPECT_EQ(cuda_archs.front(), arch::Architecture::CudaGeneric);
  EXPECT_NE(std::find(cuda_archs.begin(), cuda_archs.end(),
                      arch::Architecture::CudaSm80),
            cuda_archs.end());
  EXPECT_EQ(cuda_archs.back(), arch::Architecture::CudaSm90);

  const auto cpu_archs = arch::architecturesOf(execution::Execution::Cpu);
  ASSERT_GE(cpu_archs.size(), 3u);
  EXPECT_NE(std::find(cpu_archs.begin(), cpu_archs.end(),
                      arch::Architecture::CpuZen4),
            cpu_archs.end());
  EXPECT_NE(std::find(cpu_archs.begin(), cpu_archs.end(),
                      arch::Architecture::CpuIntelCometLake),
            cpu_archs.end());
}

TEST(ArchitectureLookup, FromExecutionAndLocalIndexRoundsTrip) {
  const auto arch_id =
      arch::fromExecutionAndLocalIndex(execution::Execution::Cuda, 3);
  EXPECT_EQ(arch_id, arch::Architecture::CudaSm90);
  EXPECT_TRUE(arch::hasLocalIndex(execution::Execution::Cuda, 3));
  EXPECT_FALSE(arch::hasLocalIndex(execution::Execution::Cuda, 5));
}

TEST(ArchitectureParent, GenericHasNoParent) {
  EXPECT_FALSE(arch::hasParent(arch::Architecture::CudaGeneric));
  EXPECT_FALSE(arch::hasParent(arch::Architecture::MpsGeneric));
  EXPECT_FALSE(arch::hasParent(arch::Architecture::CpuGeneric));
}

TEST(ArchitectureParent, NonGenericHasParent) {
  // All non-generic architectures have a parent (defaults to Generic)
  EXPECT_TRUE(arch::hasParent(arch::Architecture::CudaSm80));
  EXPECT_TRUE(arch::hasParent(arch::Architecture::MpsM3));
  EXPECT_TRUE(arch::hasParent(arch::Architecture::CpuZen4));
}

TEST(ArchitectureParent, ParentOfGenericReturnsSelf) {
  EXPECT_EQ(arch::parentOf(arch::Architecture::CudaGeneric),
            arch::Architecture::CudaGeneric);
  EXPECT_EQ(arch::parentOf(arch::Architecture::MpsGeneric),
            arch::Architecture::MpsGeneric);
}

TEST(ArchitectureParent, ParentOfNonGenericIsWithinSameExecution) {
  // Without explicit parent, defaults to Generic
  const auto parent = arch::parentOf(arch::Architecture::CudaSm80);
  EXPECT_EQ(arch::executionOf(parent), execution::Execution::Cuda);
  EXPECT_EQ(parent, arch::Architecture::CudaGeneric);
}

TEST(ArchitectureParent, GenericOfReturnsCorrectGeneric) {
  EXPECT_EQ(arch::genericOf(arch::Architecture::CudaSm80),
            arch::Architecture::CudaGeneric);
  EXPECT_EQ(arch::genericOf(arch::Architecture::MpsM3),
            arch::Architecture::MpsGeneric);
  EXPECT_EQ(arch::genericOf(arch::Architecture::CpuZen4),
            arch::Architecture::CpuGeneric);
}

TEST(ArchitectureParent, ForEachFallbackVisitsChain) {
  std::vector<arch::Architecture> visited;
  arch::forEachFallback(arch::Architecture::CudaSm80,
                        [&](arch::Architecture a) { visited.push_back(a); });

  // Verifies parent chain: Sm80 -> CudaGeneric
  ASSERT_EQ(visited.size(), 2u);
  EXPECT_EQ(visited[0], arch::Architecture::CudaSm80);
  EXPECT_EQ(visited[1], arch::Architecture::CudaGeneric);
}

TEST(ArchitectureParent, ForEachFallbackCanEarlyReturn) {
  std::vector<arch::Architecture> visited;
  arch::forEachFallback(arch::Architecture::CudaSm80,
                        [&](arch::Architecture a) -> bool {
                          visited.push_back(a);
                          return false; // Stop immediately
                        });

  ASSERT_EQ(visited.size(), 1u);
  EXPECT_EQ(visited[0], arch::Architecture::CudaSm80);
}
