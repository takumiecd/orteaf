#include "orteaf/internal/kernel/core/kernel_key.h"

#include <gtest/gtest.h>

#include <unordered_map>
#include <unordered_set>

namespace kernel = orteaf::internal::kernel;
namespace kk = kernel::kernel_key;
using Architecture = orteaf::internal::architecture::Architecture;
using Execution = orteaf::internal::execution::Execution;
using DType = orteaf::internal::DType;
using Op = orteaf::internal::ops::Op;

// ============================================================
// KernelKey encoding tests
// ============================================================

TEST(KernelKey, MakeEncodesAllFields) {
  auto key = kk::make(static_cast<Op>(123), Architecture::CpuGeneric,
                      static_cast<kernel::Layout>(5), DType::F32,
                      static_cast<kernel::Variant>(7));

  EXPECT_NE(static_cast<std::uint64_t>(key), 0);
}

TEST(KernelKey, RoundTripPreservesValues) {
  const auto op_id = static_cast<Op>(42);
  const auto architecture = Architecture::CpuGeneric;
  const auto layout = static_cast<kernel::Layout>(3);
  const auto dtype = DType::F32;
  const auto variant = static_cast<kernel::Variant>(2);

  auto key = kk::make(op_id, architecture, layout, dtype, variant);

  EXPECT_EQ(kk::getOp(key), op_id);
  EXPECT_EQ(kk::getArchitecture(key), architecture);
  EXPECT_EQ(kk::getExecution(key), Execution::Cpu);
  EXPECT_EQ(kk::getLayout(key), layout);
  EXPECT_EQ(kk::getDType(key), dtype);
  EXPECT_EQ(kk::getVariant(key), variant);
}

// ============================================================
// KernelKey decoding tests
// ============================================================

TEST(KernelKey, getOpExtractsCorrectBits) {
  auto key = kk::make(static_cast<Op>(0xABCD), Architecture::CpuGeneric,
                      static_cast<kernel::Layout>(0), DType::F32,
                      static_cast<kernel::Variant>(0));

  EXPECT_EQ(static_cast<std::uint64_t>(kk::getOp(key)), 0xABCD);
}

TEST(KernelKey, GetArchitectureExtractsCorrectBits) {
  auto key = kk::make(static_cast<Op>(0), Architecture::CpuGeneric,
                      static_cast<kernel::Layout>(0), DType::F32,
                      static_cast<kernel::Variant>(0));

  EXPECT_EQ(kk::getArchitecture(key), Architecture::CpuGeneric);
}

TEST(KernelKey, GetExecutionDerivedFromArchitecture) {
  auto cpu_key = kk::make(static_cast<Op>(0), Architecture::CpuGeneric,
                          static_cast<kernel::Layout>(0), DType::F32,
                          static_cast<kernel::Variant>(0));

  EXPECT_EQ(kk::getExecution(cpu_key), Execution::Cpu);

  auto mps_key = kk::make(static_cast<Op>(0), Architecture::MpsGeneric,
                          static_cast<kernel::Layout>(0), DType::F32,
                          static_cast<kernel::Variant>(0));

  EXPECT_EQ(kk::getExecution(mps_key), Execution::Mps);
}

TEST(KernelKey, GetLayoutExtractsCorrectBits) {
  auto key = kk::make(static_cast<Op>(0), Architecture::CpuGeneric,
                      static_cast<kernel::Layout>(0x42), DType::F32,
                      static_cast<kernel::Variant>(0));

  EXPECT_EQ(static_cast<std::uint64_t>(kk::getLayout(key)), 0x42);
}

TEST(KernelKey, GetDTypeExtractsCorrectBits) {
  auto key = kk::make(static_cast<Op>(0), Architecture::CpuGeneric,
                      static_cast<kernel::Layout>(0), DType::I32,
                      static_cast<kernel::Variant>(0));

  EXPECT_EQ(kk::getDType(key), DType::I32);
}

TEST(KernelKey, GetVariantExtractsCorrectBits) {
  auto key = kk::make(static_cast<Op>(0), Architecture::CpuGeneric,
                      static_cast<kernel::Layout>(0), DType::F32,
                      static_cast<kernel::Variant>(0x99));

  EXPECT_EQ(static_cast<std::uint64_t>(kk::getVariant(key)), 0x99);
}

TEST(KernelKey, GetVersionExtractsCorrectBits) {
  auto key = kk::make(static_cast<Op>(0), Architecture::CpuGeneric,
                      static_cast<kernel::Layout>(0), DType::F32,
                      static_cast<kernel::Variant>(0), 5);

  EXPECT_EQ(kk::getVersion(key), 5);
}

// ============================================================
// KernelKey comparison tests
// ============================================================

TEST(KernelKey, EqualityOperator) {
  auto key1 = kk::make(static_cast<Op>(1), Architecture::CpuGeneric,
                       static_cast<kernel::Layout>(0), DType::F32,
                       static_cast<kernel::Variant>(0));

  auto key2 = kk::make(static_cast<Op>(1), Architecture::CpuGeneric,
                       static_cast<kernel::Layout>(0), DType::F32,
                       static_cast<kernel::Variant>(0));

  auto key3 = kk::make(static_cast<Op>(2), Architecture::CpuGeneric,
                       static_cast<kernel::Layout>(0), DType::F32,
                       static_cast<kernel::Variant>(0));

  EXPECT_TRUE(key1 == key2);
  EXPECT_FALSE(key1 == key3);
}

TEST(KernelKey, InequalityOperator) {
  auto key1 = kk::make(static_cast<Op>(1), Architecture::CpuGeneric,
                       static_cast<kernel::Layout>(0), DType::F32,
                       static_cast<kernel::Variant>(0));

  auto key2 = kk::make(static_cast<Op>(2), Architecture::CpuGeneric,
                       static_cast<kernel::Layout>(0), DType::F32,
                       static_cast<kernel::Variant>(0));

  EXPECT_TRUE(key1 != key2);
  EXPECT_FALSE(key1 != key1);
}

TEST(KernelKey, LessThanOperator) {
  auto key1 = kk::make(static_cast<Op>(1), Architecture::CpuGeneric,
                       static_cast<kernel::Layout>(0), DType::F32,
                       static_cast<kernel::Variant>(0));

  auto key2 = kk::make(static_cast<Op>(2), Architecture::CpuGeneric,
                       static_cast<kernel::Layout>(0), DType::F32,
                       static_cast<kernel::Variant>(0));

  EXPECT_TRUE(key1 < key2);
  EXPECT_FALSE(key2 < key1);
}

// ============================================================
// KernelKey hash support tests
// ============================================================

TEST(KernelKey, HashSupport) {
  auto key1 = kk::make(static_cast<Op>(1), Architecture::CpuGeneric,
                       static_cast<kernel::Layout>(0), DType::F32,
                       static_cast<kernel::Variant>(0));

  auto key2 = kk::make(static_cast<Op>(1), Architecture::CpuGeneric,
                       static_cast<kernel::Layout>(0), DType::F32,
                       static_cast<kernel::Variant>(0));

  auto key3 = kk::make(static_cast<Op>(2), Architecture::CpuGeneric,
                       static_cast<kernel::Layout>(0), DType::F32,
                       static_cast<kernel::Variant>(0));

  std::hash<kernel::KernelKey> hasher;

  EXPECT_EQ(hasher(key1), hasher(key2));
  EXPECT_NE(hasher(key1), hasher(key3));
}

TEST(KernelKey, UnorderedMapUsage) {
  std::unordered_map<kernel::KernelKey, std::string> kernel_map;

  auto key1 = kk::make(static_cast<Op>(1), Architecture::CpuGeneric,
                       static_cast<kernel::Layout>(0), DType::F32,
                       static_cast<kernel::Variant>(0));

  auto key2 = kk::make(static_cast<Op>(2), Architecture::CpuGeneric,
                       static_cast<kernel::Layout>(1), DType::I32,
                       static_cast<kernel::Variant>(1));

  kernel_map[key1] = "CPU F32 Add Baseline";
  kernel_map[key2] = "CPU I32 Mul Optimized";

  EXPECT_EQ(kernel_map.size(), 2);
  EXPECT_EQ(kernel_map[key1], "CPU F32 Add Baseline");
  EXPECT_EQ(kernel_map[key2], "CPU I32 Mul Optimized");
}

TEST(KernelKey, UnorderedSetUsage) {
  std::unordered_set<kernel::KernelKey> kernel_set;

  auto key1 = kk::make(static_cast<Op>(1), Architecture::CpuGeneric,
                       static_cast<kernel::Layout>(0), DType::F32,
                       static_cast<kernel::Variant>(0));

  auto key2 = kk::make(static_cast<Op>(1), Architecture::CpuGeneric,
                       static_cast<kernel::Layout>(0), DType::F32,
                       static_cast<kernel::Variant>(0));

  kernel_set.insert(key1);
  kernel_set.insert(key2);

  EXPECT_EQ(kernel_set.size(), 1);
}

// ============================================================
// KernelKey boundary tests
// ============================================================

TEST(KernelKey, MaxValuesBoundary) {
  auto key =
      kk::make(static_cast<Op>(0xFFFF), static_cast<Architecture>(0xFFFF),
               static_cast<kernel::Layout>(0xFF), static_cast<DType>(0xFFF),
               static_cast<kernel::Variant>(0xFF), 0xF);

  EXPECT_EQ(static_cast<std::uint64_t>(kk::getOp(key)), 0xFFFF);
  EXPECT_EQ(static_cast<std::uint64_t>(kk::getArchitecture(key)), 0xFFFF);
  EXPECT_EQ(static_cast<std::uint64_t>(kk::getLayout(key)), 0xFF);
  EXPECT_EQ(static_cast<std::uint64_t>(kk::getDType(key)), 0xFFF);
  EXPECT_EQ(static_cast<std::uint64_t>(kk::getVariant(key)), 0xFF);
  EXPECT_EQ(kk::getVersion(key), 0xF);
}

TEST(KernelKey, ZeroValues) {
  auto key = kk::make(static_cast<Op>(0), static_cast<Architecture>(0),
                      static_cast<kernel::Layout>(0), static_cast<DType>(0),
                      static_cast<kernel::Variant>(0), 0);

  EXPECT_EQ(static_cast<std::uint64_t>(key), 0);
  EXPECT_EQ(static_cast<std::uint64_t>(kk::getOp(key)), 0);
  EXPECT_EQ(static_cast<std::uint64_t>(kk::getArchitecture(key)), 0);
  EXPECT_EQ(static_cast<std::uint64_t>(kk::getLayout(key)), 0);
  EXPECT_EQ(static_cast<std::uint64_t>(kk::getDType(key)), 0);
  EXPECT_EQ(static_cast<std::uint64_t>(kk::getVariant(key)), 0);
  EXPECT_EQ(kk::getVersion(key), 0);
}

// ============================================================
// KernelKey constexpr tests
// ============================================================

TEST(KernelKey, ConstexprSupport) {
  constexpr auto key = kk::make(static_cast<Op>(42), Architecture::CpuGeneric,
                                static_cast<kernel::Layout>(3), DType::F32,
                                static_cast<kernel::Variant>(1));

  constexpr auto op_id = kk::getOp(key);
  constexpr auto architecture = kk::getArchitecture(key);
  constexpr auto layout = kk::getLayout(key);
  constexpr auto dtype = kk::getDType(key);
  constexpr auto variant = kk::getVariant(key);
  constexpr auto version = kk::getVersion(key);

  static_assert(static_cast<std::uint64_t>(op_id) == 42);
  static_assert(architecture == Architecture::CpuGeneric);
  static_assert(static_cast<std::uint64_t>(layout) == 3);
  static_assert(dtype == DType::F32);
  static_assert(static_cast<std::uint64_t>(variant) == 1);
  static_assert(version == 0);
}

// ============================================================
// KernelKey practical use case tests
// ============================================================

TEST(KernelKey, PracticalUseCaseExample) {
  // Simulate a kernel registry
  using KernelFunction = void (*)();
  std::unordered_map<kernel::KernelKey, KernelFunction> registry;

  // Register different kernel variants
  auto cpu_f32_add_baseline =
      kk::make(static_cast<Op>(1), // Add
               Architecture::CpuGeneric,
               static_cast<kernel::Layout>(0), // RowMajor
               DType::F32,
               static_cast<kernel::Variant>(0) // Baseline
      );

  auto cpu_f32_add_optimized =
      kk::make(static_cast<Op>(1), // Add
               Architecture::CpuGeneric,
               static_cast<kernel::Layout>(0), // RowMajor
               DType::F32,
               static_cast<kernel::Variant>(1) // Optimized
      );

  registry[cpu_f32_add_baseline] = nullptr; // Would be actual function ptr
  registry[cpu_f32_add_optimized] = nullptr;

  EXPECT_EQ(registry.size(), 2);
  EXPECT_TRUE(registry.count(cpu_f32_add_baseline) > 0);
  EXPECT_TRUE(registry.count(cpu_f32_add_optimized) > 0);

  // Verify keys are distinct
  EXPECT_NE(cpu_f32_add_baseline, cpu_f32_add_optimized);
}

// ============================================================
// KernelKey architecture-specific tests
// ============================================================

TEST(KernelKey, DifferentArchitecturesProduceDifferentKeys) {
  auto cpu_key = kk::make(static_cast<Op>(1), Architecture::CpuGeneric,
                          static_cast<kernel::Layout>(0), DType::F32,
                          static_cast<kernel::Variant>(0));

  auto mps_key = kk::make(static_cast<Op>(1), Architecture::MpsGeneric,
                          static_cast<kernel::Layout>(0), DType::F32,
                          static_cast<kernel::Variant>(0));

  EXPECT_NE(cpu_key, mps_key);
  EXPECT_EQ(kk::getExecution(cpu_key), Execution::Cpu);
  EXPECT_EQ(kk::getExecution(mps_key), Execution::Mps);
}
