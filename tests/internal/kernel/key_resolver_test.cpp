#include "orteaf/internal/kernel/core/key_resolver.h"

#include <gtest/gtest.h>

#include <unordered_set>

#include "orteaf/internal/kernel/core/key_components.h"

namespace kernel = orteaf::internal::kernel;
namespace resolver = kernel::key_resolver;
using Architecture = orteaf::internal::architecture::Architecture;
using DType = orteaf::internal::DType;
using Op = orteaf::internal::ops::Op;

// Mock registry for testing
class MockRegistry {
public:
  void add(kernel::KernelKey key) { keys_.insert(key); }

  bool contains(kernel::KernelKey key) const {
    return keys_.find(key) != keys_.end();
  }

private:
  std::unordered_set<kernel::KernelKey> keys_;
};

// ============================================================
// KeyRequest tests
// ============================================================

TEST(KeyComponents, KeyRequestEquality) {
  kernel::KeyRequest a{static_cast<Op>(1), DType::F32,
                       Architecture::CpuGeneric};
  kernel::KeyRequest b{static_cast<Op>(1), DType::F32,
                       Architecture::CpuGeneric};
  kernel::KeyRequest c{static_cast<Op>(2), DType::F32,
                       Architecture::CpuGeneric};

  EXPECT_EQ(a, b);
  EXPECT_NE(a, c);
}

// ============================================================
// FixedKeyComponents tests
// ============================================================

TEST(KeyComponents, FixedKeyComponentsEquality) {
  kernel::FixedKeyComponents a{static_cast<Op>(1), DType::F32};
  kernel::FixedKeyComponents b{static_cast<Op>(1), DType::F32};
  kernel::FixedKeyComponents c{static_cast<Op>(2), DType::F32};

  EXPECT_EQ(a, b);
  EXPECT_NE(a, c);
}

// ============================================================
// VariableKeyComponents tests
// ============================================================

TEST(KeyComponents, VariableKeyComponentsEquality) {
  kernel::VariableKeyComponents a{Architecture::CpuGeneric,
                                  static_cast<kernel::Layout>(0),
                                  static_cast<kernel::Variant>(0)};
  kernel::VariableKeyComponents b{Architecture::CpuGeneric,
                                  static_cast<kernel::Layout>(0),
                                  static_cast<kernel::Variant>(0)};
  kernel::VariableKeyComponents c{Architecture::CpuGeneric,
                                  static_cast<kernel::Layout>(1),
                                  static_cast<kernel::Variant>(0)};

  EXPECT_EQ(a, b);
  EXPECT_NE(a, c);
}

// ============================================================
// KeyRule tests
// ============================================================

TEST(KeyComponents, KeyRuleEquality) {
  kernel::KeyRule a{{Architecture::CpuGeneric, static_cast<kernel::Layout>(0),
                     static_cast<kernel::Variant>(0)},
                    nullptr};
  kernel::KeyRule b{{Architecture::CpuGeneric, static_cast<kernel::Layout>(0),
                     static_cast<kernel::Variant>(0)},
                    nullptr};
  kernel::KeyRule c{{Architecture::CpuGeneric, static_cast<kernel::Layout>(1),
                     static_cast<kernel::Variant>(0)},
                    nullptr};

  EXPECT_EQ(a, b);
  EXPECT_NE(a, c);
}

// ============================================================
// makeKey tests
// ============================================================

TEST(KeyComponents, MakeKeyCreatesValidKey) {
  kernel::FixedKeyComponents fixed{static_cast<Op>(42), DType::F32};
  kernel::VariableKeyComponents variable{Architecture::CpuGeneric,
                                         static_cast<kernel::Layout>(3),
                                         static_cast<kernel::Variant>(1)};

  auto key = kernel::makeKey(fixed, variable);

  EXPECT_EQ(kernel::kernel_key::getOp(key), fixed.op);
  EXPECT_EQ(kernel::kernel_key::getDType(key), fixed.dtype);
  EXPECT_EQ(kernel::kernel_key::getArchitecture(key), variable.arch);
  EXPECT_EQ(kernel::kernel_key::getLayout(key), variable.layout);
  EXPECT_EQ(kernel::kernel_key::getVariant(key), variable.variant);
}

// ============================================================
// buildContext tests
// ============================================================

TEST(KeyResolver, BuildContextExtractsFixed) {
  kernel::KeyRequest request{static_cast<Op>(42), DType::F32,
                             Architecture::CpuGeneric};

  auto context = resolver::buildContext(request);

  EXPECT_EQ(context.fixed.op, request.op);
  EXPECT_EQ(context.fixed.dtype, request.dtype);
}

TEST(KeyResolver, BuildContextGeneratesRules) {
  kernel::KeyRequest request{static_cast<Op>(1), DType::F32,
                             Architecture::CpuGeneric};

  auto context = resolver::buildContext(request);

  EXPECT_GT(context.rules.size(), 0u);
  // First rule should be the requested architecture
  EXPECT_EQ(context.rules.front().components.arch, request.architecture);
}

TEST(KeyResolver, BuildContextPutsRequestedArchFirst) {
  kernel::KeyRequest request{static_cast<Op>(1), DType::F32,
                             Architecture::CpuZen4};

  auto context = resolver::buildContext(request);

  // First rule should be CpuZen4 (the requested one)
  EXPECT_EQ(context.rules.front().components.arch, Architecture::CpuZen4);
  // Last rule should be CpuGeneric (fallback)
  EXPECT_EQ(context.rules.back().components.arch, Architecture::CpuGeneric);
}

// ============================================================
// verify tests
// ============================================================

TEST(KeyResolver, VerifyWithNullPredicateUsesDefaultVerify) {
  kernel::KeyRule rule{{Architecture::CpuGeneric,
                        static_cast<kernel::Layout>(0),
                        static_cast<kernel::Variant>(0)},
                       nullptr};
  kernel::KernelArgs args;

  EXPECT_TRUE(resolver::verify(rule, args));
}

TEST(KeyResolver, VerifyWithCustomPredicateUsesIt) {
  auto alwaysFalse = [](const kernel::KernelArgs &) { return false; };

  kernel::KeyRule rule{{Architecture::CpuGeneric,
                        static_cast<kernel::Layout>(0),
                        static_cast<kernel::Variant>(0)},
                       alwaysFalse};
  kernel::KernelArgs args;

  EXPECT_FALSE(resolver::verify(rule, args));
}

// ============================================================
// resolve tests
// ============================================================

TEST(KeyResolver, ResolveFindsRegisteredKey) {
  MockRegistry registry;
  kernel::KeyRequest request{static_cast<Op>(1), DType::F32,
                             Architecture::CpuGeneric};
  kernel::VariableKeyComponents variable{Architecture::CpuGeneric,
                                         static_cast<kernel::Layout>(0),
                                         static_cast<kernel::Variant>(0)};

  auto expected_key = kernel::makeKey({request.op, request.dtype}, variable);
  registry.add(expected_key);

  kernel::KernelArgs args;
  auto result = resolver::resolve(registry, request, args);

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, expected_key);
}

TEST(KeyResolver, ResolveReturnsNulloptWhenNotFound) {
  MockRegistry registry;
  kernel::KeyRequest request{static_cast<Op>(1), DType::F32,
                             Architecture::CpuGeneric};
  kernel::KernelArgs args;

  auto result = resolver::resolve(registry, request, args);

  EXPECT_FALSE(result.has_value());
}

TEST(KeyResolver, ResolvePrefersRequestedArchitecture) {
  MockRegistry registry;
  kernel::KeyRequest request{static_cast<Op>(1), DType::F32,
                             Architecture::CpuZen4};

  kernel::FixedKeyComponents fixed{request.op, request.dtype};
  kernel::VariableKeyComponents specific{Architecture::CpuZen4,
                                         static_cast<kernel::Layout>(0),
                                         static_cast<kernel::Variant>(0)};
  kernel::VariableKeyComponents generic{Architecture::CpuGeneric,
                                        static_cast<kernel::Layout>(0),
                                        static_cast<kernel::Variant>(0)};

  auto specific_key = kernel::makeKey(fixed, specific);
  auto generic_key = kernel::makeKey(fixed, generic);

  registry.add(specific_key);
  registry.add(generic_key);

  kernel::KernelArgs args;
  auto result = resolver::resolve(registry, request, args);

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, specific_key);
}

TEST(KeyResolver, ResolveFallsBackToGeneric) {
  MockRegistry registry;
  kernel::KeyRequest request{static_cast<Op>(1), DType::F32,
                             Architecture::CpuZen4};

  kernel::FixedKeyComponents fixed{request.op, request.dtype};
  kernel::VariableKeyComponents generic{Architecture::CpuGeneric,
                                        static_cast<kernel::Layout>(0),
                                        static_cast<kernel::Variant>(0)};

  auto generic_key = kernel::makeKey(fixed, generic);
  registry.add(generic_key);

  kernel::KernelArgs args;
  auto result = resolver::resolve(registry, request, args);

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, generic_key);
}

TEST(KeyResolver, BuildContextGeneratesHierarchicalRules) {
  // Test Sm86 -> Sm80 -> CudaGeneric
  kernel::KeyRequest request{static_cast<Op>(1), DType::F32,
                             Architecture::CudaSm86};

  auto context = resolver::buildContext(request);

  ASSERT_EQ(context.rules.size(), 3u);
  EXPECT_EQ(context.rules[0].components.arch, Architecture::CudaSm86);
  EXPECT_EQ(context.rules[1].components.arch, Architecture::CudaSm80);
  EXPECT_EQ(context.rules[2].components.arch, Architecture::CudaGeneric);
}

TEST(KeyResolver, ResolveFollowsHierarchy) {
  MockRegistry registry;
  kernel::KeyRequest request{static_cast<Op>(1), DType::F32,
                             Architecture::CudaSm86};

  kernel::FixedKeyComponents fixed{request.op, request.dtype};

  // Only Sm80 kernel is available
  kernel::VariableKeyComponents sm80_comp{Architecture::CudaSm80,
                                          static_cast<kernel::Layout>(0),
                                          static_cast<kernel::Variant>(0)};
  auto sm80_key = kernel::makeKey(fixed, sm80_comp);
  registry.add(sm80_key);

  kernel::KernelArgs args;
  auto result = resolver::resolve(registry, request, args);

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, sm80_key);
}
