#include "orteaf/internal/kernel/schema/kernel_param_schema.h"
#include "orteaf/internal/kernel/cpu/cpu_kernel_args.h"

#include <gtest/gtest.h>

namespace kernel = orteaf::internal::kernel;

// ============================================================
// CPU Kernel Parameter Schema Tests
// ============================================================

// Test that generic schema works with CpuKernelArgs
struct CpuTestSchema : kernel::ParamSchema<CpuTestSchema> {
  kernel::Field<kernel::ParamId::Alpha, float> alpha;
  kernel::Field<kernel::ParamId::Beta, float> beta;
  
  ORTEAF_EXTRACT_FIELDS(alpha, beta)
};

// Schema with optional field
struct SchemaWithOptional : kernel::ParamSchema<SchemaWithOptional> {
  kernel::Field<kernel::ParamId::Alpha, float> alpha;
  kernel::OptionalField<kernel::ParamId::Beta, float> beta{1.0f};
  
  ORTEAF_EXTRACT_FIELDS(alpha, beta)
};

// Shared schema that can be used across different kernel arg types
struct SharedSchema : kernel::ParamSchema<SharedSchema> {
  kernel::Field<kernel::ParamId::Epsilon, float> epsilon;
  kernel::Field<kernel::ParamId::Dim, std::size_t> dim;
  
  ORTEAF_EXTRACT_FIELDS(epsilon, dim)
};

// Test extracting directly from ParamList (no device manager needed)
TEST(CpuKernelParamSchemaTest, ExtractFromParamList) {
  kernel::ParamList params;
  params.pushBack(kernel::Param(kernel::ParamId::Alpha, 9.5f));
  params.pushBack(kernel::Param(kernel::ParamId::Beta, 10.5f));
  
  CpuTestSchema schema;
  kernel::detail::extractFieldsFromList(params, schema.alpha, schema.beta);
  
  EXPECT_FLOAT_EQ(schema.alpha, 9.5f);
  EXPECT_FLOAT_EQ(schema.beta, 10.5f);
}

// Test extracting with individual field calls
TEST(CpuKernelParamSchemaTest, DirectFieldExtraction) {
  kernel::ParamList params;
  params.pushBack(kernel::Param(kernel::ParamId::Epsilon, 1e-5f));
  params.pushBack(kernel::Param(kernel::ParamId::Dim, std::size_t{512}));
  
  kernel::Field<kernel::ParamId::Epsilon, float> epsilon;
  kernel::Field<kernel::ParamId::Dim, std::size_t> dim;
  
  epsilon.extract(params);
  dim.extract(params);
  
  EXPECT_FLOAT_EQ(epsilon, 1e-5f);
  EXPECT_EQ(dim, 512u);
}
