#include "orteaf/internal/kernel/schema/kernel_param_schema.h"
#include "orteaf/internal/kernel/core/kernel_args.h"

#include <orteaf/internal/base/array_view.h>
#include <orteaf/internal/base/inline_vector.h>
#include <orteaf/internal/kernel/param/transform/array_view_inline_vector.h>

#include <array>
#include <gtest/gtest.h>

namespace kernel = orteaf::internal::kernel;

// ============================================================
// CPU Kernel Parameter Schema Tests
// ============================================================

// Test that generic schema works with KernelArgs
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
TEST(KernelParamSchemaTest, ExtractFromParamList) {
  kernel::ParamList params;
  params.pushBack(kernel::Param(kernel::ParamId::Alpha, 9.5f));
  params.pushBack(kernel::Param(kernel::ParamId::Beta, 10.5f));
  
  CpuTestSchema schema;
  kernel::detail::extractFieldsFromList(params, schema.alpha, schema.beta);
  
  EXPECT_FLOAT_EQ(schema.alpha, 9.5f);
  EXPECT_FLOAT_EQ(schema.beta, 10.5f);
}

// Test extracting with individual field calls
TEST(KernelParamSchemaTest, DirectFieldExtraction) {
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

TEST(KernelParamSchemaTest, ExtractArrayViewToInlineVector) {
  using Dim = std::int64_t;
  using InlineDims = ::orteaf::internal::base::InlineVector<Dim, 4>;
  std::array<Dim, 3> dims = {Dim{1}, Dim{2}, Dim{4}};
  ::orteaf::internal::base::ArrayView<const Dim> view{dims.data(), dims.size()};
  kernel::ParamList params;
  params.pushBack(kernel::Param(kernel::ParamId::Shape, view));

  kernel::Field<kernel::ParamId::Shape, InlineDims> shape;
  shape.extract(params);

  EXPECT_EQ(shape.get().size, 3u);
  EXPECT_EQ(shape.get().data[0], 1);
  EXPECT_EQ(shape.get().data[1], 2);
  EXPECT_EQ(shape.get().data[2], 4);
}
