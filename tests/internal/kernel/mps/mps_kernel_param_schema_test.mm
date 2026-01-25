#include "orteaf/internal/kernel/schema/kernel_param_schema.h"

#include <gtest/gtest.h>

#include "orteaf/internal/execution/mps/api/mps_execution_api.h"
#include "orteaf/internal/execution/mps/platform/mps_slow_ops.h"
#include "orteaf/internal/execution_context/mps/current_context.h"
#include "orteaf/internal/kernel/mps/mps_kernel_args.h"
#include "orteaf/internal/kernel/param/param.h"
#include "orteaf/internal/kernel/param/param_id.h"

namespace kernel = orteaf::internal::kernel;
namespace mps_kernel = orteaf::internal::kernel::mps;

// ============================================================
// Test Fixture
// ============================================================

class MpsKernelParamSchemaTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Configure MPS execution API
    namespace mps_api = ::orteaf::internal::execution::mps::api;
    namespace mps_platform = ::orteaf::internal::execution::mps::platform;

    auto *ops = new mps_platform::MpsSlowOpsImpl();
    const int device_count = ops->getDeviceCount();
    if (device_count <= 0) {
      delete ops;
      GTEST_SKIP() << "No MPS devices available";
    }

    mps_api::MpsExecutionApi::ExecutionManager::Config config{};
    config.slow_ops = ops;

    const auto capacity = static_cast<std::size_t>(device_count);
    auto &device_cfg = config.device_config;
    device_cfg.control_block_capacity = capacity;
    device_cfg.control_block_block_size = capacity;
    device_cfg.control_block_growth_chunk_size = 1;
    device_cfg.payload_capacity = capacity;
    device_cfg.payload_block_size = capacity;
    device_cfg.payload_growth_chunk_size = 1;

    auto configure_pool = [](auto &cfg) {
      cfg.control_block_capacity = 1;
      cfg.control_block_block_size = 1;
      cfg.control_block_growth_chunk_size = 1;
      cfg.payload_capacity = 1;
      cfg.payload_block_size = 1;
      cfg.payload_growth_chunk_size = 1;
    };
    configure_pool(device_cfg.command_queue_config);
    configure_pool(device_cfg.event_config);
    configure_pool(device_cfg.fence_config);
    configure_pool(device_cfg.heap_config);
    configure_pool(device_cfg.library_config);
    configure_pool(device_cfg.graph_config);

    try {
      mps_api::MpsExecutionApi::configure(config);
    } catch (const std::exception &ex) {
      delete ops;
      GTEST_SKIP() << "Failed to configure MPS: " << ex.what();
    }
    ::orteaf::internal::execution_context::mps::reset();
  }

  void TearDown() override {
    // Cleanup
    namespace mps_api = ::orteaf::internal::execution::mps::api;
    ::orteaf::internal::execution_context::mps::reset();
    mps_api::MpsExecutionApi::shutdown();
  }
};

// ============================================================
// Basic Field Tests
// ============================================================

TEST_F(MpsKernelParamSchemaTest, FieldDefaultConstruction) {
  kernel::Field<kernel::ParamId::Alpha, float> field;
  EXPECT_FLOAT_EQ(field.value, 0.0f);
  EXPECT_FLOAT_EQ(field.get(), 0.0f);
}

TEST_F(MpsKernelParamSchemaTest, FieldImplicitConversion) {
  kernel::Field<kernel::ParamId::Alpha, float> field;
  field.value = 3.14f;

  float value = field; // Implicit conversion
  EXPECT_FLOAT_EQ(value, 3.14f);
}

TEST_F(MpsKernelParamSchemaTest, FieldExplicitGet) {
  kernel::Field<kernel::ParamId::Beta, float> field;
  field.value = 2.71f;

  EXPECT_FLOAT_EQ(field.get(), 2.71f);

  field.get() = 1.5f;
  EXPECT_FLOAT_EQ(field.value, 1.5f);
}

TEST_F(MpsKernelParamSchemaTest, FieldExtractSuccess) {
  mps_kernel::MpsKernelArgs args;
  args.addParam(kernel::Param(kernel::ParamId::Alpha, 5.0f));

  kernel::Field<kernel::ParamId::Alpha, float> field;
  field.extract(args);

  EXPECT_FLOAT_EQ(field.value, 5.0f);
}

TEST_F(MpsKernelParamSchemaTest, FieldExtractMissingThrows) {
  mps_kernel::MpsKernelArgs args;

  kernel::Field<kernel::ParamId::Alpha, float> field;
  EXPECT_THROW(field.extract(args), std::runtime_error);
}

TEST_F(MpsKernelParamSchemaTest, FieldExtractTypeMismatchThrows) {
  mps_kernel::MpsKernelArgs args;
  args.addParam(kernel::Param(kernel::ParamId::Alpha, 42)); // int, not float

  kernel::Field<kernel::ParamId::Alpha, float> field;
  EXPECT_THROW(field.extract(args), std::runtime_error);
}

// ============================================================
// OptionalField Tests
// ============================================================

TEST_F(MpsKernelParamSchemaTest, OptionalFieldDefaultConstruction) {
  kernel::OptionalField<kernel::ParamId::Alpha, float> field;
  EXPECT_FLOAT_EQ(field.value, 0.0f);
  EXPECT_FALSE(field.present);
}

TEST_F(MpsKernelParamSchemaTest, OptionalFieldWithDefaultValue) {
  kernel::OptionalField<kernel::ParamId::Scale, double> field(1.0);
  EXPECT_DOUBLE_EQ(field.value, 1.0);
  EXPECT_FALSE(field.present);
}

TEST_F(MpsKernelParamSchemaTest, OptionalFieldExtractSuccess) {
  mps_kernel::MpsKernelArgs args;
  args.addParam(kernel::Param(kernel::ParamId::Epsilon, 0.001f));

  kernel::OptionalField<kernel::ParamId::Epsilon, float> field;
  field.extract(args);

  EXPECT_FLOAT_EQ(field.value, 0.001f);
  EXPECT_TRUE(field.present);
  EXPECT_TRUE(static_cast<bool>(field));
}

TEST_F(MpsKernelParamSchemaTest, OptionalFieldExtractMissingDoesNotThrow) {
  mps_kernel::MpsKernelArgs args;

  kernel::OptionalField<kernel::ParamId::Epsilon, float> field(1e-5f);
  field.extract(args);

  EXPECT_FLOAT_EQ(field.value, 1e-5f); // Keeps default value
  EXPECT_FALSE(field.present);
  EXPECT_FALSE(static_cast<bool>(field));
}

TEST_F(MpsKernelParamSchemaTest, OptionalFieldValueOr) {
  kernel::OptionalField<kernel::ParamId::Scale, double> field;

  // Not present
  EXPECT_DOUBLE_EQ(field.valueOr(2.0), 2.0);

  // Present
  field.value = 3.5;
  field.present = true;
  EXPECT_DOUBLE_EQ(field.valueOr(2.0), 3.5);
}

// ============================================================
// Schema Tests
// ============================================================

// Simple schema definition
struct SimpleSchema : kernel::ParamSchema<SimpleSchema> {
  kernel::Field<kernel::ParamId::Alpha, float> alpha;
  kernel::Field<kernel::ParamId::Beta, float> beta;
  kernel::Field<kernel::ParamId::Dim, std::size_t> dim;

  ORTEAF_EXTRACT_FIELDS(alpha, beta, dim)
};

TEST_F(MpsKernelParamSchemaTest, SimpleSchemaExtract) {
  mps_kernel::MpsKernelArgs args;
  args.addParam(kernel::Param(kernel::ParamId::Alpha, 1.5f));
  args.addParam(kernel::Param(kernel::ParamId::Beta, 2.5f));
  args.addParam(kernel::Param(kernel::ParamId::Dim, std::size_t{128}));

  auto schema = SimpleSchema::extract(args);

  EXPECT_FLOAT_EQ(schema.alpha, 1.5f);
  EXPECT_FLOAT_EQ(schema.beta, 2.5f);
  EXPECT_EQ(schema.dim, 128u);
}

TEST_F(MpsKernelParamSchemaTest, SimpleSchemaImplicitConversion) {
  mps_kernel::MpsKernelArgs args;
  args.addParam(kernel::Param(kernel::ParamId::Alpha, 3.14f));
  args.addParam(kernel::Param(kernel::ParamId::Beta, 2.71f));
  args.addParam(kernel::Param(kernel::ParamId::Dim, std::size_t{256}));

  auto schema = SimpleSchema::extract(args);

  // Implicit conversion to underlying types
  float alpha_val = schema.alpha;
  float beta_val = schema.beta;
  std::size_t dim_val = schema.dim;

  EXPECT_FLOAT_EQ(alpha_val, 3.14f);
  EXPECT_FLOAT_EQ(beta_val, 2.71f);
  EXPECT_EQ(dim_val, 256u);
}

TEST_F(MpsKernelParamSchemaTest, SchemaMissingParamThrows) {
  mps_kernel::MpsKernelArgs args;
  args.addParam(kernel::Param(kernel::ParamId::Alpha, 1.0f));
  // Missing Beta and Dim

  EXPECT_THROW(SimpleSchema::extract(args), std::runtime_error);
}

// Schema with optional fields
struct MixedSchema : kernel::ParamSchema<MixedSchema> {
  kernel::Field<kernel::ParamId::Epsilon, float> epsilon;
  kernel::Field<kernel::ParamId::Axis, int> axis;
  kernel::OptionalField<kernel::ParamId::Scale, double> scale{1.0};

  ORTEAF_EXTRACT_FIELDS(epsilon, axis, scale)
};

TEST_F(MpsKernelParamSchemaTest, MixedSchemaAllPresent) {
  mps_kernel::MpsKernelArgs args;
  args.addParam(kernel::Param(kernel::ParamId::Epsilon, 1e-5f));
  args.addParam(kernel::Param(kernel::ParamId::Axis, -1));
  args.addParam(kernel::Param(kernel::ParamId::Scale, 2.0));

  auto schema = MixedSchema::extract(args);

  EXPECT_FLOAT_EQ(schema.epsilon, 1e-5f);
  EXPECT_EQ(schema.axis, -1);
  EXPECT_DOUBLE_EQ(schema.scale.value, 2.0);
  EXPECT_TRUE(schema.scale.present);
}

TEST_F(MpsKernelParamSchemaTest, MixedSchemaOptionalMissing) {
  mps_kernel::MpsKernelArgs args;
  args.addParam(kernel::Param(kernel::ParamId::Epsilon, 1e-5f));
  args.addParam(kernel::Param(kernel::ParamId::Axis, 0));

  auto schema = MixedSchema::extract(args);

  EXPECT_FLOAT_EQ(schema.epsilon, 1e-5f);
  EXPECT_EQ(schema.axis, 0);
  EXPECT_DOUBLE_EQ(schema.scale.value, 1.0); // Default value
  EXPECT_FALSE(schema.scale.present);
}

// Schema with various types
struct MultiTypeSchema : kernel::ParamSchema<MultiTypeSchema> {
  kernel::Field<kernel::ParamId::Alpha, float> alpha;
  kernel::Field<kernel::ParamId::Scale, double> scale;
  kernel::Field<kernel::ParamId::Count, std::size_t> count;
  kernel::Field<kernel::ParamId::Axis, int> axis;

  ORTEAF_EXTRACT_FIELDS(alpha, scale, count, axis)
};

TEST_F(MpsKernelParamSchemaTest, MultiTypeSchemaExtract) {
  mps_kernel::MpsKernelArgs args;
  args.addParam(kernel::Param(kernel::ParamId::Alpha, 0.5f));
  args.addParam(kernel::Param(kernel::ParamId::Scale, 3.14159));
  args.addParam(kernel::Param(kernel::ParamId::Count, std::size_t{1024}));
  args.addParam(kernel::Param(kernel::ParamId::Axis, 2));

  auto schema = MultiTypeSchema::extract(args);

  EXPECT_FLOAT_EQ(schema.alpha, 0.5f);
  EXPECT_DOUBLE_EQ(schema.scale, 3.14159);
  EXPECT_EQ(schema.count, 1024u);
  EXPECT_EQ(schema.axis, 2);
}

// ============================================================
// Practical Usage Tests
// ============================================================

// Simulate a real kernel parameter schema
struct NormalizationKernelParams : kernel::ParamSchema<NormalizationKernelParams> {
  kernel::Field<kernel::ParamId::Epsilon, float> epsilon;
  kernel::Field<kernel::ParamId::Axis, int> axis;
  kernel::Field<kernel::ParamId::Dim, std::size_t> dim;
  kernel::OptionalField<kernel::ParamId::Scale, double> scale{1.0};
  kernel::OptionalField<kernel::ParamId::Beta, float> beta{0.0f};

  ORTEAF_EXTRACT_FIELDS(epsilon, axis, dim, scale, beta)
};

TEST_F(MpsKernelParamSchemaTest, PracticalNormalizationKernel) {
  mps_kernel::MpsKernelArgs args;
  args.addParam(kernel::Param(kernel::ParamId::Epsilon, 1e-5f));
  args.addParam(kernel::Param(kernel::ParamId::Axis, -1));
  args.addParam(kernel::Param(kernel::ParamId::Dim, std::size_t{512}));
  args.addParam(kernel::Param(kernel::ParamId::Scale, 2.0));

  auto params = NormalizationKernelParams::extract(args);

  // Use parameters in kernel logic
  float eps = params.epsilon;
  int axis = params.axis;
  std::size_t dim = params.dim;
  double scale = params.scale.valueOr(1.0);
  float beta = params.beta.valueOr(0.0f);

  EXPECT_FLOAT_EQ(eps, 1e-5f);
  EXPECT_EQ(axis, -1);
  EXPECT_EQ(dim, 512u);
  EXPECT_DOUBLE_EQ(scale, 2.0);
  EXPECT_FLOAT_EQ(beta, 0.0f); // Not provided, uses default

  // Check presence
  EXPECT_TRUE(params.scale.present);
  EXPECT_FALSE(params.beta.present);
}

TEST_F(MpsKernelParamSchemaTest, PracticalConditionalBehavior) {
  mps_kernel::MpsKernelArgs args;
  args.addParam(kernel::Param(kernel::ParamId::Epsilon, 1e-5f));
  args.addParam(kernel::Param(kernel::ParamId::Axis, 0));
  args.addParam(kernel::Param(kernel::ParamId::Dim, std::size_t{256}));
  args.addParam(kernel::Param(kernel::ParamId::Beta, 0.5f));

  auto params = NormalizationKernelParams::extract(args);

  // Conditional logic based on optional parameters
  if (params.scale) {
    // Apply scaling
    FAIL() << "Scale should not be present";
  }

  if (params.beta) {
    // Apply beta offset
    EXPECT_FLOAT_EQ(params.beta.value, 0.5f);
  } else {
    FAIL() << "Beta should be present";
  }
}
