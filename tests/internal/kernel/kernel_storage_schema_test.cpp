#include <orteaf/internal/kernel/schema/kernel_storage_schema.h>
#include <orteaf/internal/kernel/storage/operand.h>
#include <orteaf/internal/storage/storage_lease.h>

#include <gtest/gtest.h>

#include <orteaf/internal/kernel/core/kernel_args.h>
#include <orteaf/internal/kernel/storage/operand_id.h>

#include <orteaf/internal/execution/cpu/api/cpu_execution_api.h>
#include <orteaf/internal/execution_context/cpu/current_context.h>

namespace orteaf::internal::kernel {
namespace {

using ::orteaf::internal::kernel::KernelArgs;

// Test fixture to set up CPU device manager
class KernelStorageSchemaTest : public ::testing::Test {
protected:
  void SetUp() override {
    namespace cpu_api = ::orteaf::internal::execution::cpu::api;
    cpu_api::CpuExecutionApi::ExecutionManager::Config config{};
    cpu_api::CpuExecutionApi::configure(config);
    ::orteaf::internal::execution_context::cpu::reset();
  }

  void TearDown() override {
    namespace cpu_api = ::orteaf::internal::execution::cpu::api;
    ::orteaf::internal::execution_context::cpu::reset();
    cpu_api::CpuExecutionApi::shutdown();
  }
};

// Define storage schema outside test function
struct SimpleStorageSchema : StorageSchema<SimpleStorageSchema> {
  OptionalStorageField<OperandId::Input0> input;
  OptionalStorageField<OperandId::Output> output;

  ORTEAF_EXTRACT_STORAGES(input, output)
};

TEST_F(KernelStorageSchemaTest, BasicExtraction) {
  KernelArgs args;

  // Extract from empty args
  auto schema = SimpleStorageSchema::extract(args);

  EXPECT_FALSE(schema.input);
  EXPECT_FALSE(schema.output);
  EXPECT_FALSE(schema.input.present());
  EXPECT_FALSE(schema.output.present());
}

struct OptionalStorageSchema : StorageSchema<OptionalStorageSchema> {
  OptionalStorageField<OperandId::Input0> input;
  OptionalStorageField<OperandId::Output> output;
  OptionalStorageField<OperandId::Workspace> workspace;

  ORTEAF_EXTRACT_STORAGES(input, output, workspace)
};

TEST_F(KernelStorageSchemaTest, OptionalStorageField) {
  KernelArgs args;

  auto schema = OptionalStorageSchema::extract(args);

  EXPECT_FALSE(schema.input);
  EXPECT_FALSE(schema.output);
  EXPECT_FALSE(schema.workspace);

  EXPECT_FALSE(schema.output.present());
  EXPECT_FALSE(schema.workspace.present());

  // Optional field returns nullptr when not present
  using AnyBinding = Operand;
  auto *workspace_binding = schema.workspace.bindingOr<AnyBinding>();
  EXPECT_EQ(workspace_binding, nullptr);
}

struct RequiredStorageSchema : StorageSchema<RequiredStorageSchema> {
  StorageField<OperandId::Input0> input;
  StorageField<OperandId::Output> output;

  ORTEAF_EXTRACT_STORAGES(input, output)
};

TEST_F(KernelStorageSchemaTest, MissingRequiredStorage) {
  KernelArgs args;

  EXPECT_THROW(RequiredStorageSchema::extract(args), std::runtime_error);
}

TEST_F(KernelStorageSchemaTest, OptionalFieldNotPresent) {
  KernelArgs args;

  // Single field extraction
  OptionalStorageField<OperandId::Workspace> workspace_field;
  workspace_field.extract(args);

  EXPECT_FALSE(workspace_field);
  EXPECT_FALSE(workspace_field.present());
}

} // namespace
} // namespace orteaf::internal::kernel
