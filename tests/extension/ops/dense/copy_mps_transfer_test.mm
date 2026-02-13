#if ORTEAF_ENABLE_MPS

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <system_error>

#include <orteaf/extension/ops/tensor_ops.h>
#include <orteaf/extension/tensor/dense_tensor_impl.h>
#include <orteaf/internal/dtype/dtype.h>
#include <orteaf/internal/execution/execution.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_device.h>
#include <orteaf/internal/execution_context/mps/current_context.h>
#include <orteaf/internal/init/library_init.h>
#include <orteaf/internal/storage/registry/storage_types.h>
#include <orteaf/user/tensor/tensor.h>

namespace ops = ::orteaf::extension::ops;
namespace tensor = ::orteaf::user::tensor;
namespace init = ::orteaf::internal::init;
namespace mps_context = ::orteaf::internal::execution_context::mps;
namespace mps_wrapper = ::orteaf::internal::execution::mps::platform::wrapper;

using DType = ::orteaf::internal::DType;
using Execution = ::orteaf::internal::execution::Execution;
using DenseTensorImpl = ::orteaf::extension::tensor::DenseTensorImpl;
using CpuStorageLease = ::orteaf::internal::storage::CpuStorageLease;

namespace {

tensor::Tensor makeDense(std::span<const std::int64_t> shape, DType dtype,
                         Execution execution, std::size_t alignment = 0) {
  return tensor::Tensor::denseBuilder()
      .withShape(shape)
      .withDType(dtype)
      .withExecution(execution)
      .withAlignment(alignment)
      .build();
}

float *getCpuF32Buffer(tensor::Tensor &t) {
  auto *lease = t.tryAs<DenseTensorImpl>();
  if (!lease || !(*lease)) {
    return nullptr;
  }
  auto *impl = lease->operator->();
  if (impl == nullptr) {
    return nullptr;
  }
  auto *cpu_lease = impl->storageLease().tryAs<CpuStorageLease>();
  if (!cpu_lease || !(*cpu_lease)) {
    return nullptr;
  }
  auto *cpu_storage = cpu_lease->operator->();
  if (cpu_storage == nullptr) {
    return nullptr;
  }
  return static_cast<float *>(cpu_storage->buffer());
}

void fillSequential(tensor::Tensor &cpu_tensor, float start = 0.0f) {
  auto *data = getCpuF32Buffer(cpu_tensor);
  ASSERT_NE(data, nullptr);
  for (std::size_t i = 0; i < static_cast<std::size_t>(cpu_tensor.numel()); ++i) {
    data[i] = start + static_cast<float>(i);
  }
}

} // namespace

class CopyMpsTransferTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (mps_wrapper::getDeviceCount() <= 0) {
      GTEST_SKIP() << "No Metal devices available";
    }
    init::initialize();
  }

  void TearDown() override {
    mps_context::reset();
    init::shutdown();
  }
};

TEST_F(CopyMpsTransferTest, CopiesHostToMpsContiguous) {
  std::array<std::int64_t, 1> shape{8};
  auto host_src = makeDense(shape, DType::F32, Execution::Cpu);
  fillSequential(host_src, 1.0f);

  auto mps_dst = makeDense(shape, DType::F32, Execution::Mps);
  ops::TensorOps::copyHostToDevice(mps_dst, host_src);

  auto host_out = makeDense(shape, DType::F32, Execution::Cpu);
  ops::TensorOps::copyDeviceToHost(host_out, mps_dst);

  auto *src_data = getCpuF32Buffer(host_src);
  auto *out_data = getCpuF32Buffer(host_out);
  ASSERT_NE(src_data, nullptr);
  ASSERT_NE(out_data, nullptr);
  for (std::size_t i = 0; i < static_cast<std::size_t>(host_out.numel()); ++i) {
    EXPECT_FLOAT_EQ(out_data[i], src_data[i]);
  }
}

TEST_F(CopyMpsTransferTest, CopiesHostToDeviceAndBackContiguous) {
  std::array<std::int64_t, 1> shape{8};
  auto host_src = makeDense(shape, DType::F32, Execution::Cpu);
  fillSequential(host_src, 3.0f);

  auto mps_dst = makeDense(shape, DType::F32, Execution::Mps);
  ops::TensorOps::copyHostToDevice(mps_dst, host_src);

  auto host_out = makeDense(shape, DType::F32, Execution::Cpu);
  ops::TensorOps::copyDeviceToHost(host_out, mps_dst);

  auto *src_data = getCpuF32Buffer(host_src);
  auto *out_data = getCpuF32Buffer(host_out);
  ASSERT_NE(src_data, nullptr);
  ASSERT_NE(out_data, nullptr);
  for (std::size_t i = 0; i < static_cast<std::size_t>(host_out.numel()); ++i) {
    EXPECT_FLOAT_EQ(out_data[i], src_data[i]);
  }
}

TEST_F(CopyMpsTransferTest, CopiesHostToMpsIntoStridedView) {
  std::array<std::int64_t, 2> input_shape{2, 3};
  auto host_src = makeDense(input_shape, DType::F32, Execution::Cpu);
  fillSequential(host_src, 1.0f);

  std::array<std::int64_t, 2> base_shape{4, 4};
  auto mps_base = makeDense(base_shape, DType::F32, Execution::Mps);
  ops::TensorOps::fill(mps_base, -1.0);

  std::array<std::int64_t, 2> starts{1, 0};
  std::array<std::int64_t, 2> sizes{2, 3};
  auto mps_view = mps_base.slice(starts, sizes);
  ASSERT_TRUE(mps_view.valid());
  ASSERT_FALSE(mps_view.isContiguous());

  ops::TensorOps::copyHostToDevice(mps_view, host_src);

  auto host_base = makeDense(base_shape, DType::F32, Execution::Cpu);
  ops::TensorOps::copyDeviceToHost(host_base, mps_base);
  auto *base_data = getCpuF32Buffer(host_base);
  ASSERT_NE(base_data, nullptr);

  for (std::size_t i = 0; i < static_cast<std::size_t>(host_base.numel()); ++i) {
    if (i == 4u) EXPECT_FLOAT_EQ(base_data[i], 1.0f);
    if (i == 5u) EXPECT_FLOAT_EQ(base_data[i], 2.0f);
    if (i == 6u) EXPECT_FLOAT_EQ(base_data[i], 3.0f);
    if (i == 8u) EXPECT_FLOAT_EQ(base_data[i], 4.0f);
    if (i == 9u) EXPECT_FLOAT_EQ(base_data[i], 5.0f);
    if (i == 10u) EXPECT_FLOAT_EQ(base_data[i], 6.0f);
    if (i != 4u && i != 5u && i != 6u && i != 8u && i != 9u && i != 10u) {
      EXPECT_FLOAT_EQ(base_data[i], -1.0f);
    }
  }
}

TEST_F(CopyMpsTransferTest, CopiesMpsToHostFromStridedView) {
  std::array<std::int64_t, 2> base_shape{4, 4};
  auto host_src = makeDense(base_shape, DType::F32, Execution::Cpu);
  fillSequential(host_src, 0.0f);

  auto mps_base = makeDense(base_shape, DType::F32, Execution::Mps);
  ops::TensorOps::copyHostToDevice(mps_base, host_src);

  std::array<std::int64_t, 2> starts{1, 0};
  std::array<std::int64_t, 2> sizes{2, 3};
  auto mps_view = mps_base.slice(starts, sizes);
  ASSERT_TRUE(mps_view.valid());
  ASSERT_FALSE(mps_view.isContiguous());

  std::array<std::int64_t, 2> out_shape{2, 3};
  auto host_out = makeDense(out_shape, DType::F32, Execution::Cpu);
  ops::TensorOps::copyDeviceToHost(host_out, mps_view);

  auto *out = getCpuF32Buffer(host_out);
  ASSERT_NE(out, nullptr);
  EXPECT_FLOAT_EQ(out[0], 4.0f);
  EXPECT_FLOAT_EQ(out[1], 5.0f);
  EXPECT_FLOAT_EQ(out[2], 6.0f);
  EXPECT_FLOAT_EQ(out[3], 8.0f);
  EXPECT_FLOAT_EQ(out[4], 9.0f);
  EXPECT_FLOAT_EQ(out[5], 10.0f);
}

TEST_F(CopyMpsTransferTest, CopiesMpsToHostIntoStridedView) {
  std::array<std::int64_t, 2> src_shape{2, 3};
  auto host_src = makeDense(src_shape, DType::F32, Execution::Cpu);
  fillSequential(host_src, 10.0f);

  auto mps_src = makeDense(src_shape, DType::F32, Execution::Mps);
  ops::TensorOps::copyHostToDevice(mps_src, host_src);

  std::array<std::int64_t, 2> base_shape{4, 4};
  auto host_base = makeDense(base_shape, DType::F32, Execution::Cpu);
  auto *base_data = getCpuF32Buffer(host_base);
  ASSERT_NE(base_data, nullptr);
  for (std::size_t i = 0; i < static_cast<std::size_t>(host_base.numel()); ++i) {
    base_data[i] = -2.0f;
  }

  std::array<std::int64_t, 2> starts{1, 0};
  std::array<std::int64_t, 2> sizes{2, 3};
  auto host_view = host_base.slice(starts, sizes);
  ASSERT_TRUE(host_view.valid());
  ASSERT_FALSE(host_view.isContiguous());

  ops::TensorOps::copyDeviceToHost(host_view, mps_src);

  EXPECT_FLOAT_EQ(base_data[4], 10.0f);
  EXPECT_FLOAT_EQ(base_data[5], 11.0f);
  EXPECT_FLOAT_EQ(base_data[6], 12.0f);
  EXPECT_FLOAT_EQ(base_data[8], 13.0f);
  EXPECT_FLOAT_EQ(base_data[9], 14.0f);
  EXPECT_FLOAT_EQ(base_data[10], 15.0f);
  for (std::size_t i = 0; i < static_cast<std::size_t>(host_base.numel()); ++i) {
    if (i == 4u || i == 5u || i == 6u || i == 8u || i == 9u || i == 10u) {
      continue;
    }
    EXPECT_FLOAT_EQ(base_data[i], -2.0f);
  }
}

TEST_F(CopyMpsTransferTest, RejectsExecutionMismatch) {
  std::array<std::int64_t, 1> shape{4};
  auto cpu = makeDense(shape, DType::F32, Execution::Cpu);
  auto mps = makeDense(shape, DType::F32, Execution::Mps);

  EXPECT_THROW(ops::TensorOps::copyHostToDevice(cpu, cpu), std::system_error);
  EXPECT_THROW(ops::TensorOps::copyHostToDevice(mps, mps), std::system_error);
  EXPECT_THROW(ops::TensorOps::copyDeviceToHost(mps, mps), std::system_error);
  EXPECT_THROW(ops::TensorOps::copyDeviceToHost(cpu, cpu), std::system_error);
}

TEST_F(CopyMpsTransferTest, RejectsDTypeMismatch) {
  std::array<std::int64_t, 1> shape{4};
  auto cpu_i32 = makeDense(shape, DType::I32, Execution::Cpu);
  auto cpu_f32 = makeDense(shape, DType::F32, Execution::Cpu);
  auto mps_f32 = makeDense(shape, DType::F32, Execution::Mps);

  EXPECT_THROW(ops::TensorOps::copyHostToDevice(mps_f32, cpu_i32),
               std::system_error);
  EXPECT_THROW(ops::TensorOps::copyDeviceToHost(cpu_i32, mps_f32),
               std::system_error);
  EXPECT_NO_THROW(ops::TensorOps::copyHostToDevice(mps_f32, cpu_f32));
}

#endif // ORTEAF_ENABLE_MPS
