#if ORTEAF_ENABLE_CUDA

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <system_error>

#include <orteaf/extension/ops/tensor_ops.h>
#include <orteaf/extension/tensor/dense_tensor_impl.h>
#include <orteaf/internal/dtype/dtype.h>
#include <orteaf/internal/execution/cuda/platform/wrapper/cuda_device.h>
#include <orteaf/internal/execution/cuda/platform/wrapper/cuda_init.h>
#include <orteaf/internal/execution_context/cuda/current_context.h>
#include <orteaf/internal/execution/execution.h>
#include <orteaf/internal/init/library_init.h>
#include <orteaf/internal/storage/registry/storage_types.h>
#include <orteaf/user/tensor/tensor.h>

namespace ops = ::orteaf::extension::ops;
namespace tensor = ::orteaf::user::tensor;
namespace init = ::orteaf::internal::init;
namespace cuda_context = ::orteaf::internal::execution_context::cuda;
namespace cuda_wrapper = ::orteaf::internal::execution::cuda::platform::wrapper;

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

class CopyCudaTransferTest : public ::testing::Test {
protected:
  void SetUp() override {
    cuda_wrapper::cudaInit();
    if (cuda_wrapper::getDeviceCount() <= 0) {
      GTEST_SKIP() << "No CUDA devices available";
    }
    init::initialize();
  }

  void TearDown() override {
    cuda_context::reset();
    init::shutdown();
  }
};

TEST_F(CopyCudaTransferTest, CopiesHostToDeviceAndBackContiguous) {
  std::array<std::int64_t, 1> shape{8};
  auto host_src = makeDense(shape, DType::F32, Execution::Cpu);
  fillSequential(host_src, 1.0f);

  auto cuda_dst = makeDense(shape, DType::F32, Execution::Cuda);
  ops::TensorOps::copyHostToDevice(cuda_dst, host_src);

  auto host_out = makeDense(shape, DType::F32, Execution::Cpu);
  ops::TensorOps::copyDeviceToHost(host_out, cuda_dst);

  auto *src_data = getCpuF32Buffer(host_src);
  auto *out_data = getCpuF32Buffer(host_out);
  ASSERT_NE(src_data, nullptr);
  ASSERT_NE(out_data, nullptr);
  for (std::size_t i = 0; i < static_cast<std::size_t>(host_out.numel()); ++i) {
    EXPECT_FLOAT_EQ(out_data[i], src_data[i]);
  }
}

TEST_F(CopyCudaTransferTest, CopiesHostToDeviceIntoStridedView) {
  std::array<std::int64_t, 2> input_shape{2, 3};
  auto host_src = makeDense(input_shape, DType::F32, Execution::Cpu);
  fillSequential(host_src, 1.0f);

  std::array<std::int64_t, 2> base_shape{4, 4};
  auto cuda_base = makeDense(base_shape, DType::F32, Execution::Cuda);
  ops::TensorOps::fill(cuda_base, -1.0);

  std::array<std::int64_t, 2> starts{1, 0};
  std::array<std::int64_t, 2> sizes{2, 3};
  auto cuda_view = cuda_base.slice(starts, sizes);
  ASSERT_TRUE(cuda_view.valid());
  ASSERT_FALSE(cuda_view.isContiguous());

  ops::TensorOps::copyHostToDevice(cuda_view, host_src);

  auto host_base = makeDense(base_shape, DType::F32, Execution::Cpu);
  ops::TensorOps::copyDeviceToHost(host_base, cuda_base);
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

TEST_F(CopyCudaTransferTest, CopiesDeviceToHostFromStridedView) {
  std::array<std::int64_t, 2> base_shape{4, 4};
  auto host_src = makeDense(base_shape, DType::F32, Execution::Cpu);
  fillSequential(host_src, 0.0f);

  auto cuda_base = makeDense(base_shape, DType::F32, Execution::Cuda);
  ops::TensorOps::copyHostToDevice(cuda_base, host_src);

  std::array<std::int64_t, 2> starts{1, 0};
  std::array<std::int64_t, 2> sizes{2, 3};
  auto cuda_view = cuda_base.slice(starts, sizes);
  ASSERT_TRUE(cuda_view.valid());
  ASSERT_FALSE(cuda_view.isContiguous());

  std::array<std::int64_t, 2> out_shape{2, 3};
  auto host_out = makeDense(out_shape, DType::F32, Execution::Cpu);
  ops::TensorOps::copyDeviceToHost(host_out, cuda_view);

  auto *out = getCpuF32Buffer(host_out);
  ASSERT_NE(out, nullptr);
  EXPECT_FLOAT_EQ(out[0], 4.0f);
  EXPECT_FLOAT_EQ(out[1], 5.0f);
  EXPECT_FLOAT_EQ(out[2], 6.0f);
  EXPECT_FLOAT_EQ(out[3], 8.0f);
  EXPECT_FLOAT_EQ(out[4], 9.0f);
  EXPECT_FLOAT_EQ(out[5], 10.0f);
}

TEST_F(CopyCudaTransferTest, RejectsExecutionMismatch) {
  std::array<std::int64_t, 1> shape{4};
  auto cpu = makeDense(shape, DType::F32, Execution::Cpu);
  auto cuda = makeDense(shape, DType::F32, Execution::Cuda);

  EXPECT_THROW(ops::TensorOps::copyHostToDevice(cpu, cpu), std::system_error);
  EXPECT_THROW(ops::TensorOps::copyHostToDevice(cuda, cuda), std::system_error);
  EXPECT_THROW(ops::TensorOps::copyDeviceToHost(cuda, cuda), std::system_error);
  EXPECT_THROW(ops::TensorOps::copyDeviceToHost(cpu, cpu), std::system_error);
}

} // namespace

#endif // ORTEAF_ENABLE_CUDA
