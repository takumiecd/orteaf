#if ORTEAF_ENABLE_CUDA

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <system_error>
#include <vector>

#include <orteaf/extension/ops/tensor_ops.h>
#include <orteaf/extension/tensor/dense_tensor_impl.h>
#include <orteaf/internal/dtype/dtype.h>
#include <orteaf/internal/execution/cuda/platform/wrapper/cuda_alloc.h>
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
using CudaStorageLease = ::orteaf::internal::storage::CudaStorageLease;

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

bool copyToHost(const tensor::Tensor &tensor_ref, std::vector<float> &host) {
  auto *lease = tensor_ref.tryAs<DenseTensorImpl>();
  if (!lease || !(*lease)) {
    return false;
  }

  auto *impl = lease->operator->();
  if (impl == nullptr) {
    return false;
  }

  auto *cuda_lease = impl->storageLease().tryAs<CudaStorageLease>();
  if (!cuda_lease || !(*cuda_lease)) {
    return false;
  }

  auto *storage = cuda_lease->operator->();
  if (storage == nullptr) {
    return false;
  }

  auto view = storage->bufferView();
  if (!view) {
    return false;
  }

  host.resize(storage->numel());
  if (host.empty()) {
    return true;
  }

  cuda_wrapper::copyToHost(view.data(), host.data(),
                           host.size() * sizeof(float));
  return true;
}

class FillCudaOpTest : public ::testing::Test {
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

TEST_F(FillCudaOpTest, FillsDenseTensorF32) {
  std::array<std::int64_t, 1> shape{8};
  auto tensor_ref = makeDense(shape, DType::F32, Execution::Cuda);

  ops::TensorOps::fill(tensor_ref, 3.5);

  std::vector<float> host;
  ASSERT_TRUE(copyToHost(tensor_ref, host));
  ASSERT_EQ(host.size(), static_cast<std::size_t>(tensor_ref.numel()));
  for (float v : host) {
    EXPECT_FLOAT_EQ(v, 3.5f);
  }
}

TEST_F(FillCudaOpTest, FillsContiguousSliceViewF32) {
  std::array<std::int64_t, 1> shape{8};
  auto base = makeDense(shape, DType::F32, Execution::Cuda);

  ops::TensorOps::fill(base, -1.0);

  std::array<std::int64_t, 1> starts{2};
  std::array<std::int64_t, 1> sizes{3};
  auto view = base.slice(starts, sizes);
  ASSERT_TRUE(view.valid());
  ASSERT_TRUE(view.isContiguous());

  ops::TensorOps::fill(view, 2.25);

  std::vector<float> host;
  ASSERT_TRUE(copyToHost(base, host));
  ASSERT_EQ(host.size(), static_cast<std::size_t>(base.numel()));
  for (std::size_t i = 0; i < host.size(); ++i) {
    const bool should_fill = (i >= 2u && i < 5u);
    EXPECT_FLOAT_EQ(host[i], should_fill ? 2.25f : -1.0f);
  }
}

TEST_F(FillCudaOpTest, FillsNonContiguousSliceViewF32) {
  std::array<std::int64_t, 2> shape{4, 4};
  auto base = makeDense(shape, DType::F32, Execution::Cuda);

  ops::TensorOps::fill(base, -3.0);

  std::array<std::int64_t, 2> starts{1, 0};
  std::array<std::int64_t, 2> sizes{2, 3};
  auto view = base.slice(starts, sizes);
  ASSERT_TRUE(view.valid());
  ASSERT_FALSE(view.isContiguous());

  ops::TensorOps::fill(view, 7.0);

  std::vector<float> host;
  ASSERT_TRUE(copyToHost(base, host));
  ASSERT_EQ(host.size(), static_cast<std::size_t>(base.numel()));
  for (std::size_t i = 0; i < host.size(); ++i) {
    const bool should_fill =
        (i == 4u || i == 5u || i == 6u || i == 8u || i == 9u || i == 10u);
    EXPECT_FLOAT_EQ(host[i], should_fill ? 7.0f : -3.0f);
  }
}

TEST_F(FillCudaOpTest, RejectsNonF32DType) {
  std::array<std::int64_t, 1> shape{4};
  auto tensor_ref = makeDense(shape, DType::I32, Execution::Cuda);

  EXPECT_THROW(ops::TensorOps::fill(tensor_ref, 1.0), std::system_error);
}

} // namespace

#endif // ORTEAF_ENABLE_CUDA
