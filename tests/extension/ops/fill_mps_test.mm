#if ORTEAF_ENABLE_MPS

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include <orteaf/extension/ops/fill.h>
#include <orteaf/extension/tensor/dense_tensor_impl.h>
#include <orteaf/internal/dtype/dtype.h>
#include <orteaf/internal/execution/execution.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_buffer.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_command_buffer.h>
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
using MpsStorageLease = ::orteaf::internal::storage::MpsStorageLease;

namespace {

const float *getMpsBuffer(const tensor::Tensor &t) {
  auto *lease = t.tryAs<DenseTensorImpl>();
  if (!lease || !(*lease)) {
    return nullptr;
  }
  auto *impl = lease->operator->();
  if (impl == nullptr) {
    return nullptr;
  }
  auto *mps_lease = impl->storageLease().tryAs<MpsStorageLease>();
  if (!mps_lease || !(*mps_lease)) {
    return nullptr;
  }
  auto *storage = mps_lease->operator->();
  if (storage == nullptr || storage->buffer() == nullptr) {
    return nullptr;
  }
  return static_cast<const float *>(
      mps_wrapper::getBufferContentsConst(storage->buffer()));
}

void syncCurrentMpsQueue() {
  auto queue_lease = mps_context::currentCommandQueue();
  auto *resource = queue_lease.operator->();
  ASSERT_NE(resource, nullptr);
  ASSERT_NE(resource->queue(), nullptr);

  auto command_buffer = mps_wrapper::createCommandBuffer(resource->queue());
  ASSERT_NE(command_buffer, nullptr);
  mps_wrapper::commit(command_buffer);
  mps_wrapper::waitUntilCompleted(command_buffer);
  mps_wrapper::destroyCommandBuffer(command_buffer);
}

} // namespace

class FillMpsOpTest : public ::testing::Test {
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

TEST_F(FillMpsOpTest, FillsDenseTensorF32) {
  std::array<std::int64_t, 1> shape{8};
  auto t = tensor::Tensor::dense(shape, DType::F32, Execution::Mps);

  ops::fill(t, 3.5);
  syncCurrentMpsQueue();

  const float *data = getMpsBuffer(t);
  ASSERT_NE(data, nullptr);
  for (std::size_t i = 0; i < static_cast<std::size_t>(t.numel()); ++i) {
    EXPECT_FLOAT_EQ(data[i], 3.5f);
  }
}

TEST_F(FillMpsOpTest, FillsContiguousSliceViewF32) {
  std::array<std::int64_t, 1> shape{8};
  auto base = tensor::Tensor::dense(shape, DType::F32, Execution::Mps);

  ops::fill(base, -1.0);
  syncCurrentMpsQueue();

  std::array<std::int64_t, 1> starts{2};
  std::array<std::int64_t, 1> sizes{3};
  auto view = base.slice(starts, sizes);
  ASSERT_TRUE(view.valid());
  ASSERT_TRUE(view.isContiguous());

  ops::fill(view, 2.25);
  syncCurrentMpsQueue();

  const float *data = getMpsBuffer(base);
  ASSERT_NE(data, nullptr);
  for (std::size_t i = 0; i < static_cast<std::size_t>(base.numel()); ++i) {
    const bool should_fill = (i >= 2u && i < 5u);
    EXPECT_FLOAT_EQ(data[i], should_fill ? 2.25f : -1.0f);
  }
}

TEST_F(FillMpsOpTest, FillsNonContiguousSliceViewF32) {
  std::array<std::int64_t, 2> shape{4, 4};
  auto base = tensor::Tensor::dense(shape, DType::F32, Execution::Mps);

  ops::fill(base, -3.0);
  syncCurrentMpsQueue();

  std::array<std::int64_t, 2> starts{1, 0};
  std::array<std::int64_t, 2> sizes{2, 3};
  auto view = base.slice(starts, sizes);
  ASSERT_TRUE(view.valid());
  ASSERT_FALSE(view.isContiguous());

  ops::fill(view, 7.0);
  syncCurrentMpsQueue();

  const float *data = getMpsBuffer(base);
  ASSERT_NE(data, nullptr);
  for (std::size_t i = 0; i < static_cast<std::size_t>(base.numel()); ++i) {
    const bool should_fill =
        (i == 4u || i == 5u || i == 6u || i == 8u || i == 9u || i == 10u);
    EXPECT_FLOAT_EQ(data[i], should_fill ? 7.0f : -3.0f);
  }
}

#endif // ORTEAF_ENABLE_MPS
