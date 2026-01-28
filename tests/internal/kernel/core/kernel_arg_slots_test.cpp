#include "orteaf/internal/kernel/core/kernel_arg_slots.h"

#include <orteaf/internal/base/array_view.h>
#include <orteaf/internal/base/small_vector.h>
#include <orteaf/internal/kernel/core/kernel_args.h>
#include <orteaf/internal/kernel/param/param_id.h>
#include <orteaf/internal/kernel/param/transform/small_vector_array_view.h>
#include <orteaf/internal/kernel/storage/storage_binding.h>
#include <orteaf/internal/kernel/storage/storage_id.h>
#include <orteaf/internal/kernel/storage/storage_key.h>
#include <orteaf/internal/kernel/storage/storage_role.h>
#include <orteaf/internal/storage/storage_lease.h>

#include <gtest/gtest.h>

namespace kernel = ::orteaf::internal::kernel;

TEST(ParamSlot, BindGlobal) {
  kernel::KernelArgs args;
  kernel::ParamSlot<float, kernel::ParamId::Alpha> slot(1.5f);

  slot.bindGlobal(args);

  const auto *param = args.findParam(kernel::ParamId::Alpha);
  ASSERT_NE(param, nullptr);
  EXPECT_FALSE(param->key().storage.has_value());
  EXPECT_FLOAT_EQ(*param->tryGet<float>(), 1.5f);
}

TEST(ParamSlot, BindScoped) {
  kernel::KernelArgs args;
  kernel::ParamSlot<float, kernel::ParamId::Alpha, kernel::StorageRole::Index>
      slot(2.5f);

  slot.bindScoped(args, kernel::StorageId::Input0);

  EXPECT_EQ(args.findParam(kernel::ParamId::Alpha), nullptr);

  const auto key = kernel::ParamKey::scoped(
      kernel::ParamId::Alpha,
      kernel::makeStorageKey(kernel::StorageId::Input0,
                             kernel::StorageRole::Index));
  const auto *param = args.findParam(key);
  ASSERT_NE(param, nullptr);
  ASSERT_TRUE(param->key().storage.has_value());
  EXPECT_EQ(*param->key().storage,
            kernel::makeStorageKey(kernel::StorageId::Input0,
                                   kernel::StorageRole::Index));
  EXPECT_FLOAT_EQ(*param->tryGet<float>(), 2.5f);
}

TEST(ParamSlot, ToParam) {
  kernel::ParamSlot<float, kernel::ParamId::Beta> global_slot(3.0f);
  kernel::Param expected_global(kernel::ParamKey::global(kernel::ParamId::Beta),
                                3.0f);
  EXPECT_EQ(global_slot.toGlobalParam(), expected_global);

  kernel::ParamSlot<float, kernel::ParamId::Beta, kernel::StorageRole::Data>
      scoped_slot(4.0f);
  const auto key = kernel::ParamKey::scoped(
      kernel::ParamId::Beta,
      kernel::makeStorageKey(kernel::StorageId::Output,
                             kernel::StorageRole::Data));
  kernel::Param expected_scoped(key, 4.0f);
  EXPECT_EQ(scoped_slot.toScopedParam(kernel::StorageId::Output),
            expected_scoped);
}

TEST(ParamSlot, BindGlobalSmallVectorToArrayView) {
  using Dim = std::int64_t;
  using Dims = ::orteaf::internal::base::SmallVector<Dim, 4>;
  kernel::KernelArgs args;
  Dims dims{Dim{2}, Dim{3}, Dim{5}};
  kernel::ParamSlot<Dims, kernel::ParamId::Shape> slot(dims);

  slot.bindGlobal(args);

  const auto *param = args.findParam(kernel::ParamId::Shape);
  ASSERT_NE(param, nullptr);
  const auto *view =
      param->tryGet<::orteaf::internal::base::ArrayView<const Dim>>();
  ASSERT_NE(view, nullptr);
  EXPECT_EQ(view->count, 3u);
  ASSERT_NE(view->data, nullptr);
  EXPECT_EQ(view->data[0], 2);
  EXPECT_EQ(view->data[1], 3);
  EXPECT_EQ(view->data[2], 5);
}

TEST(StorageSlot, Bind) {
  kernel::KernelArgs args;
  ::orteaf::internal::storage::StorageLease lease;
  kernel::StorageSlot<kernel::StorageRole::Index> slot(std::move(lease));

  slot.bind(args, kernel::StorageId::Input0);

  EXPECT_EQ(args.storageCount(), 1u);
  const auto *binding =
      args.findStorage(kernel::makeStorageKey(kernel::StorageId::Input0,
                                              kernel::StorageRole::Index));
  ASSERT_NE(binding, nullptr);
  EXPECT_EQ(binding->key.id, kernel::StorageId::Input0);
  EXPECT_EQ(binding->key.role, kernel::StorageRole::Index);
}
