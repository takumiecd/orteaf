#if ORTEAF_ENABLE_MPS

#include <cstddef>
#include <cstdint>
#include <limits>
#include <system_error>

#include <orteaf/internal/architecture/architecture.h>
#include <orteaf/internal/diagnostics/error/error.h>
#include <orteaf/internal/dtype/dtype.h>
#include <orteaf/internal/execution/mps/api/mps_execution_api.h>
#include <orteaf/internal/kernel/api/kernel_registry_api.h>
#include <orteaf/internal/kernel/core/kernel_args.h>
#include <orteaf/internal/kernel/core/kernel_entry.h>
#include <orteaf/internal/kernel/core/kernel_key.h>
#include <orteaf/internal/kernel/core/kernel_metadata.h>
#include <orteaf/internal/kernel/mps/mps_kernel_session.h>
#include <orteaf/internal/kernel/param/param.h>
#include <orteaf/internal/kernel/param/param_id.h>
#include <orteaf/internal/kernel/param/transform/array_view_inline_vector.h>
#include <orteaf/internal/kernel/registry/kernel_auto_registry.h>
#include <orteaf/internal/kernel/schema/kernel_param_schema.h>
#include <orteaf/internal/kernel/schema/kernel_storage_schema.h>
#include <orteaf/internal/kernel/storage/operand_id.h>
#include <orteaf/internal/storage/registry/storage_types.h>

#include "../common/layout_common.h"

namespace orteaf::extension::kernel::mps {

namespace kernel = ::orteaf::internal::kernel;
namespace error = ::orteaf::internal::diagnostics::error;
namespace mps_kernel = ::orteaf::internal::kernel::mps;
namespace common_layout = ::orteaf::extension::kernel::common::layout;

struct FillMpsStorages : kernel::StorageSchema<FillMpsStorages> {
  kernel::StorageField<kernel::OperandId::Output> output;

  ORTEAF_EXTRACT_STORAGES(output)
};

struct FillMpsParams : kernel::ParamSchema<FillMpsParams> {
  using ShapeVector = common_layout::ShapeVector;

  kernel::Field<kernel::ParamId::Value, double> value;
  kernel::ScopedField<kernel::ParamId::Shape, ShapeVector,
                      kernel::OperandId::Output>
      shape;
  kernel::ScopedField<kernel::ParamId::Strides, ShapeVector,
                      kernel::OperandId::Output>
      strides;
  kernel::ScopedField<kernel::ParamId::Offset, std::int64_t,
                      kernel::OperandId::Output>
      offset;

  ORTEAF_EXTRACT_FIELDS(value, shape, strides, offset)
};

void fillMpsExecute(
    ::orteaf::internal::execution::mps::resource::MpsKernelBase &base,
    kernel::KernelArgs &args) {
  auto storages = FillMpsStorages::extract(args);
  auto params = FillMpsParams::extract(args);

  using AnyBinding = kernel::KernelArgs::StorageListType::Storage::value_type;
  auto &lease = storages.output.lease<AnyBinding>();
  auto &storage = storages.output.payloadAs<
      AnyBinding, ::orteaf::internal::storage::MpsStorageLease>(
      "MPS fill kernel requires MPS output storage",
      "MPS fill kernel output buffer is unavailable",
      [](const auto &typed_storage) { return typed_storage.buffer() != nullptr; });

  if (lease.dtype() != ::orteaf::internal::DType::F32 ||
      storage.dtype() != ::orteaf::internal::DType::F32) {
    error::throwError(error::OrteafErrc::Unsupported,
                      "MPS fill kernel supports only F32");
  }
  const auto storage_numel_raw = storage.numel();
  if (storage_numel_raw >
      static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "MPS fill kernel storage size exceeds index range");
  }
  const auto storage_numel = static_cast<std::int64_t>(storage_numel_raw);

  const auto raw_offset = params.offset.get();
  if (raw_offset < 0) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "MPS fill kernel received negative offset");
  }

  const auto &shape = params.shape.get();
  const auto &strides = params.strides.get();
  const auto layout = common_layout::analyzeLayout(
      shape, strides, raw_offset, "MPS fill kernel", "output");
  if (layout.has_zero) {
    return;
  }

  if (layout.min_index < 0 || layout.max_index < 0 ||
      layout.max_index >= storage_numel) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "MPS fill kernel view exceeds output storage bounds");
  }

  const auto offset = static_cast<std::size_t>(raw_offset);
  if (layout.numel > static_cast<std::size_t>(
                        std::numeric_limits<std::uint32_t>::max()) ||
      offset >
          static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) ||
      static_cast<std::size_t>(layout.max_index) >
          static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "MPS fill kernel size exceeds uint32 range");
  }

  common_layout::FillLayoutParams layout_params{};
  common_layout::fillFillLayoutParams(layout_params, shape, strides,
                                      "MPS fill kernel");

  const auto offset_u32 = static_cast<std::uint32_t>(offset);
  const auto numel_u32 = static_cast<std::uint32_t>(layout.numel);
  const float fill_value = static_cast<float>(params.value.get());

  auto session = mps_kernel::MpsKernelSession::begin(base, args, 0);
  if (!session) {
    error::throwError(error::OrteafErrc::InvalidState,
                      "MPS fill kernel could not begin execution session");
  }

  kernel::Param offset_param(kernel::ParamId::Offset, offset_u32);
  kernel::Param numel_param(kernel::ParamId::NumElements, numel_u32);
  kernel::Param fill_value_param(kernel::ParamId::Value, fill_value);

  session->waitDependencies(storages.output);
  session->bindStorages<0>(storages.output);
  session->bindParams<1, 2, 3>(offset_param, numel_param, fill_value_param);
  session->setBytes(&layout_params, sizeof(layout_params), 4);
  session->dispatch1D(static_cast<std::size_t>(numel_u32));

  if (!session->updateTokens(storages.output)) {
    error::throwError(error::OrteafErrc::InvalidState,
                      "MPS fill kernel failed to update synchronization tokens");
  }
}

kernel::core::KernelMetadataLease createFillMpsMetadata() {
  using MpsExecutionApi =
      ::orteaf::internal::execution::mps::api::MpsExecutionApi;

  MpsExecutionApi::KernelKeys keys;
  keys.pushBack(MpsExecutionApi::KernelKey{
      MpsExecutionApi::LibraryKey::Named("fill_kernel"),
      MpsExecutionApi::FunctionKey::Named("orteaf_fill_strided_f32")});

  auto metadata_lease = MpsExecutionApi::acquireKernelMetadata(keys);
  if (auto *meta_ptr = metadata_lease.operator->()) {
    meta_ptr->setExecute(fillMpsExecute);
  }

  return kernel::core::KernelMetadataLease{std::move(metadata_lease)};
}

void registerFillKernel(
    ::orteaf::internal::architecture::Architecture architecture =
        ::orteaf::internal::architecture::Architecture::MpsGeneric) {
  kernel::core::KernelMetadataLease metadata;
  metadata = createFillMpsMetadata();
  auto key = ::orteaf::internal::kernel::kernel_key::make(
      ::orteaf::internal::ops::Op::Fill, architecture,
      static_cast<::orteaf::internal::kernel::Layout>(0),
      ::orteaf::internal::DType::F32,
      static_cast<::orteaf::internal::kernel::Variant>(0));
  ::orteaf::internal::kernel::api::KernelRegistryApi::registerKernel(
      key, std::move(metadata));
}

void registerFillKernelDefault() { registerFillKernel(); }

ORTEAF_REGISTER_KERNEL(
    ::orteaf::extension::kernel::mps::registerFillKernelDefault);

} // namespace orteaf::extension::kernel::mps

#endif // ORTEAF_ENABLE_MPS
