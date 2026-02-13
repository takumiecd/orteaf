#if ORTEAF_ENABLE_MPS

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <system_error>

#include <orteaf/internal/architecture/architecture.h>
#include <orteaf/internal/diagnostics/error/error.h>
#include <orteaf/internal/dtype/dtype.h>
#include <orteaf/internal/execution/mps/api/mps_execution_api.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_buffer.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_command_buffer.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_heap.h>
#include <orteaf/internal/kernel/api/kernel_registry_api.h>
#include <orteaf/internal/kernel/core/kernel_args.h>
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

#include "copy_kernel_common.h"
#include "../common/copy_plan_common.h"
#include "../common/layout_common.h"

namespace orteaf::extension::kernel::mps {

namespace kernel = ::orteaf::internal::kernel;
namespace error = ::orteaf::internal::diagnostics::error;
namespace mps_kernel = ::orteaf::internal::kernel::mps;
namespace common_layout = ::orteaf::extension::kernel::common::layout;
namespace copy_plan = ::orteaf::extension::kernel::common::copy_plan;

using ShapeVector = common_layout::ShapeVector;

struct CopyHostToMpsStorages : kernel::StorageSchema<CopyHostToMpsStorages> {
  kernel::StorageField<kernel::OperandId::Input0> input;
  kernel::StorageField<kernel::OperandId::Output> output;

  ORTEAF_EXTRACT_STORAGES(input, output)
};

struct CopyHostToMpsParams : kernel::ParamSchema<CopyHostToMpsParams> {
  kernel::ScopedField<kernel::ParamId::Shape, ShapeVector,
                      kernel::OperandId::Input0>
      input_shape;
  kernel::ScopedField<kernel::ParamId::Strides, ShapeVector,
                      kernel::OperandId::Input0>
      input_strides;
  kernel::ScopedField<kernel::ParamId::Offset, std::int64_t,
                      kernel::OperandId::Input0>
      input_offset;

  kernel::ScopedField<kernel::ParamId::Shape, ShapeVector,
                      kernel::OperandId::Output>
      output_shape;
  kernel::ScopedField<kernel::ParamId::Strides, ShapeVector,
                      kernel::OperandId::Output>
      output_strides;
  kernel::ScopedField<kernel::ParamId::Offset, std::int64_t,
                      kernel::OperandId::Output>
      output_offset;

  ORTEAF_EXTRACT_FIELDS(input_shape, input_strides, input_offset, output_shape,
                        output_strides, output_offset)
};

namespace {

constexpr const char *kOpName = "MPS copyHostToDevice kernel";

} // namespace

void copyHostToMpsExecute(
    ::orteaf::internal::execution::mps::resource::MpsKernelBase &base,
    kernel::KernelArgs &args) {
  auto storages = CopyHostToMpsStorages::extract(args);
  auto params = CopyHostToMpsParams::extract(args);

  using AnyBinding = kernel::KernelArgs::StorageListType::Storage::value_type;
  auto &input_any = storages.input.lease<AnyBinding>();
  auto &output_any = storages.output.lease<AnyBinding>();

  auto &input_storage = storages.input.payloadAs<
      AnyBinding, ::orteaf::internal::storage::CpuStorageLease>(
      "MPS copyHostToDevice kernel requires CPU input and MPS output storage",
      "MPS copyHostToDevice kernel buffer is unavailable",
      [](const auto &typed_storage) { return typed_storage.buffer() != nullptr; });
  auto &output_storage = storages.output.payloadAs<
      AnyBinding, ::orteaf::internal::storage::MpsStorageLease>(
      "MPS copyHostToDevice kernel requires CPU input and MPS output storage",
      "MPS copyHostToDevice kernel buffer is unavailable",
      [](const auto &typed_storage) { return typed_storage.buffer() != nullptr; });

  const auto dtype = input_any.dtype();
  if (dtype != output_any.dtype() || dtype != input_storage.dtype() ||
      dtype != output_storage.dtype()) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "MPS copyHostToDevice kernel requires matching dtype");
  }

  const auto input_storage_numel_raw = input_storage.numel();
  const auto output_storage_numel_raw = output_storage.numel();
  if (input_storage_numel_raw >
          static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
      output_storage_numel_raw >
          static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "MPS copyHostToDevice kernel storage size exceeds index range");
  }
  const auto input_storage_numel = static_cast<std::int64_t>(input_storage_numel_raw);
  const auto output_storage_numel =
      static_cast<std::int64_t>(output_storage_numel_raw);

  const auto &input_shape = params.input_shape.get();
  const auto &output_shape = params.output_shape.get();
  const auto &input_strides = params.input_strides.get();
  const auto &output_strides = params.output_strides.get();
  const auto input_offset = params.input_offset.get();
  const auto output_offset = params.output_offset.get();

  const auto validation = copy_plan::validateCopyLayouts(
      input_shape, input_strides, input_offset, input_storage_numel, output_shape,
      output_strides, output_offset, output_storage_numel, dtype, kOpName);
  if (validation.has_zero) {
    return;
  }
  const auto &input_layout = validation.input_layout;
  const auto &output_layout = validation.output_layout;
  const auto elem_size = validation.elem_size;
  if (elem_size > static_cast<std::size_t>(
                      std::numeric_limits<std::uint32_t>::max())) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "MPS copyHostToDevice kernel element size exceeds uint32 range");
  }
  const auto numel = validation.numel;
  if (numel > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) ||
      static_cast<std::size_t>(output_offset) >
          static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) ||
      static_cast<std::size_t>(output_layout.max_index) >
          static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "MPS copyHostToDevice kernel size exceeds uint32 range");
  }

  auto staging_lease = copy_detail::acquireSharedMpsStaging(dtype, numel, kOpName);
  auto *staging_storage = staging_lease.operator->();
  if (staging_storage == nullptr || staging_storage->buffer() == nullptr) {
    error::throwError(error::OrteafErrc::InvalidState,
                      "MPS copyHostToDevice kernel staging is unavailable");
  }

  auto *staging_ptr =
      ::orteaf::internal::execution::mps::platform::wrapper::getBufferContents(
          staging_storage->buffer());
  if (staging_ptr == nullptr) {
    error::throwError(error::OrteafErrc::InvalidState,
                      "MPS copyHostToDevice kernel staging is not CPU-visible");
  }

  const auto *input_base = static_cast<const std::byte *>(input_storage.buffer());
  auto *staging_bytes = static_cast<std::byte *>(staging_ptr);
  for (std::size_t linear = 0; linear < numel; ++linear) {
    const auto src_index = common_layout::physicalIndexForLinear(
        static_cast<std::uint64_t>(linear), input_shape, input_strides,
        input_offset);
    const auto src_byte_index = static_cast<std::size_t>(src_index) * elem_size;
    std::copy_n(input_base + src_byte_index, elem_size,
                staging_bytes + linear * elem_size);
  }

  const auto staging_strides = common_layout::makeContiguousStrides(
      output_shape, kOpName, "output");
  common_layout::TransferLayoutParams layout_params{};
  common_layout::fillTransferLayoutParams(layout_params, output_shape,
                                          staging_strides, output_strides,
                                          kOpName);

  const std::uint32_t input_offset_u32 = 0;
  const auto output_offset_u32 = static_cast<std::uint32_t>(output_offset);
  const auto numel_u32 = static_cast<std::uint32_t>(numel);
  const auto elem_size_u32 = static_cast<std::uint32_t>(elem_size);

  auto session = mps_kernel::MpsKernelSession::begin(base, args, 0);
  if (!session) {
    error::throwError(error::OrteafErrc::InvalidState,
                      "MPS copyHostToDevice kernel could not begin execution session");
  }

  session->waitDependencies(storages.output);
  mps_kernel::MpsKernelSession::Ops::setBuffer(session->encoder(),
                                                *staging_storage, 0);
  mps_kernel::MpsKernelSession::Ops::setBuffer(session->encoder(),
                                                output_storage, 1);
  session->setBytes(&input_offset_u32, sizeof(input_offset_u32), 2);
  session->setBytes(&output_offset_u32, sizeof(output_offset_u32), 3);
  session->setBytes(&numel_u32, sizeof(numel_u32), 4);
  session->setBytes(&elem_size_u32, sizeof(elem_size_u32), 5);
  session->setBytes(&layout_params, sizeof(layout_params), 6);
  session->dispatch1D(static_cast<std::size_t>(numel_u32));

  if (!session->updateTokens(storages.output)) {
    error::throwError(error::OrteafErrc::InvalidState,
                      "MPS copyHostToDevice kernel failed to update synchronization tokens");
  }
}

kernel::core::KernelMetadataLease createCopyHostToMpsMetadata() {
  using MpsExecutionApi =
      ::orteaf::internal::execution::mps::api::MpsExecutionApi;

  MpsExecutionApi::KernelKeys keys;
  keys.pushBack(MpsExecutionApi::KernelKey{
      MpsExecutionApi::LibraryKey::Named("transfer_kernel"),
      MpsExecutionApi::FunctionKey::Named("orteaf_copy_contiguous_to_strided_u8")});

  auto metadata_lease = MpsExecutionApi::acquireKernelMetadata(keys);
  if (auto *meta_ptr = metadata_lease.operator->()) {
    meta_ptr->setExecute(copyHostToMpsExecute);
  }

  return kernel::core::KernelMetadataLease{std::move(metadata_lease)};
}

void registerCopyHostToMpsKernel(
    ::orteaf::internal::architecture::Architecture architecture =
        ::orteaf::internal::architecture::Architecture::MpsGeneric) {
  kernel::core::KernelMetadataLease metadata;
  metadata = createCopyHostToMpsMetadata();

  auto key = ::orteaf::internal::kernel::kernel_key::makeAnyDType(
      ::orteaf::internal::ops::Op::CopyHostToDevice, architecture,
      static_cast<::orteaf::internal::kernel::Layout>(0),
      static_cast<::orteaf::internal::kernel::Variant>(0));
  ::orteaf::internal::kernel::api::KernelRegistryApi::registerKernel(
      key, std::move(metadata));
}

void registerCopyHostToMpsKernelDefault() { registerCopyHostToMpsKernel(); }

ORTEAF_REGISTER_KERNEL(
    ::orteaf::extension::kernel::mps::registerCopyHostToMpsKernelDefault);

} // namespace orteaf::extension::kernel::mps

#endif // ORTEAF_ENABLE_MPS
