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
#include "transfer_layout_common.h"

namespace orteaf::extension::kernel::mps {

namespace kernel = ::orteaf::internal::kernel;
namespace error = ::orteaf::internal::diagnostics::error;
namespace mps_kernel = ::orteaf::internal::kernel::mps;
namespace mps_wrapper = ::orteaf::internal::execution::mps::platform::wrapper;

using ShapeVector = detail::ShapeVector;

struct CopyMpsToHostStorages : kernel::StorageSchema<CopyMpsToHostStorages> {
  kernel::StorageField<kernel::OperandId::Input0> input;
  kernel::StorageField<kernel::OperandId::Output> output;

  ORTEAF_EXTRACT_STORAGES(input, output)
};

struct CopyMpsToHostParams : kernel::ParamSchema<CopyMpsToHostParams> {
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

constexpr const char *kOpName = "MPS copyDeviceToHost kernel";

void waitCurrentQueue(::orteaf::internal::kernel::KernelArgs &args) {
  auto *context =
      args.context().tryAs<::orteaf::internal::execution_context::mps::Context>();
  if (context == nullptr) {
    error::throwError(error::OrteafErrc::InvalidState,
                      "MPS copyDeviceToHost kernel requires MPS context");
  }

  auto *queue_resource = context->command_queue.operator->();
  if (queue_resource == nullptr || queue_resource->queue() == nullptr) {
    error::throwError(error::OrteafErrc::InvalidState,
                      "MPS copyDeviceToHost kernel command queue unavailable");
  }

  auto command_buffer = mps_wrapper::createCommandBuffer(queue_resource->queue());
  if (command_buffer == nullptr) {
    error::throwError(error::OrteafErrc::InvalidState,
                      "MPS copyDeviceToHost kernel failed to create command buffer");
  }
  mps_wrapper::commit(command_buffer);
  mps_wrapper::waitUntilCompleted(command_buffer);
  mps_wrapper::destroyCommandBuffer(command_buffer);
}

} // namespace

void copyMpsToHostExecute(
    ::orteaf::internal::execution::mps::resource::MpsKernelBase &base,
    kernel::KernelArgs &args) {
  auto storages = CopyMpsToHostStorages::extract(args);
  auto params = CopyMpsToHostParams::extract(args);

  using AnyBinding = kernel::KernelArgs::StorageListType::Storage::value_type;
  auto &input_any = storages.input.lease<AnyBinding>();
  auto &output_any = storages.output.lease<AnyBinding>();

  auto *input_lease = input_any.tryAs<::orteaf::internal::storage::MpsStorageLease>();
  auto *output_lease =
      output_any.tryAs<::orteaf::internal::storage::CpuStorageLease>();
  if (!input_lease || !(*input_lease) || !output_lease || !(*output_lease)) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "MPS copyDeviceToHost kernel requires MPS input and CPU output storage");
  }
  auto *input_storage = input_lease->operator->();
  auto *output_storage = output_lease->operator->();
  if (input_storage == nullptr || output_storage == nullptr ||
      input_storage->buffer() == nullptr || output_storage->buffer() == nullptr) {
    error::throwError(error::OrteafErrc::InvalidState,
                      "MPS copyDeviceToHost kernel buffer is unavailable");
  }

  const auto dtype = input_any.dtype();
  if (dtype != output_any.dtype() || dtype != input_storage->dtype() ||
      dtype != output_storage->dtype()) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "MPS copyDeviceToHost kernel requires matching dtype");
  }

  const auto input_storage_numel_raw = input_storage->numel();
  const auto output_storage_numel_raw = output_storage->numel();
  if (input_storage_numel_raw >
          static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
      output_storage_numel_raw >
          static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "MPS copyDeviceToHost kernel storage size exceeds index range");
  }
  const auto input_storage_numel = static_cast<std::int64_t>(input_storage_numel_raw);
  const auto output_storage_numel =
      static_cast<std::int64_t>(output_storage_numel_raw);

  const auto input_offset = params.input_offset.get();
  const auto output_offset = params.output_offset.get();
  if (input_offset < 0 || output_offset < 0) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "MPS copyDeviceToHost kernel received negative offset");
  }

  const auto &input_shape = params.input_shape.get();
  const auto &output_shape = params.output_shape.get();
  const auto &input_strides = params.input_strides.get();
  const auto &output_strides = params.output_strides.get();
  detail::ensureSameShape(input_shape, output_shape, kOpName);

  const auto input_layout =
      detail::analyzeLayout(input_shape, input_strides, input_offset, kOpName,
                            "input");
  const auto output_layout =
      detail::analyzeLayout(output_shape, output_strides, output_offset, kOpName,
                            "output");
  if (input_layout.has_zero) {
    return;
  }
  if (input_layout.numel != output_layout.numel) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "MPS copyDeviceToHost kernel numel mismatch");
  }

  if (input_layout.min_index < 0 || input_layout.max_index < 0 ||
      input_layout.max_index >= input_storage_numel ||
      output_layout.min_index < 0 || output_layout.max_index < 0 ||
      output_layout.max_index >= output_storage_numel) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "MPS copyDeviceToHost kernel view exceeds storage bounds");
  }

  const auto elem_size = ::orteaf::internal::sizeOf(dtype);
  if (elem_size > static_cast<std::size_t>(
                      std::numeric_limits<std::uint32_t>::max())) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "MPS copyDeviceToHost kernel element size exceeds uint32 range");
  }
  const auto numel = input_layout.numel;
  if (numel > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) ||
      static_cast<std::size_t>(input_offset) >
          static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) ||
      static_cast<std::size_t>(input_layout.max_index) >
          static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "MPS copyDeviceToHost kernel size exceeds uint32 range");
  }

  auto staging_lease = copy_detail::acquireSharedMpsStaging(dtype, numel, kOpName);
  auto *staging_storage = staging_lease.operator->();
  if (staging_storage == nullptr || staging_storage->buffer() == nullptr) {
    error::throwError(error::OrteafErrc::InvalidState,
                      "MPS copyDeviceToHost kernel staging is unavailable");
  }

  const auto contiguous_strides =
      detail::makeContiguousStrides(input_shape, kOpName, "input");
  detail::TransferLayoutParams layout_params{};
  detail::fillLayoutParams(layout_params, input_shape, input_strides,
                           contiguous_strides, kOpName);

  const auto input_offset_u32 = static_cast<std::uint32_t>(input_offset);
  const std::uint32_t output_offset_u32 = 0;
  const auto numel_u32 = static_cast<std::uint32_t>(numel);
  const auto elem_size_u32 = static_cast<std::uint32_t>(elem_size);

  {
    auto session = mps_kernel::MpsKernelSession::begin(base, args, 0);
    if (!session) {
      error::throwError(error::OrteafErrc::InvalidState,
                        "MPS copyDeviceToHost kernel could not begin execution session");
    }

    session->waitDependencies(storages.input);
    mps_kernel::MpsKernelSession::Ops::setBuffer(session->encoder(),
                                                  *input_storage, 0);
    mps_kernel::MpsKernelSession::Ops::setBuffer(session->encoder(),
                                                  *staging_storage, 1);
    session->setBytes(&input_offset_u32, sizeof(input_offset_u32), 2);
    session->setBytes(&output_offset_u32, sizeof(output_offset_u32), 3);
    session->setBytes(&numel_u32, sizeof(numel_u32), 4);
    session->setBytes(&elem_size_u32, sizeof(elem_size_u32), 5);
    session->setBytes(&layout_params, sizeof(layout_params), 6);
    session->dispatch1D(static_cast<std::size_t>(numel_u32));

    if (!session->updateTokens(storages.input)) {
      error::throwError(error::OrteafErrc::InvalidState,
                        "MPS copyDeviceToHost kernel failed to update synchronization tokens");
    }
  }

  waitCurrentQueue(args);

  const auto *staging_ptr = mps_wrapper::getBufferContentsConst(staging_storage->buffer());
  if (staging_ptr == nullptr) {
    error::throwError(error::OrteafErrc::InvalidState,
                      "MPS copyDeviceToHost kernel staging is not CPU-visible");
  }

  auto *output_base = static_cast<std::byte *>(output_storage->buffer());
  const auto *staging_bytes = static_cast<const std::byte *>(staging_ptr);
  for (std::size_t linear = 0; linear < numel; ++linear) {
    const auto dst_index = detail::physicalIndexForLinear(
        static_cast<std::uint64_t>(linear), output_shape, output_strides,
        output_offset);
    const auto dst_byte_index = static_cast<std::size_t>(dst_index) * elem_size;
    std::copy_n(staging_bytes + linear * elem_size, elem_size,
                output_base + dst_byte_index);
  }
}

kernel::core::KernelMetadataLease createCopyMpsToHostMetadata() {
  using MpsExecutionApi =
      ::orteaf::internal::execution::mps::api::MpsExecutionApi;

  MpsExecutionApi::KernelKeys keys;
  keys.pushBack(MpsExecutionApi::KernelKey{
      MpsExecutionApi::LibraryKey::Named("transfer_kernel"),
      MpsExecutionApi::FunctionKey::Named("orteaf_copy_strided_to_contiguous_u8")});

  auto metadata_lease = MpsExecutionApi::acquireKernelMetadata(keys);
  if (auto *meta_ptr = metadata_lease.operator->()) {
    meta_ptr->setExecute(copyMpsToHostExecute);
  }

  return kernel::core::KernelMetadataLease{std::move(metadata_lease)};
}

void registerCopyMpsToHostKernel(
    ::orteaf::internal::architecture::Architecture architecture =
        ::orteaf::internal::architecture::Architecture::MpsGeneric) {
  kernel::core::KernelMetadataLease metadata;
  try {
    metadata = createCopyMpsToHostMetadata();
  } catch (const std::system_error &err) {
    const auto invalid_state =
        error::makeErrorCode(error::OrteafErrc::InvalidState);
    if (err.code() == invalid_state) {
      return;
    }
    throw;
  }

  auto key = ::orteaf::internal::kernel::kernel_key::makeAnyDType(
      ::orteaf::internal::ops::Op::CopyDeviceToHost, architecture,
      static_cast<::orteaf::internal::kernel::Layout>(0),
      static_cast<::orteaf::internal::kernel::Variant>(0));
  ::orteaf::internal::kernel::api::KernelRegistryApi::registerKernel(
      key, std::move(metadata));
}

void registerCopyMpsToHostKernelDefault() { registerCopyMpsToHostKernel(); }

ORTEAF_REGISTER_KERNEL(
    ::orteaf::extension::kernel::mps::registerCopyMpsToHostKernelDefault);

} // namespace orteaf::extension::kernel::mps

#endif // ORTEAF_ENABLE_MPS
