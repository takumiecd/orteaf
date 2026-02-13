#if ORTEAF_ENABLE_CUDA

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include <orteaf/internal/architecture/architecture.h>
#include <orteaf/internal/diagnostics/error/error.h>
#include <orteaf/internal/dtype/dtype.h>
#include <orteaf/internal/execution/cuda/api/cuda_execution_api.h>
#include <orteaf/internal/execution/cuda/platform/wrapper/cuda_check.h>
#include <orteaf/internal/execution/cuda/platform/wrapper/cuda_objc_bridge.h>
#include <orteaf/internal/kernel/api/kernel_registry_api.h>
#include <orteaf/internal/kernel/core/kernel_args.h>
#include <orteaf/internal/kernel/core/kernel_entry.h>
#include <orteaf/internal/kernel/core/kernel_key.h>
#include <orteaf/internal/kernel/core/kernel_metadata.h>
#include <orteaf/internal/kernel/cuda/cuda_kernel_session.h>
#include <orteaf/internal/kernel/param/param_id.h>
#include <orteaf/internal/kernel/param/transform/array_view_inline_vector.h>
#include <orteaf/internal/kernel/registry/kernel_auto_registry.h>
#include <orteaf/internal/kernel/schema/kernel_param_schema.h>
#include <orteaf/internal/kernel/schema/kernel_storage_schema.h>
#include <orteaf/internal/kernel/storage/operand_id.h>
#include <orteaf/internal/storage/registry/storage_types.h>

#include "transfer_layout_common.h"

namespace orteaf::extension::kernel::cuda {

namespace kernel = ::orteaf::internal::kernel;
namespace cuda_kernel = ::orteaf::internal::kernel::cuda;
namespace error = ::orteaf::internal::diagnostics::error;
namespace cuda_wrapper =
    ::orteaf::internal::execution::cuda::platform::wrapper;

using ShapeVector = detail::ShapeVector;

struct CopyDeviceToHostStorages
    : kernel::StorageSchema<CopyDeviceToHostStorages> {
  kernel::StorageField<kernel::OperandId::Input0> input;
  kernel::StorageField<kernel::OperandId::Output> output;

  ORTEAF_EXTRACT_STORAGES(input, output)
};

struct CopyDeviceToHostParams : kernel::ParamSchema<CopyDeviceToHostParams> {
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

constexpr const char *kOpName = "CUDA copyDeviceToHost kernel";

} // namespace

void copyDeviceToHostExecute(
    ::orteaf::internal::execution::cuda::resource::CudaKernelBase &base,
    kernel::KernelArgs &args) {
  auto storages = CopyDeviceToHostStorages::extract(args);
  auto params = CopyDeviceToHostParams::extract(args);

  using AnyBinding = kernel::KernelArgs::StorageListType::Storage::value_type;
  auto &input_any = storages.input.lease<AnyBinding>();
  auto &output_any = storages.output.lease<AnyBinding>();

  auto *input_lease = input_any.tryAs<::orteaf::internal::storage::CudaStorageLease>();
  auto *output_lease =
      output_any.tryAs<::orteaf::internal::storage::CudaStorageLease>();
  if (!input_lease || !(*input_lease) || !output_lease || !(*output_lease)) {
    error::throwError(
        error::OrteafErrc::InvalidParameter,
        "CUDA copyDeviceToHost kernel requires CUDA input/output storage");
  }
  auto *input_storage = input_lease->operator->();
  auto *output_storage = output_lease->operator->();
  if (input_storage == nullptr || output_storage == nullptr ||
      !input_storage->bufferView() || !output_storage->bufferView()) {
    error::throwError(error::OrteafErrc::InvalidState,
                      "CUDA copyDeviceToHost kernel buffer is unavailable");
  }

  const auto dtype = input_any.dtype();
  if (dtype != output_any.dtype() || dtype != input_storage->dtype() ||
      dtype != output_storage->dtype()) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "CUDA copyDeviceToHost kernel requires matching dtype");
  }

  const auto input_storage_numel_raw = input_storage->numel();
  const auto output_storage_numel_raw = output_storage->numel();
  if (input_storage_numel_raw >
          static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
      output_storage_numel_raw >
          static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    error::throwError(
        error::OrteafErrc::InvalidParameter,
        "CUDA copyDeviceToHost kernel storage size exceeds index range");
  }
  const auto input_storage_numel = static_cast<std::int64_t>(input_storage_numel_raw);
  const auto output_storage_numel =
      static_cast<std::int64_t>(output_storage_numel_raw);

  const auto input_offset = params.input_offset.get();
  const auto output_offset = params.output_offset.get();
  if (input_offset < 0 || output_offset < 0) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "CUDA copyDeviceToHost kernel received negative offset");
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
                      "CUDA copyDeviceToHost kernel numel mismatch");
  }
  if (!output_layout.contiguous) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "CUDA copyDeviceToHost kernel requires contiguous output");
  }

  if (input_layout.min_index < 0 || input_layout.max_index < 0 ||
      input_layout.max_index >= input_storage_numel ||
      output_layout.min_index < 0 || output_layout.max_index < 0 ||
      output_layout.max_index >= output_storage_numel) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "CUDA copyDeviceToHost kernel view exceeds storage bounds");
  }

  const auto elem_size = ::orteaf::internal::sizeOf(dtype);
  if (elem_size > static_cast<std::size_t>(
                      std::numeric_limits<std::uint32_t>::max())) {
    error::throwError(
        error::OrteafErrc::InvalidParameter,
        "CUDA copyDeviceToHost kernel element size exceeds uint32 range");
  }
  const auto numel = input_layout.numel;
  if (numel > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) ||
      static_cast<std::size_t>(input_offset) >
          static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) ||
      static_cast<std::size_t>(output_offset) >
          static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) ||
      static_cast<std::size_t>(input_layout.max_index) >
          static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) ||
      static_cast<std::size_t>(output_layout.max_index) >
          static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "CUDA copyDeviceToHost kernel size exceeds uint32 range");
  }

  detail::TransferLayoutParams layout_params{};
  detail::fillLayoutParams(layout_params, input_shape, input_strides,
                           output_strides, kOpName);

  const auto input_offset_u32 = static_cast<std::uint32_t>(input_offset);
  const auto output_offset_u32 = static_cast<std::uint32_t>(output_offset);
  const auto numel_u32 = static_cast<std::uint32_t>(numel);
  const auto elem_size_u32 = static_cast<std::uint32_t>(elem_size);

  auto session = cuda_kernel::CudaKernelSession::begin(base, args, 0);
  if (!session) {
    error::throwError(
        error::OrteafErrc::InvalidState,
        "CUDA copyDeviceToHost kernel could not begin execution session");
  }

  auto input_view = input_storage->bufferView();
  auto output_view = output_storage->bufferView();
  auto input_ptr = cuda_wrapper::cuDeviceptrFromOpaque(input_view.raw());
  auto output_ptr = cuda_wrapper::cuDeviceptrFromOpaque(output_view.raw());
  auto function = cuda_wrapper::objcFromOpaqueNoown<CUfunction>(session->function());
  auto stream = cuda_wrapper::objcFromOpaqueNoown<CUstream>(session->stream());

  void *kernel_args[] = {
      &input_ptr,       &output_ptr,      &input_offset_u32, &output_offset_u32,
      &numel_u32,       &elem_size_u32,   &layout_params,
  };

  const auto block = cuda_kernel::CudaKernelSession::makeBlock1D(256);
  const auto grid =
      cuda_kernel::CudaKernelSession::makeGrid1D(numel, block.x);

  CU_CHECK(cuLaunchKernel(function, grid.x, grid.y, grid.z, block.x, block.y,
                          block.z, 0, stream, kernel_args, nullptr));

  session->synchronize();
}

kernel::core::KernelMetadataLease createCopyDeviceToHostCudaMetadata() {
  using CudaExecutionApi =
      ::orteaf::internal::execution::cuda::api::CudaExecutionApi;

  CudaExecutionApi::KernelKeys keys;
  keys.pushBack(CudaExecutionApi::KernelKey{
      CudaExecutionApi::ModuleKey::Embedded("transfer_kernel"),
      std::string{"orteaf_copy_strided_to_contiguous_u8"}});

  auto metadata_lease = CudaExecutionApi::acquireKernelMetadata(keys);
  if (auto *meta_ptr = metadata_lease.operator->()) {
    meta_ptr->setExecute(copyDeviceToHostExecute);
  }

  return kernel::core::KernelMetadataLease{std::move(metadata_lease)};
}

void registerCopyDeviceToHostKernel(
    ::orteaf::internal::architecture::Architecture architecture =
        ::orteaf::internal::architecture::Architecture::CudaGeneric) {
  auto metadata = createCopyDeviceToHostCudaMetadata();
  auto key = ::orteaf::internal::kernel::kernel_key::makeAnyDType(
      ::orteaf::internal::ops::Op::CopyDeviceToHost, architecture,
      static_cast<::orteaf::internal::kernel::Layout>(0),
      static_cast<::orteaf::internal::kernel::Variant>(0));
  ::orteaf::internal::kernel::api::KernelRegistryApi::registerKernel(
      key, std::move(metadata));
}

void registerCopyDeviceToHostKernelDefault() { registerCopyDeviceToHostKernel(); }

ORTEAF_REGISTER_KERNEL(
    ::orteaf::extension::kernel::cuda::registerCopyDeviceToHostKernelDefault);

} // namespace orteaf::extension::kernel::cuda

#endif // ORTEAF_ENABLE_CUDA
