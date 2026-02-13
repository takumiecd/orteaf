#if ORTEAF_ENABLE_CUDA

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <orteaf/internal/architecture/architecture.h>
#include <orteaf/internal/diagnostics/error/error.h>
#include <orteaf/internal/dtype/dtype.h>
#include <orteaf/internal/execution/cuda/api/cuda_execution_api.h>
#include <orteaf/internal/kernel/api/kernel_registry_api.h>
#include <orteaf/internal/kernel/core/kernel_args.h>
#include <orteaf/internal/kernel/core/kernel_key.h>
#include <orteaf/internal/kernel/core/kernel_metadata.h>
#include <orteaf/internal/kernel/param/param_id.h>
#include <orteaf/internal/kernel/param/transform/array_view_inline_vector.h>
#include <orteaf/internal/kernel/registry/kernel_auto_registry.h>
#include <orteaf/internal/kernel/schema/kernel_param_schema.h>
#include <orteaf/internal/kernel/schema/kernel_storage_schema.h>
#include <orteaf/internal/kernel/storage/operand_id.h>
#include <orteaf/internal/storage/registry/storage_types.h>

#include "copy_kernel_common.h"
#include "transfer_layout_common.h"

namespace orteaf::extension::kernel::cuda {

namespace kernel = ::orteaf::internal::kernel;
namespace error = ::orteaf::internal::diagnostics::error;
namespace cuda_wrapper = copy_detail::cuda_wrapper;

using ShapeVector = detail::ShapeVector;
using TransferLayoutParams = detail::TransferLayoutParams;

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

extern "C" __global__ void orteaf_copy_strided_to_contiguous_u8(
    const std::uint8_t *input, std::uint8_t *output, std::uint32_t input_offset,
    std::uint32_t output_offset, std::uint32_t numel, std::uint32_t elem_size,
    TransferLayoutParams layout);

void launchStridedToContiguousKernel(cuda_wrapper::CudaDevicePtr_t input_base,
                                     cuda_wrapper::CudaDevicePtr_t output_base,
                                     std::uint32_t input_offset,
                                     std::uint32_t numel,
                                     std::uint32_t elem_size,
                                     const TransferLayoutParams &layout) {
  auto *src = reinterpret_cast<std::uint8_t *>(
      static_cast<std::uintptr_t>(input_base));
  auto *dst = reinterpret_cast<std::uint8_t *>(
      static_cast<std::uintptr_t>(output_base));

  constexpr std::uint32_t kThreads = 256;
  const auto blocks = static_cast<std::uint32_t>((numel + kThreads - 1) / kThreads);
  orteaf_copy_strided_to_contiguous_u8<<<blocks, kThreads>>>(
      src, dst, input_offset, 0, numel, elem_size, layout);

  const auto launch_status = cudaGetLastError();
  if (launch_status != cudaSuccess) {
    copy_detail::throwCudaRuntimeError(
        "CUDA copyDeviceToHost kernel launch failed", launch_status);
  }
  const auto sync_status = cudaDeviceSynchronize();
  if (sync_status != cudaSuccess) {
    copy_detail::throwCudaRuntimeError(
        "CUDA copyDeviceToHost kernel synchronization failed", sync_status);
  }
}

} // namespace

void copyDeviceToHostExecute(
    ::orteaf::internal::execution::cuda::resource::CudaKernelBase &,
    kernel::KernelArgs &args) {
  auto storages = CopyDeviceToHostStorages::extract(args);
  auto params = CopyDeviceToHostParams::extract(args);

  using AnyBinding = kernel::KernelArgs::StorageListType::Storage::value_type;
  auto &input_any = storages.input.lease<AnyBinding>();
  auto &output_any = storages.output.lease<AnyBinding>();

  auto *input_lease = input_any.tryAs<::orteaf::internal::storage::CudaStorageLease>();
  auto *output_lease =
      output_any.tryAs<::orteaf::internal::storage::CpuStorageLease>();
  if (!input_lease || !(*input_lease) || !output_lease || !(*output_lease)) {
    error::throwError(
        error::OrteafErrc::InvalidParameter,
        "CUDA copyDeviceToHost kernel requires CUDA input and CPU output storage");
  }

  auto *input_storage = input_lease->operator->();
  auto *output_storage = output_lease->operator->();
  if (input_storage == nullptr || output_storage == nullptr ||
      !input_storage->bufferView() || output_storage->buffer() == nullptr) {
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

  if (input_layout.min_index < 0 || input_layout.max_index < 0 ||
      input_layout.max_index >= input_storage_numel ||
      output_layout.min_index < 0 || output_layout.max_index < 0 ||
      output_layout.max_index >= output_storage_numel) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "CUDA copyDeviceToHost kernel view exceeds storage bounds");
  }

  const auto elem_size = ::orteaf::internal::sizeOf(dtype);
  if (input_layout.numel > std::numeric_limits<std::size_t>::max() / elem_size) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "CUDA copyDeviceToHost kernel byte size overflow");
  }

  const auto bytes = input_layout.numel * elem_size;
  const auto input_base = input_storage->bufferView().data();
  auto *output_base = static_cast<std::byte *>(output_storage->buffer());
  if (input_base == 0) {
    error::throwError(error::OrteafErrc::InvalidState,
                      "CUDA copyDeviceToHost kernel source is invalid");
  }

  if (input_layout.contiguous && output_layout.contiguous) {
    const auto src_byte_offset = static_cast<std::size_t>(input_offset) * elem_size;
    const auto dst_byte_offset = static_cast<std::size_t>(output_offset) * elem_size;
    cuda_wrapper::copyToHost(input_base + src_byte_offset,
                             output_base + dst_byte_offset, bytes);
    return;
  }

  std::vector<std::byte> packed(bytes);

  if (input_layout.contiguous) {
    const auto src_byte_offset = static_cast<std::size_t>(input_offset) * elem_size;
    cuda_wrapper::copyToHost(input_base + src_byte_offset, packed.data(), bytes);
  } else {
    copy_detail::DeviceBufferGuard staging{cuda_wrapper::alloc(bytes), bytes};
    const auto input_offset_u32 = static_cast<std::uint32_t>(input_offset);
    const auto numel_u32 = static_cast<std::uint32_t>(input_layout.numel);
    const auto elem_size_u32 = static_cast<std::uint32_t>(elem_size);

    TransferLayoutParams layout_params{};
    const auto staging_strides =
        detail::makeContiguousStrides(input_shape, kOpName, "input");
    detail::fillLayoutParams(layout_params, input_shape, input_strides,
                             staging_strides, kOpName);

    launchStridedToContiguousKernel(input_base, staging.ptr, input_offset_u32,
                                    numel_u32, elem_size_u32, layout_params);
    cuda_wrapper::copyToHost(staging.ptr, packed.data(), bytes);
  }

  if (output_layout.contiguous) {
    const auto dst_byte_offset = static_cast<std::size_t>(output_offset) * elem_size;
    std::copy_n(packed.data(), bytes, output_base + dst_byte_offset);
    return;
  }

  for (std::size_t linear = 0; linear < input_layout.numel; ++linear) {
    const auto dst_index = detail::physicalIndexForLinear(
        static_cast<std::uint64_t>(linear), output_shape, output_strides,
        output_offset);
    const auto dst_byte_index = static_cast<std::size_t>(dst_index) * elem_size;
    std::copy_n(packed.data() + linear * elem_size, elem_size,
                output_base + dst_byte_index);
  }
}

kernel::core::KernelMetadataLease createCopyDeviceToHostCudaMetadata() {
  using CudaExecutionApi =
      ::orteaf::internal::execution::cuda::api::CudaExecutionApi;

  CudaExecutionApi::KernelKeys keys;
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
