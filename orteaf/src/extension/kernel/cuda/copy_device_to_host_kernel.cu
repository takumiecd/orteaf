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
#include <orteaf/internal/execution/cuda/platform/wrapper/cuda_check.h>
#include <orteaf/internal/execution/cuda/platform/wrapper/cuda_objc_bridge.h>
#include <orteaf/internal/kernel/api/kernel_registry_api.h>
#include <orteaf/internal/kernel/core/kernel_args.h>
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

#include "copy_kernel_common.h"
#include "../common/copy_plan_common.h"
#include "../common/layout_common.h"

namespace orteaf::extension::kernel::cuda {

namespace kernel = ::orteaf::internal::kernel;
namespace error = ::orteaf::internal::diagnostics::error;
namespace cuda_wrapper = copy_detail::cuda_wrapper;
namespace cuda_kernel = ::orteaf::internal::kernel::cuda;
namespace common_layout = ::orteaf::extension::kernel::common::layout;
namespace copy_plan = ::orteaf::extension::kernel::common::copy_plan;

using ShapeVector = common_layout::ShapeVector;
using TransferLayoutParams = common_layout::TransferLayoutParams;

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

void launchStridedToContiguousKernel(cuda_kernel::CudaKernelSession &session,
                                     cuda_wrapper::CudaDevicePtr_t input_base,
                                     cuda_wrapper::CudaDevicePtr_t output_base,
                                     std::uint32_t input_offset,
                                     std::uint32_t numel,
                                     std::uint32_t elem_size,
                                     const TransferLayoutParams &layout) {
  const auto output_offset = std::uint32_t{0};
  auto input_ptr = cuda_wrapper::cuDeviceptrFromOpaque(input_base);
  auto output_ptr = cuda_wrapper::cuDeviceptrFromOpaque(output_base);
  auto function = cuda_wrapper::objcFromOpaqueNoown<CUfunction>(session.function());
  auto stream = cuda_wrapper::objcFromOpaqueNoown<CUstream>(session.stream());

  void *kernel_args[] = {
      &input_ptr,
      &output_ptr,
      &input_offset,
      &output_offset,
      &numel,
      &elem_size,
      const_cast<TransferLayoutParams *>(&layout),
  };

  const auto block = cuda_kernel::CudaKernelSession::makeBlock1D(256);
  const auto grid = cuda_kernel::CudaKernelSession::makeGrid1D(numel, block.x);

  CU_CHECK(cuLaunchKernel(function, grid.x, grid.y, grid.z, block.x, block.y,
                          block.z, 0, stream, kernel_args, nullptr));
  session.synchronize();
}

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
  const auto bytes = validation.bytes;
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
    const auto staging_strides = common_layout::makeContiguousStrides(
        input_shape, kOpName, "input");
    common_layout::fillTransferLayoutParams(layout_params, input_shape,
                                            input_strides, staging_strides,
                                            kOpName);

    auto session = cuda_kernel::CudaKernelSession::begin(base, args, 0);
    if (!session) {
      error::throwError(error::OrteafErrc::InvalidState,
                        "CUDA copyDeviceToHost kernel could not begin execution session");
    }
    launchStridedToContiguousKernel(*session, input_base, staging.ptr,
                                    input_offset_u32, numel_u32, elem_size_u32,
                                    layout_params);
    cuda_wrapper::copyToHost(staging.ptr, packed.data(), bytes);
  }

  if (output_layout.contiguous) {
    const auto dst_byte_offset = static_cast<std::size_t>(output_offset) * elem_size;
    std::copy_n(packed.data(), bytes, output_base + dst_byte_offset);
    return;
  }

  for (std::size_t linear = 0; linear < input_layout.numel; ++linear) {
    const auto dst_index = common_layout::physicalIndexForLinear(
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
