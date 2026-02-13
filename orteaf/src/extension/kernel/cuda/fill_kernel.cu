#if ORTEAF_ENABLE_CUDA

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

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

#include "../common/layout_common.h"

namespace orteaf::extension::kernel::cuda {

namespace kernel = ::orteaf::internal::kernel;
namespace cuda_kernel = ::orteaf::internal::kernel::cuda;
namespace error = ::orteaf::internal::diagnostics::error;
namespace common_layout = ::orteaf::extension::kernel::common::layout;
namespace cuda_wrapper =
    ::orteaf::internal::execution::cuda::platform::wrapper;

struct FillCudaStorages : kernel::StorageSchema<FillCudaStorages> {
  kernel::StorageField<kernel::OperandId::Output> output;

  ORTEAF_EXTRACT_STORAGES(output)
};

struct FillCudaParams : kernel::ParamSchema<FillCudaParams> {
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

void fillCudaExecute(
    ::orteaf::internal::execution::cuda::resource::CudaKernelBase &base,
    kernel::KernelArgs &args) {
  auto storages = FillCudaStorages::extract(args);
  auto params = FillCudaParams::extract(args);

  using AnyBinding = kernel::KernelArgs::StorageListType::Storage::value_type;
  auto &lease = storages.output.lease<AnyBinding>();
  auto *cuda_lease = lease.tryAs<::orteaf::internal::storage::CudaStorageLease>();
  if (!cuda_lease || !(*cuda_lease)) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "CUDA fill kernel requires CUDA output storage");
  }

  auto *storage = cuda_lease->operator->();
  if (storage == nullptr || !storage->bufferView()) {
    error::throwError(error::OrteafErrc::InvalidState,
                      "CUDA fill kernel output buffer is unavailable");
  }

  if (lease.dtype() != ::orteaf::internal::DType::F32 ||
      storage->dtype() != ::orteaf::internal::DType::F32) {
    error::throwError(error::OrteafErrc::Unsupported,
                      "CUDA fill kernel supports only F32");
  }

  const auto storage_numel_raw = storage->numel();
  if (storage_numel_raw >
      static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "CUDA fill kernel storage size exceeds index range");
  }
  const auto storage_numel = static_cast<std::int64_t>(storage_numel_raw);

  const auto raw_offset = params.offset.get();
  if (raw_offset < 0) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "CUDA fill kernel received negative offset");
  }

  const auto &shape = params.shape.get();
  const auto &strides = params.strides.get();
  const auto layout = common_layout::analyzeLayout(
      shape, strides, raw_offset, "CUDA fill kernel", "output");
  if (layout.has_zero) {
    return;
  }

  if (layout.min_index < 0 || layout.max_index < 0 ||
      layout.max_index >= storage_numel) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "CUDA fill kernel view exceeds output storage bounds");
  }

  const auto offset = static_cast<std::size_t>(raw_offset);
  if (layout.numel > static_cast<std::size_t>(
                        std::numeric_limits<std::uint32_t>::max()) ||
      offset >
          static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) ||
      static_cast<std::size_t>(layout.max_index) >
          static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "CUDA fill kernel size exceeds uint32 range");
  }

  common_layout::FillLayoutParams layout_params{};
  common_layout::fillFillLayoutParams(layout_params, shape, strides,
                                      "CUDA fill kernel");

  auto offset_u32 = static_cast<std::uint32_t>(offset);
  auto numel_u32 = static_cast<std::uint32_t>(layout.numel);
  auto fill_value = static_cast<float>(params.value.get());

  auto session = cuda_kernel::CudaKernelSession::begin(base, args, 0);
  if (!session) {
    error::throwError(error::OrteafErrc::InvalidState,
                      "CUDA fill kernel could not begin execution session");
  }

  auto output_view = storage->bufferView();
  auto output_ptr = cuda_wrapper::cuDeviceptrFromOpaque(output_view.raw());
  auto function = cuda_wrapper::objcFromOpaqueNoown<CUfunction>(session->function());
  auto stream = cuda_wrapper::objcFromOpaqueNoown<CUstream>(session->stream());

  void *kernel_args[] = {
      &output_ptr,
      &offset_u32,
      &numel_u32,
      &fill_value,
      &layout_params,
  };

  const auto block = cuda_kernel::CudaKernelSession::makeBlock1D(256);
  const auto grid =
      cuda_kernel::CudaKernelSession::makeGrid1D(layout.numel, block.x);

  CU_CHECK(cuLaunchKernel(function, grid.x, grid.y, grid.z, block.x, block.y,
                          block.z, 0, stream, kernel_args, nullptr));

  session->synchronize();
}

kernel::core::KernelMetadataLease createFillCudaMetadata() {
  using CudaExecutionApi =
      ::orteaf::internal::execution::cuda::api::CudaExecutionApi;

  CudaExecutionApi::KernelKeys keys;
  keys.pushBack(CudaExecutionApi::KernelKey{
      CudaExecutionApi::ModuleKey::Embedded("fill_kernel"),
      std::string{"orteaf_fill_strided_f32"}});

  auto metadata_lease = CudaExecutionApi::acquireKernelMetadata(keys);
  if (auto *meta_ptr = metadata_lease.operator->()) {
    meta_ptr->setExecute(fillCudaExecute);
  }

  return kernel::core::KernelMetadataLease{std::move(metadata_lease)};
}

void registerFillKernel(
    ::orteaf::internal::architecture::Architecture architecture =
        ::orteaf::internal::architecture::Architecture::CudaGeneric) {
  auto metadata = createFillCudaMetadata();
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
    ::orteaf::extension::kernel::cuda::registerFillKernelDefault);

} // namespace orteaf::extension::kernel::cuda

#endif // ORTEAF_ENABLE_CUDA
