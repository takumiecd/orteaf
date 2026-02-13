#if ORTEAF_ENABLE_CUDA

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include <orteaf/internal/architecture/architecture.h>
#include <orteaf/internal/base/checked_int.h>
#include <orteaf/internal/base/inline_vector.h>
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

namespace orteaf::extension::kernel::cuda {

namespace kernel = ::orteaf::internal::kernel;
namespace cuda_kernel = ::orteaf::internal::kernel::cuda;
namespace error = ::orteaf::internal::diagnostics::error;
namespace cuda_wrapper =
    ::orteaf::internal::execution::cuda::platform::wrapper;

inline constexpr std::uint8_t kFillShapeInlineCapacity = 8;

struct FillCudaStorages : kernel::StorageSchema<FillCudaStorages> {
  kernel::StorageField<kernel::OperandId::Output> output;

  ORTEAF_EXTRACT_STORAGES(output)
};

struct FillCudaParams : kernel::ParamSchema<FillCudaParams> {
  using ShapeVector =
      ::orteaf::internal::base::InlineVector<std::int64_t,
                                             kFillShapeInlineCapacity>;

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

namespace {

struct LayoutInfo {
  std::size_t numel{};
  bool has_zero{};
  std::int64_t min_index{};
  std::int64_t max_index{};
};

struct FillLayoutParams {
  std::uint32_t rank{};
  std::uint32_t shape[kFillShapeInlineCapacity]{};
  std::int32_t strides[kFillShapeInlineCapacity]{};
};

LayoutInfo analyzeLayout(const FillCudaParams::ShapeVector &shape,
                         const FillCudaParams::ShapeVector &strides,
                         std::int64_t offset) {
  if (shape.size != strides.size) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "CUDA fill kernel received mismatched shape/strides");
  }

  LayoutInfo info{};
  info.numel = 1;
  info.has_zero = false;
  info.min_index = offset;
  info.max_index = offset;

  if (shape.size == 0) {
    return info;
  }

  for (std::size_t i = shape.size; i-- > 0;) {
    const auto dim = shape.data[i];
    if (dim < 0) {
      error::throwError(error::OrteafErrc::InvalidParameter,
                        "CUDA fill kernel received negative shape dimension");
    }
    if (dim == 0) {
      info.numel = 0;
      info.has_zero = true;
      return info;
    }

    const auto dim_size = static_cast<std::size_t>(dim);
    if (info.numel > std::numeric_limits<std::size_t>::max() / dim_size) {
      error::throwError(error::OrteafErrc::InvalidParameter,
                        "CUDA fill kernel shape is too large");
    }
    info.numel *= dim_size;
  }

  std::int64_t min_index = offset;
  std::int64_t max_index = offset;
  for (std::uint8_t i = 0; i < shape.size; ++i) {
    const auto dim = shape.data[i];
    if (dim <= 0) {
      continue;
    }
    const auto stride = strides.data[i];
    std::int64_t span = 0;
    if (::orteaf::internal::base::mulOverflowI64(stride, dim - 1, span)) {
      error::throwError(error::OrteafErrc::InvalidParameter,
                        "CUDA fill kernel index range overflow");
    }
    if (stride >= 0) {
      if (::orteaf::internal::base::addOverflowI64(max_index, span,
                                                   max_index)) {
        error::throwError(error::OrteafErrc::InvalidParameter,
                          "CUDA fill kernel index range overflow");
      }
    } else {
      if (::orteaf::internal::base::addOverflowI64(min_index, span,
                                                   min_index)) {
        error::throwError(error::OrteafErrc::InvalidParameter,
                          "CUDA fill kernel index range overflow");
      }
    }
  }

  info.min_index = min_index;
  info.max_index = max_index;
  return info;
}

} // namespace

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
  const auto layout = analyzeLayout(shape, strides, raw_offset);
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

  FillLayoutParams layout_params{};
  layout_params.rank = shape.size;
  for (std::size_t i = 0; i < shape.size; ++i) {
    const auto dim = shape.data[i];
    const auto stride = strides.data[i];
    if (dim < 0 ||
        dim >
            static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max())) {
      error::throwError(error::OrteafErrc::InvalidParameter,
                        "CUDA fill kernel shape dimension exceeds uint32 range");
    }
    if (stride <
            static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::min()) ||
        stride >
            static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
      error::throwError(error::OrteafErrc::InvalidParameter,
                        "CUDA fill kernel stride exceeds int32 range");
    }
    layout_params.shape[i] = static_cast<std::uint32_t>(dim);
    layout_params.strides[i] = static_cast<std::int32_t>(stride);
  }

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
