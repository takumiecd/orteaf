#if ORTEAF_ENABLE_MPS

#include <cstddef>
#include <cstdint>
#include <limits>
#include <system_error>

#include <orteaf/internal/architecture/architecture.h>
#include <orteaf/internal/base/checked_int.h>
#include <orteaf/internal/base/inline_vector.h>
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

namespace orteaf::extension::kernel::mps {

namespace kernel = ::orteaf::internal::kernel;
namespace error = ::orteaf::internal::diagnostics::error;
namespace mps_kernel = ::orteaf::internal::kernel::mps;

inline constexpr std::uint8_t kFillShapeInlineCapacity = 8;

struct FillMpsStorages : kernel::StorageSchema<FillMpsStorages> {
  kernel::StorageField<kernel::OperandId::Output> output;

  ORTEAF_EXTRACT_STORAGES(output)
};

struct FillMpsParams : kernel::ParamSchema<FillMpsParams> {
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
  bool contiguous{};
  std::int64_t min_index{};
  std::int64_t max_index{};
};

LayoutInfo analyzeLayout(const FillMpsParams::ShapeVector &shape,
                         const FillMpsParams::ShapeVector &strides,
                         std::int64_t offset) {
  if (shape.size != strides.size) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "MPS fill kernel received mismatched shape/strides");
  }

  LayoutInfo info{};
  info.numel = 1;
  info.has_zero = false;
  info.contiguous = true;
  info.min_index = offset;
  info.max_index = offset;

  if (shape.size == 0) {
    return info;
  }

  std::size_t expected_stride = 1;
  for (std::size_t i = shape.size; i-- > 0;) {
    const auto dim = shape.data[i];
    if (dim < 0) {
      error::throwError(error::OrteafErrc::InvalidParameter,
                        "MPS fill kernel received negative shape dimension");
    }
    if (dim == 0) {
      info.numel = 0;
      info.has_zero = true;
      return info;
    }

    const auto dim_size = static_cast<std::size_t>(dim);
    if (info.numel > std::numeric_limits<std::size_t>::max() / dim_size) {
      error::throwError(error::OrteafErrc::InvalidParameter,
                        "MPS fill kernel shape is too large");
    }
    info.numel *= dim_size;

    if (strides.data[i] != static_cast<std::int64_t>(expected_stride)) {
      info.contiguous = false;
    }

    if (expected_stride > std::numeric_limits<std::size_t>::max() / dim_size) {
      error::throwError(error::OrteafErrc::InvalidParameter,
                        "MPS fill kernel shape is too large");
    }
    expected_stride *= dim_size;
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
                        "MPS fill kernel index range overflow");
    }
    if (stride >= 0) {
      if (::orteaf::internal::base::addOverflowI64(max_index, span,
                                                   max_index)) {
        error::throwError(error::OrteafErrc::InvalidParameter,
                          "MPS fill kernel index range overflow");
      }
    } else {
      if (::orteaf::internal::base::addOverflowI64(min_index, span,
                                                   min_index)) {
        error::throwError(error::OrteafErrc::InvalidParameter,
                          "MPS fill kernel index range overflow");
      }
    }
  }

  info.min_index = min_index;
  info.max_index = max_index;
  return info;
}

struct FillLayoutParams {
  std::uint32_t rank{};
  std::uint32_t shape[kFillShapeInlineCapacity]{};
  std::int32_t strides[kFillShapeInlineCapacity]{};
};

} // namespace

void fillMpsExecute(
    ::orteaf::internal::execution::mps::resource::MpsKernelBase &base,
    kernel::KernelArgs &args) {
  auto storages = FillMpsStorages::extract(args);
  auto params = FillMpsParams::extract(args);

  using AnyBinding = kernel::KernelArgs::StorageListType::Storage::value_type;
  auto &lease = storages.output.lease<AnyBinding>();
  auto *mps_lease = lease.tryAs<::orteaf::internal::storage::MpsStorageLease>();
  if (!mps_lease || !(*mps_lease)) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "MPS fill kernel requires MPS output storage");
  }
  auto *storage = mps_lease->operator->();
  if (storage == nullptr || storage->buffer() == nullptr) {
    error::throwError(error::OrteafErrc::InvalidState,
                      "MPS fill kernel output buffer is unavailable");
  }

  if (lease.dtype() != ::orteaf::internal::DType::F32 ||
      storage->dtype() != ::orteaf::internal::DType::F32) {
    error::throwError(error::OrteafErrc::Unsupported,
                      "MPS fill kernel supports only F32");
  }
  const auto storage_numel_raw = storage->numel();
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
  const auto layout = analyzeLayout(shape, strides, raw_offset);
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

  FillLayoutParams layout_params{};
  layout_params.rank = shape.size;
  for (std::size_t i = 0; i < shape.size; ++i) {
    const auto dim = shape.data[i];
    const auto stride = strides.data[i];
    if (dim < 0 ||
        dim > static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max())) {
      error::throwError(error::OrteafErrc::InvalidParameter,
                        "MPS fill kernel shape dimension exceeds uint32 range");
    }
    if (stride < static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::min()) ||
        stride > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
      error::throwError(error::OrteafErrc::InvalidParameter,
                        "MPS fill kernel stride exceeds int32 range");
    }
    layout_params.shape[i] = static_cast<std::uint32_t>(dim);
    layout_params.strides[i] = static_cast<std::int32_t>(stride);
  }

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
