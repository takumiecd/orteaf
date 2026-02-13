#if ORTEAF_ENABLE_CPU

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

#include <orteaf/internal/architecture/architecture.h>
#include <orteaf/internal/diagnostics/error/error.h>
#include <orteaf/internal/execution/cpu/api/cpu_execution_api.h>
#include <orteaf/internal/kernel/api/kernel_registry_api.h>
#include <orteaf/internal/kernel/core/kernel_args.h>
#include <orteaf/internal/kernel/core/kernel_entry.h>
#include <orteaf/internal/kernel/core/kernel_key.h>
#include <orteaf/internal/kernel/core/kernel_metadata.h>
#include <orteaf/internal/kernel/param/param_id.h>
#include <orteaf/internal/kernel/param/transform/array_view_inline_vector.h>
#include <orteaf/internal/kernel/registry/kernel_auto_registry.h>
#include <orteaf/internal/kernel/schema/kernel_param_schema.h>
#include <orteaf/internal/kernel/schema/kernel_storage_schema.h>
#include <orteaf/internal/storage/registry/storage_types.h>
#include <orteaf/internal/dtype/dtype.h>

#include "../common/layout_common.h"

namespace orteaf::extension::kernel::cpu {

namespace kernel = ::orteaf::internal::kernel;
namespace error = ::orteaf::internal::diagnostics::error;
namespace common_layout = ::orteaf::extension::kernel::common::layout;

void fillTensor(void *data, std::size_t count,
                ::orteaf::internal::DType dtype, double value);
void fillTensorStrided(void *data, std::span<const std::int64_t> shape,
                       std::span<const std::int64_t> strides,
                       std::int64_t offset, ::orteaf::internal::DType dtype,
                       double value);

struct FillStorages : kernel::StorageSchema<FillStorages> {
  kernel::StorageField<kernel::OperandId::Output> output;

  ORTEAF_EXTRACT_STORAGES(output)
};

struct FillParams : kernel::ParamSchema<FillParams> {
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

void fillExecute(
    ::orteaf::internal::execution::cpu::resource::CpuKernelBase & /*base*/,
    kernel::KernelArgs &args) {
  auto storages = FillStorages::extract(args);
  auto params = FillParams::extract(args);

  using AnyBinding = kernel::KernelArgs::StorageListType::Storage::value_type;
  auto &lease = storages.output.lease<AnyBinding>();
  auto &cpu_storage = storages.output.payloadAs<
      AnyBinding, ::orteaf::internal::storage::CpuStorageLease>(
      "Fill kernel requires CPU storage for Output",
      "Fill kernel output storage payload is unavailable");

  const auto raw_storage_numel = lease.numel();
  if (raw_storage_numel >
      static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "Fill kernel storage size exceeds index range");
  }
  const auto storage_numel =
      static_cast<std::int64_t>(raw_storage_numel);
  const auto shape = params.shape.get();
  const auto strides = params.strides.get();
  const auto offset = params.offset.get();
  const auto stats = common_layout::analyzeLayout(shape, strides, offset,
                                                  "Fill kernel", "output");

  auto view = cpu_storage.bufferView();
  if (stats.has_zero) {
    return;
  }

  if (stats.min_index < 0 || stats.max_index < 0 ||
      stats.max_index >= storage_numel) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "Fill kernel view exceeds output storage bounds");
  }

  if (stats.contiguous) {
    const auto elem_size = ::orteaf::internal::sizeOf(lease.dtype());
    if (offset < 0 ||
        static_cast<std::uint64_t>(offset) >
            std::numeric_limits<std::size_t>::max() / elem_size) {
      error::throwError(error::OrteafErrc::InvalidParameter,
                        "Fill kernel byte offset exceeds addressable range");
    }
    const auto byte_offset =
        static_cast<std::size_t>(offset) * elem_size;
    auto *data = static_cast<std::byte *>(view.data()) + byte_offset;
    fillTensor(data, stats.numel, lease.dtype(), params.value.get());
    return;
  }

  fillTensorStrided(view.data(),
                    std::span<const std::int64_t>(shape.data, shape.size),
                    std::span<const std::int64_t>(strides.data, strides.size),
                    offset, lease.dtype(), params.value.get());
}

kernel::core::KernelMetadataLease createFillMetadata() {
  using CpuExecutionApi =
      ::orteaf::internal::execution::cpu::api::CpuExecutionApi;
  auto metadata_lease = CpuExecutionApi::acquireKernelMetadata(fillExecute);
  return kernel::core::KernelMetadataLease{std::move(metadata_lease)};
}

void registerFillKernel(
    ::orteaf::internal::architecture::Architecture architecture =
        ::orteaf::internal::architecture::Architecture::CpuGeneric) {
  auto metadata = createFillMetadata();
  auto key = ::orteaf::internal::kernel::kernel_key::makeAnyDType(
      ::orteaf::internal::ops::Op::Fill, architecture,
      static_cast<::orteaf::internal::kernel::Layout>(0),
      static_cast<::orteaf::internal::kernel::Variant>(0));
  ::orteaf::internal::kernel::api::KernelRegistryApi::registerKernel(
      key, std::move(metadata));
}

void registerFillKernelDefault() { registerFillKernel(); }

ORTEAF_REGISTER_KERNEL(
    ::orteaf::extension::kernel::cpu::registerFillKernelDefault);

}  // namespace orteaf::extension::kernel::cpu

#endif  // ORTEAF_ENABLE_CPU
