#if ORTEAF_ENABLE_CPU

#include <cstddef>
#include <cstdint>
#include <limits>

#include <orteaf/internal/base/inline_vector.h>
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

namespace orteaf::extension::kernel::cpu {

namespace kernel = ::orteaf::internal::kernel;
namespace error = ::orteaf::internal::diagnostics::error;

void fillTensor(void *data, std::size_t count,
                ::orteaf::internal::DType dtype, double value);

inline constexpr std::uint8_t kFillShapeInlineCapacity = 8;

struct FillStorages : kernel::StorageSchema<FillStorages> {
  kernel::StorageField<kernel::OperandId::Output> output;

  ORTEAF_EXTRACT_STORAGES(output)
};

struct FillParams : kernel::ParamSchema<FillParams> {
  using ShapeVector =
      ::orteaf::internal::base::InlineVector<std::int64_t,
                                             kFillShapeInlineCapacity>;

  kernel::Field<kernel::ParamId::Value, double> value;
  kernel::OptionalField<kernel::ParamId::Shape, ShapeVector> shape;

  ORTEAF_EXTRACT_FIELDS(value, shape)
};

namespace {

std::size_t computeShapeSize(const FillParams::ShapeVector &shape) {
  std::size_t product = 1;
  constexpr std::size_t kMaxSize =
      std::numeric_limits<std::size_t>::max();
  for (std::uint8_t i = 0; i < shape.size; ++i) {
    const auto dim = shape.data[i];
    if (dim < 0) {
      error::throwError(error::OrteafErrc::InvalidParameter,
                        "Fill kernel received negative shape dimension");
    }
    const std::size_t dim_size = static_cast<std::size_t>(dim);
    if (dim_size != 0 && product > kMaxSize / dim_size) {
      error::throwError(error::OrteafErrc::InvalidParameter,
                        "Fill kernel shape is too large");
    }
    product *= dim_size;
  }
  return product;
}

}  // namespace

void fillExecute(
    ::orteaf::internal::execution::cpu::resource::CpuKernelBase & /*base*/,
    kernel::KernelArgs &args) {
  auto storages = FillStorages::extract(args);
  auto params = FillParams::extract(args);

  using AnyBinding = kernel::KernelArgs::StorageListType::Storage::value_type;
  auto &lease = storages.output.lease<AnyBinding>();
  auto *cpu_lease = lease.tryAs<::orteaf::internal::storage::CpuStorageLease>();
  if (!cpu_lease || !(*cpu_lease)) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "Fill kernel requires CPU storage for Output");
  }

  const std::size_t numel = lease.numel();
  if (params.shape) {
    const auto expected = computeShapeSize(params.shape.value);
    if (expected != numel) {
      error::throwError(error::OrteafErrc::InvalidParameter,
                        "Fill kernel shape does not match output size");
    }
  }

  auto view = (*cpu_lease)->bufferView();
  fillTensor(view.data(), numel, lease.dtype(), params.value.get());
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
