#pragma once

#if ORTEAF_ENABLE_CPU

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>

#include <orteaf/extension/kernel/cpu/print.h>
#include <orteaf/internal/base/inline_vector.h>
#include <orteaf/internal/diagnostics/error/error.h>
#include <orteaf/internal/execution/cpu/api/cpu_execution_api.h>
#include <orteaf/internal/kernel/core/kernel_args.h>
#include <orteaf/internal/kernel/core/kernel_entry.h>
#include <orteaf/internal/kernel/core/kernel_metadata.h>
#include <orteaf/internal/kernel/param/param_id.h>
#include <orteaf/internal/kernel/param/transform/array_view_inline_vector.h>
#include <orteaf/internal/kernel/schema/kernel_param_schema.h>
#include <orteaf/internal/kernel/schema/kernel_storage_schema.h>
#include <orteaf/internal/storage/registry/storage_types.h>

namespace orteaf::extension::kernel::cpu::ops {

namespace kernel = ::orteaf::internal::kernel;
namespace error = ::orteaf::internal::diagnostics::error;

inline constexpr std::uint8_t kPrintShapeInlineCapacity = 8;

struct PrintStorages : kernel::StorageSchema<PrintStorages> {
  kernel::StorageField<kernel::OperandId::Input0> input;
  kernel::OptionalStorageField<kernel::OperandId::Output> output;

  ORTEAF_EXTRACT_STORAGES(input, output)
};

struct PrintParams : kernel::ParamSchema<PrintParams> {
  using ShapeVector =
      ::orteaf::internal::base::InlineVector<std::int64_t,
                                             kPrintShapeInlineCapacity>;

  kernel::ScopedField<kernel::ParamId::Shape, ShapeVector,
                      kernel::OperandId::Input0>
      shape;

  ORTEAF_EXTRACT_FIELDS(shape)
};

inline void printExecute(
    ::orteaf::internal::execution::cpu::resource::CpuKernelBase & /*base*/,
    kernel::KernelArgs &args) {
  auto storages = PrintStorages::extract(args);
  auto params = PrintParams::extract(args);

  using AnyBinding = kernel::KernelArgs::StorageListType::Storage::value_type;
  auto &lease = storages.input.lease<AnyBinding>();
  auto *cpu_lease = lease.tryAs<::orteaf::internal::storage::CpuStorageLease>();
  if (!cpu_lease || !(*cpu_lease)) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "Print kernel requires CPU storage for Input0");
  }

  auto view = (*cpu_lease)->bufferView();
  const void *data = view.data();
  const auto dtype = lease.dtype();
  const auto &shape = params.shape.get();

  ::orteaf::extension::kernel::cpu::printTensor(
      std::span<const std::int64_t>(shape.data, shape.size), data, dtype,
      std::cout);
  std::cout << '\n';
}

inline kernel::core::KernelMetadataLease createPrintMetadata() {
  using CpuExecutionApi =
      ::orteaf::internal::execution::cpu::api::CpuExecutionApi;
  auto metadata_lease = CpuExecutionApi::acquireKernelMetadata(printExecute);
  return kernel::core::KernelMetadataLease{std::move(metadata_lease)};
}

}  // namespace orteaf::extension::kernel::cpu::ops

#endif  // ORTEAF_ENABLE_CPU
