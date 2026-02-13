#include <orteaf/extension/ops/dense/dense_tensor_ops.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <orteaf/internal/dtype/dtype.h>
#include <orteaf/internal/execution/execution.h>
#include <orteaf/internal/kernel/core/key_components.h>
#include <orteaf/internal/kernel/storage/operand_id.h>
#include <orteaf/internal/storage/storage_lease.h>

#include "detail/dense_op_common.h"
#include "detail/dense_transfer_common.h"

namespace orteaf::extension::ops::dense {

namespace {

using DenseTensorImpl = ::orteaf::extension::tensor::DenseTensorImpl;
namespace error = ::orteaf::internal::diagnostics::error;
namespace kernel = ::orteaf::internal::kernel;
using Execution = ::orteaf::internal::execution::Execution;

#if ORTEAF_ENABLE_MPS
void copyHostToMpsImpl(const DenseTensorImpl *output_impl,
                       const DenseTensorImpl *input_impl,
                       const char *op_name) {
  detail::transfer::requireExecution(input_impl, Execution::Cpu, op_name,
                                     "input");
  detail::transfer::requireExecution(output_impl, Execution::Mps, op_name,
                                     "output");
  detail::transfer::requireMatchingShapeAndDType(output_impl, input_impl,
                                                 op_name);
  detail::transfer::ensureRankSupported(input_impl, op_name);
  detail::transfer::ensureRankSupported(output_impl, op_name);

  const auto input_stats =
      detail::transfer::validateViewBounds(input_impl, op_name, "input");
  const auto output_stats =
      detail::transfer::validateViewBounds(output_impl, op_name, "output");

  if (input_stats.numel != output_stats.numel) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      std::string(op_name) +
                          ": input/output numel must match");
  }
  if (input_stats.has_zero) {
    return;
  }

  const auto elem_size = ::orteaf::internal::sizeOf(input_impl->dtype());
  if (input_stats.numel > std::numeric_limits<std::size_t>::max() / elem_size) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      std::string(op_name) + ": byte size overflow");
  }

  const auto *input_cpu_data =
      detail::transfer::requireCpuConstData(input_impl, op_name, "input");
  auto staging_lease = detail::transfer::acquireSharedMpsStaging(
      input_impl->dtype(), input_stats.numel, op_name);
  auto *staging_data =
      detail::transfer::requireSharedMpsMutableData(staging_lease, op_name);

  detail::transfer::packStridedToContiguous(
      staging_data, input_cpu_data, elem_size,
      std::span<const std::int64_t>(input_impl->shape().data(),
                                    input_impl->shape().size()),
      std::span<const std::int64_t>(input_impl->strides().data(),
                                    input_impl->strides().size()),
      input_impl->offset(), input_stats.numel);

  auto staging_layout = DenseTensorImpl::Layout::contiguous(
      std::span<const DenseTensorImpl::Dim>(output_impl->shape().data(),
                                            output_impl->shape().size()));
  DenseTensorImpl staging_impl(
      std::move(staging_layout),
      ::orteaf::internal::storage::StorageLease::erase(std::move(staging_lease)));

  auto args = detail::makeArgsForCpuOrMps(Execution::Mps, op_name);
  staging_impl.bindAllArgs(args, kernel::OperandId::Input0);
  output_impl->bindAllArgs(args, kernel::OperandId::Output);

  const kernel::KeyRequest request{
      ::orteaf::internal::ops::Op::CopyHostToDevice, output_impl->dtype(),
      detail::architectureForArgs(args, op_name)};

  detail::dispatchOrThrow(request, args,
                          "copyHostToDevice kernel not found",
                          "copyHostToDevice kernel execution failed");

  detail::transfer::syncCurrentMpsQueue(op_name);
}

void copyMpsToHostImpl(const DenseTensorImpl *output_impl,
                       const DenseTensorImpl *input_impl,
                       const char *op_name) {
  detail::transfer::requireExecution(input_impl, Execution::Mps, op_name,
                                     "input");
  detail::transfer::requireExecution(output_impl, Execution::Cpu, op_name,
                                     "output");
  detail::transfer::requireMatchingShapeAndDType(output_impl, input_impl,
                                                 op_name);
  detail::transfer::ensureRankSupported(input_impl, op_name);
  detail::transfer::ensureRankSupported(output_impl, op_name);

  const auto input_stats =
      detail::transfer::validateViewBounds(input_impl, op_name, "input");
  const auto output_stats =
      detail::transfer::validateViewBounds(output_impl, op_name, "output");

  if (input_stats.numel != output_stats.numel) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      std::string(op_name) +
                          ": input/output numel must match");
  }
  if (input_stats.has_zero) {
    return;
  }

  const auto elem_size = ::orteaf::internal::sizeOf(input_impl->dtype());
  if (input_stats.numel > std::numeric_limits<std::size_t>::max() / elem_size) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      std::string(op_name) + ": byte size overflow");
  }

  auto staging_lease = detail::transfer::acquireSharedMpsStaging(
      input_impl->dtype(), input_stats.numel, op_name);
  auto staging_read_lease = staging_lease;

  auto staging_layout = DenseTensorImpl::Layout::contiguous(
      std::span<const DenseTensorImpl::Dim>(input_impl->shape().data(),
                                            input_impl->shape().size()));
  DenseTensorImpl staging_impl(
      std::move(staging_layout),
      ::orteaf::internal::storage::StorageLease::erase(std::move(staging_lease)));

  auto args = detail::makeArgsForCpuOrMps(Execution::Mps, op_name);
  input_impl->bindAllArgs(args, kernel::OperandId::Input0);
  staging_impl.bindAllArgs(args, kernel::OperandId::Output);

  const kernel::KeyRequest request{
      ::orteaf::internal::ops::Op::CopyDeviceToHost, input_impl->dtype(),
      detail::architectureForArgs(args, op_name)};

  detail::dispatchOrThrow(request, args,
                          "copyDeviceToHost kernel not found",
                          "copyDeviceToHost kernel execution failed");

  detail::transfer::syncCurrentMpsQueue(op_name);

  auto *output_cpu_data =
      detail::transfer::requireCpuMutableData(output_impl, op_name, "output");
  const auto *staging_data =
      detail::transfer::requireSharedMpsConstData(staging_read_lease, op_name);
  detail::transfer::unpackContiguousToStrided(
      output_cpu_data, staging_data, elem_size,
      std::span<const std::int64_t>(output_impl->shape().data(),
                                    output_impl->shape().size()),
      std::span<const std::int64_t>(output_impl->strides().data(),
                                    output_impl->strides().size()),
      output_impl->offset(), output_stats.numel);
}
#endif

} // namespace

void DenseTensorOps::copyHostToDevice(Tensor &output, const Tensor &input) {
  constexpr const char *kOpName = "copyHostToDevice";
  detail::ensureValidTensor(output, kOpName);
  detail::ensureValidTensor(input, kOpName);
  const auto *output_impl = detail::requireDenseImpl(output, kOpName);
  const auto *input_impl = detail::requireDenseImpl(input, kOpName);

  detail::transfer::requireExecution(input_impl, Execution::Cpu, kOpName,
                                     "input");

  switch (output_impl->execution()) {
#if ORTEAF_ENABLE_MPS
  case Execution::Mps:
    return copyHostToMpsImpl(output_impl, input_impl, kOpName);
#endif
#if ORTEAF_ENABLE_CUDA
  case Execution::Cuda:
    break;
#endif
  default:
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "copyHostToDevice: output must be a device tensor");
  }

#if ORTEAF_ENABLE_CUDA
  detail::transfer::requireMatchingShapeAndDType(output_impl, input_impl,
                                                 kOpName);
  detail::transfer::ensureRankSupported(input_impl, kOpName);
  detail::transfer::ensureRankSupported(output_impl, kOpName);

  const auto input_stats =
      detail::transfer::validateViewBounds(input_impl, kOpName, "input");
  const auto output_stats =
      detail::transfer::validateViewBounds(output_impl, kOpName, "output");

  if (input_stats.numel != output_stats.numel) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "copyHostToDevice: input/output numel must match");
  }
  if (input_stats.has_zero) {
    return;
  }

  const auto elem_size = ::orteaf::internal::sizeOf(input_impl->dtype());
  if (input_stats.numel > std::numeric_limits<std::size_t>::max() / elem_size) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "copyHostToDevice: byte size overflow");
  }
  const auto bytes = input_stats.numel * elem_size;

  const auto *input_cpu_data =
      detail::transfer::requireCpuConstData(input_impl, kOpName, "input");
  std::vector<std::byte> packed(bytes);
  detail::transfer::packStridedToContiguous(
      packed.data(), input_cpu_data, elem_size,
      std::span<const std::int64_t>(input_impl->shape().data(),
                                    input_impl->shape().size()),
      std::span<const std::int64_t>(input_impl->strides().data(),
                                    input_impl->strides().size()),
      input_impl->offset(), input_stats.numel);

  auto staging_lease = detail::transfer::acquireCudaStaging(
      input_impl->dtype(), input_stats.numel, kOpName);
  detail::transfer::copyHostToCudaStaging(packed.data(), staging_lease, bytes,
                                          kOpName);

  auto staging_layout = DenseTensorImpl::Layout::contiguous(
      std::span<const DenseTensorImpl::Dim>(output_impl->shape().data(),
                                            output_impl->shape().size()));
  DenseTensorImpl staging_impl(
      std::move(staging_layout),
      ::orteaf::internal::storage::StorageLease::erase(std::move(staging_lease)));

  auto args = detail::makeArgsForCpuOrCuda(Execution::Cuda, kOpName);
  staging_impl.bindAllArgs(args, kernel::OperandId::Input0);
  output_impl->bindAllArgs(args, kernel::OperandId::Output);

  const kernel::KeyRequest request{
      ::orteaf::internal::ops::Op::CopyHostToDevice, output_impl->dtype(),
      detail::architectureForArgs(args, kOpName)};

  detail::dispatchOrThrow(request, args, "copyHostToDevice kernel not found",
                          "copyHostToDevice kernel execution failed");
#else
  detail::throwExecutionUnavailable(kOpName, "CUDA");
#endif
}

void DenseTensorOps::copyDeviceToHost(Tensor &output, const Tensor &input) {
  constexpr const char *kOpName = "copyDeviceToHost";
  detail::ensureValidTensor(output, kOpName);
  detail::ensureValidTensor(input, kOpName);
  const auto *output_impl = detail::requireDenseImpl(output, kOpName);
  const auto *input_impl = detail::requireDenseImpl(input, kOpName);

  detail::transfer::requireExecution(output_impl, Execution::Cpu, kOpName,
                                     "output");

  switch (input_impl->execution()) {
#if ORTEAF_ENABLE_MPS
  case Execution::Mps:
    return copyMpsToHostImpl(output_impl, input_impl, kOpName);
#endif
#if ORTEAF_ENABLE_CUDA
  case Execution::Cuda:
    break;
#endif
  default:
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "copyDeviceToHost: input must be a device tensor");
  }

#if ORTEAF_ENABLE_CUDA
  detail::transfer::requireMatchingShapeAndDType(output_impl, input_impl,
                                                 kOpName);
  detail::transfer::ensureRankSupported(input_impl, kOpName);
  detail::transfer::ensureRankSupported(output_impl, kOpName);

  const auto input_stats =
      detail::transfer::validateViewBounds(input_impl, kOpName, "input");
  const auto output_stats =
      detail::transfer::validateViewBounds(output_impl, kOpName, "output");

  if (input_stats.numel != output_stats.numel) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "copyDeviceToHost: input/output numel must match");
  }
  if (input_stats.has_zero) {
    return;
  }

  const auto elem_size = ::orteaf::internal::sizeOf(input_impl->dtype());
  if (input_stats.numel > std::numeric_limits<std::size_t>::max() / elem_size) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "copyDeviceToHost: byte size overflow");
  }
  const auto bytes = input_stats.numel * elem_size;

  auto staging_lease = detail::transfer::acquireCudaStaging(
      input_impl->dtype(), input_stats.numel, kOpName);
  auto staging_read_lease = staging_lease;

  auto staging_layout = DenseTensorImpl::Layout::contiguous(
      std::span<const DenseTensorImpl::Dim>(input_impl->shape().data(),
                                            input_impl->shape().size()));
  DenseTensorImpl staging_impl(
      std::move(staging_layout),
      ::orteaf::internal::storage::StorageLease::erase(std::move(staging_lease)));

  auto args = detail::makeArgsForCpuOrCuda(Execution::Cuda, kOpName);
  input_impl->bindAllArgs(args, kernel::OperandId::Input0);
  staging_impl.bindAllArgs(args, kernel::OperandId::Output);

  const kernel::KeyRequest request{
      ::orteaf::internal::ops::Op::CopyDeviceToHost, input_impl->dtype(),
      detail::architectureForArgs(args, kOpName)};

  detail::dispatchOrThrow(request, args, "copyDeviceToHost kernel not found",
                          "copyDeviceToHost kernel execution failed");

  std::vector<std::byte> packed(bytes);
  detail::transfer::copyCudaStagingToHost(staging_read_lease, packed.data(),
                                          bytes, kOpName);

  auto *output_cpu_data =
      detail::transfer::requireCpuMutableData(output_impl, kOpName, "output");
  detail::transfer::unpackContiguousToStrided(
      output_cpu_data, packed.data(), elem_size,
      std::span<const std::int64_t>(output_impl->shape().data(),
                                    output_impl->shape().size()),
      std::span<const std::int64_t>(output_impl->strides().data(),
                                    output_impl->strides().size()),
      output_impl->offset(), output_stats.numel);
#else
  detail::throwExecutionUnavailable(kOpName, "CUDA");
#endif
}

} // namespace orteaf::extension::ops::dense
