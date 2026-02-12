#include <orteaf/extension/ops/dense/dense_tensor_ops.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>

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

} // namespace

void DenseTensorOps::copyHostToMps(Tensor &output, const Tensor &input) {
  constexpr const char *kOpName = "copyHostToMps";
#if !ORTEAF_ENABLE_MPS
  detail::throwMpsUnavailable(kOpName);
#else
  detail::ensureValidTensor(output, kOpName);
  detail::ensureValidTensor(input, kOpName);
  const auto *output_impl = detail::requireDenseImpl(output, kOpName);
  const auto *input_impl = detail::requireDenseImpl(input, kOpName);

  detail::transfer::requireExecution(input_impl, Execution::Cpu, kOpName,
                                     "input");
  detail::transfer::requireExecution(output_impl, Execution::Mps, kOpName,
                                     "output");
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
                      "copyHostToMps: input/output numel must match");
  }
  if (input_stats.has_zero) {
    return;
  }

  const auto elem_size = ::orteaf::internal::sizeOf(input_impl->dtype());
  if (input_stats.numel > std::numeric_limits<std::size_t>::max() / elem_size) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "copyHostToMps: byte size overflow");
  }

  const auto *input_cpu_data =
      detail::transfer::requireCpuConstData(input_impl, kOpName, "input");
  auto staging_lease = detail::transfer::acquireSharedMpsStaging(
      input_impl->dtype(), input_stats.numel, kOpName);
  auto *staging_data =
      detail::transfer::requireSharedMpsMutableData(staging_lease, kOpName);

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

  auto args = detail::makeArgsForCpuOrMps(Execution::Mps, kOpName);
  staging_impl.bindAllArgs(args, kernel::OperandId::Input0);
  output_impl->bindAllArgs(args, kernel::OperandId::Output);

  const kernel::KeyRequest request{
      ::orteaf::internal::ops::Op::CopyHostToMps, output_impl->dtype(),
      detail::architectureForArgs(args, kOpName)};

  detail::dispatchOrThrow(request, args, "copyHostToMps kernel not found",
                          "copyHostToMps kernel execution failed");

  // Staging storage is temporary. Ensure transfer completion before release.
  detail::transfer::syncCurrentMpsQueue(kOpName);
#endif
}

void DenseTensorOps::copyMpsToHost(Tensor &output, const Tensor &input) {
  constexpr const char *kOpName = "copyMpsToHost";
#if !ORTEAF_ENABLE_MPS
  detail::throwMpsUnavailable(kOpName);
#else
  detail::ensureValidTensor(output, kOpName);
  detail::ensureValidTensor(input, kOpName);
  const auto *output_impl = detail::requireDenseImpl(output, kOpName);
  const auto *input_impl = detail::requireDenseImpl(input, kOpName);

  detail::transfer::requireExecution(input_impl, Execution::Mps, kOpName,
                                     "input");
  detail::transfer::requireExecution(output_impl, Execution::Cpu, kOpName,
                                     "output");
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
                      "copyMpsToHost: input/output numel must match");
  }
  if (input_stats.has_zero) {
    return;
  }

  const auto elem_size = ::orteaf::internal::sizeOf(input_impl->dtype());
  if (input_stats.numel > std::numeric_limits<std::size_t>::max() / elem_size) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "copyMpsToHost: byte size overflow");
  }

  auto staging_lease = detail::transfer::acquireSharedMpsStaging(
      input_impl->dtype(), input_stats.numel, kOpName);
  auto staging_read_lease = staging_lease;

  auto staging_layout = DenseTensorImpl::Layout::contiguous(
      std::span<const DenseTensorImpl::Dim>(input_impl->shape().data(),
                                            input_impl->shape().size()));
  DenseTensorImpl staging_impl(
      std::move(staging_layout),
      ::orteaf::internal::storage::StorageLease::erase(std::move(staging_lease)));

  auto args = detail::makeArgsForCpuOrMps(Execution::Mps, kOpName);
  input_impl->bindAllArgs(args, kernel::OperandId::Input0);
  staging_impl.bindAllArgs(args, kernel::OperandId::Output);

  const kernel::KeyRequest request{
      ::orteaf::internal::ops::Op::CopyMpsToHost, input_impl->dtype(),
      detail::architectureForArgs(args, kOpName)};

  detail::dispatchOrThrow(request, args, "copyMpsToHost kernel not found",
                          "copyMpsToHost kernel execution failed");

  detail::transfer::syncCurrentMpsQueue(kOpName);

  auto *output_cpu_data =
      detail::transfer::requireCpuMutableData(output_impl, kOpName, "output");
  const auto *staging_data =
      detail::transfer::requireSharedMpsConstData(staging_read_lease, kOpName);
  detail::transfer::unpackContiguousToStrided(
      output_cpu_data, staging_data, elem_size,
      std::span<const std::int64_t>(output_impl->shape().data(),
                                    output_impl->shape().size()),
      std::span<const std::int64_t>(output_impl->strides().data(),
                                    output_impl->strides().size()),
      output_impl->offset(), output_stats.numel);
#endif
}

} // namespace orteaf::extension::ops::dense
