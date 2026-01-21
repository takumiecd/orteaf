#pragma once

#include <utility>
#include <variant>

#include <orteaf/internal/execution/execution.h>
#include <orteaf/internal/kernel/cpu/cpu_kernel_args.h>

#if ORTEAF_ENABLE_MPS
#include <orteaf/internal/kernel/mps/mps_kernel_args.h>
#endif

#if ORTEAF_ENABLE_CUDA
#include <orteaf/internal/kernel/cuda/cuda_kernel_args.h>
#endif

namespace orteaf::internal::kernel {

/**
 * @brief Type-erased kernel arguments container.
 *
 * Wraps backend-specific KernelArgs (Cpu/Mps/Cuda) in a std::variant,
 * providing a unified interface similar to the Storage class pattern.
 *
 */
class KernelArgs {
public:
  using Execution = ::orteaf::internal::execution::Execution;
  using CpuKernelArgsType = cpu::CpuKernelArgs;
#if ORTEAF_ENABLE_MPS
  using MpsKernelArgsType = mps::MpsKernelArgs;
#endif
#if ORTEAF_ENABLE_CUDA
  using CudaKernelArgsType = cuda::CudaKernelArgs;
#endif

  // Variant type holding all backend implementations
  using Variant = std::variant<std::monostate, CpuKernelArgsType
#if ORTEAF_ENABLE_MPS
                               ,
                               MpsKernelArgsType
#endif
                               >;

  /**
   * @brief Default constructor. Creates an invalid (uninitialized) KernelArgs.
   */
  KernelArgs() = default;

  /**
   * @brief Type-erase a backend-specific KernelArgs object.
   *
   * @tparam T A backend-specific KernelArgs type.
   * @param args The backend-specific KernelArgs to wrap.
   * @return A new KernelArgs instance containing the wrapped args.
   */
  template <typename T> static KernelArgs erase(T args) {
    return KernelArgs(Variant{std::move(args)});
  }

  /**
   * @brief Attempt to retrieve as a specific type.
   *
   * @tparam T The target KernelArgs type.
   * @return Pointer if it holds type T, nullptr otherwise.
   */
  template <typename T> T *tryAs() { return std::get_if<T>(&variant_); }

  template <typename T> const T *tryAs() const {
    return std::get_if<T>(&variant_);
  }

  /**
   * @brief Apply a visitor to the underlying KernelArgs.
   */
  template <typename Visitor> decltype(auto) visit(Visitor &&v) {
    return std::visit(std::forward<Visitor>(v), variant_);
  }

  template <typename Visitor> decltype(auto) visit(Visitor &&v) const {
    return std::visit(std::forward<Visitor>(v), variant_);
  }

  /**
   * @brief Check if the KernelArgs is initialized.
   */
  bool valid() const {
    return !std::holds_alternative<std::monostate>(variant_);
  }

  /**
   * @brief Return the execution backend for this KernelArgs.
   */
  Execution execution() const {
    return std::visit(
        [](const auto &args) -> Execution {
          using T = std::decay_t<decltype(args)>;
          if constexpr (std::is_same_v<T, std::monostate>) {
            return Execution::Cpu; // Default for invalid
          } else if constexpr (std::is_same_v<T, CpuKernelArgsType>) {
            return Execution::Cpu;
          }
#if ORTEAF_ENABLE_MPS
          else if constexpr (std::is_same_v<T, MpsKernelArgsType>) {
            return Execution::Mps;
          }
#endif
#if ORTEAF_ENABLE_CUDA
          else if constexpr (std::is_same_v<T, CudaKernelArgsType>) {
            return Execution::Cuda;
          }
#endif
          else {
            return Execution::Cpu;
          }
        },
        variant_);
  }

private:
  explicit KernelArgs(Variant v) : variant_(std::move(v)) {}

  Variant variant_{};
};

} // namespace orteaf::internal::kernel
