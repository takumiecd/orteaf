#pragma once

#include <type_traits>
#include <utility>
#include <variant>

#include <orteaf/internal/execution/execution.h>
#include <orteaf/internal/execution_context/cpu/context.h>

#if ORTEAF_ENABLE_MPS
#include <orteaf/internal/execution_context/mps/context.h>
#endif

#if ORTEAF_ENABLE_CUDA
#include <orteaf/internal/execution_context/cuda/context.h>
#endif

namespace orteaf::internal::kernel {

using CpuContext = ::orteaf::internal::execution_context::cpu::Context;

#if ORTEAF_ENABLE_MPS
using MpsContext = ::orteaf::internal::execution_context::mps::Context;
#endif

#if ORTEAF_ENABLE_CUDA
using CudaContext = ::orteaf::internal::execution_context::cuda::Context;
#endif

/**
 * @brief Execution mapping for context types.
 */
template <typename T> struct ContextExecution;

template <> struct ContextExecution<CpuContext> {
  static constexpr ::orteaf::internal::execution::Execution kValue =
      ::orteaf::internal::execution::Execution::Cpu;
};

#if ORTEAF_ENABLE_MPS
template <> struct ContextExecution<MpsContext> {
  static constexpr ::orteaf::internal::execution::Execution kValue =
      ::orteaf::internal::execution::Execution::Mps;
};
#endif

#if ORTEAF_ENABLE_CUDA
template <> struct ContextExecution<CudaContext> {
  static constexpr ::orteaf::internal::execution::Execution kValue =
      ::orteaf::internal::execution::Execution::Cuda;
};
#endif

/**
 * @brief Concept for allowed context types.
 */
template <typename T>
concept KernelContextType = requires {
  ContextExecution<std::decay_t<T>>::kValue;
};

/**
 * @brief Type-erased execution context container.
 *
 * Uses std::variant to avoid virtual dispatch. Provides execution() mapping
 * via ContextExecution specializations.
 */
class ContextAny {
public:
  using Execution = ::orteaf::internal::execution::Execution;
  using Variant = std::variant<std::monostate, CpuContext
#if ORTEAF_ENABLE_MPS
                               ,
                               MpsContext
#endif
#if ORTEAF_ENABLE_CUDA
                               ,
                               CudaContext
#endif
                               >;

  ContextAny() = default;

  template <typename T>
    requires KernelContextType<std::decay_t<T>>
  static ContextAny erase(T ctx) {
    return ContextAny(Variant{std::move(ctx)});
  }

  template <typename T> T *tryAs() { return std::get_if<T>(&variant_); }

  template <typename T> const T *tryAs() const {
    return std::get_if<T>(&variant_);
  }

  template <typename Visitor> decltype(auto) visit(Visitor &&v) {
    return std::visit(std::forward<Visitor>(v), variant_);
  }

  template <typename Visitor> decltype(auto) visit(Visitor &&v) const {
    return std::visit(std::forward<Visitor>(v), variant_);
  }

  bool valid() const {
    return !std::holds_alternative<std::monostate>(variant_);
  }

  Execution execution() const {
    return std::visit(
        [](const auto &ctx) -> Execution {
          using T = std::decay_t<decltype(ctx)>;
          if constexpr (std::is_same_v<T, std::monostate>) {
            return Execution::Cpu;
          } else {
            static_assert(KernelContextType<T>,
                          "Context type must have Execution mapping");
            return ContextExecution<T>::kValue;
          }
        },
        variant_);
  }

private:
  explicit ContextAny(Variant v) : variant_(std::move(v)) {}

  Variant variant_{};
};

} // namespace orteaf::internal::kernel
