#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <type_traits>

#include <orteaf/internal/execution/execution.h>

namespace orteaf::internal::architecture {

/// @brief Enumerates per-execution optimization architectures (Generic, Sm90,
/// etc.).
///
/// The generator serializes architectures as `(execution, local_index)` pairs.
/// Local index `0` is automatically reserved for the Generic entry and can be
/// used as a universal fallback.
enum class Architecture : std::uint16_t {
#define ARCHITECTURE(ENUM_NAME, EXECUTION_ENUM, LOCAL_INDEX, ID, DISPLAY_NAME, \
                     DESCRIPTION)                                              \
  ENUM_NAME,
#include <orteaf/architecture/architecture.def>
#undef ARCHITECTURE
  Count,
};

/// @brief Convert an architecture to its generated-table index.
constexpr std::size_t toIndex(Architecture arch) {
  return static_cast<std::size_t>(arch);
}

inline constexpr std::size_t kArchitectureCount =
    static_cast<std::size_t>(Architecture::Count);

} // namespace orteaf::internal::architecture

#include <orteaf/architecture/architecture_tables.h>

namespace orteaf::internal::architecture {

namespace tables = ::orteaf::generated::architecture_tables;

static_assert(kArchitectureCount == tables::kArchitectureCount,
              "Architecture enum size must match generated table size");
static_assert(execution::kExecutionCount == tables::kExecutionCount,
              "Architecture metadata must match execution count");

inline constexpr std::array<Architecture, kArchitectureCount>
    kAllArchitectures = {
#define ARCHITECTURE(ENUM_NAME, EXECUTION_ENUM, LOCAL_INDEX, ID, DISPLAY_NAME, \
                     DESCRIPTION)                                              \
  Architecture::ENUM_NAME,
#include <orteaf/architecture/architecture.def>
#undef ARCHITECTURE
};

inline constexpr std::array<std::string_view, kArchitectureCount>
    kArchitectureIds = {
#define ARCHITECTURE(ENUM_NAME, EXECUTION_ENUM, LOCAL_INDEX, ID, DISPLAY_NAME, \
                     DESCRIPTION)                                              \
  std::string_view{ID},
#include <orteaf/architecture/architecture.def>
#undef ARCHITECTURE
};

inline constexpr std::array<std::string_view, kArchitectureCount>
    kArchitectureDisplayNames = {
#define ARCHITECTURE(ENUM_NAME, EXECUTION_ENUM, LOCAL_INDEX, ID, DISPLAY_NAME, \
                     DESCRIPTION)                                              \
  std::string_view{DISPLAY_NAME},
#include <orteaf/architecture/architecture.def>
#undef ARCHITECTURE
};

/// @brief Check whether an index is within range.
constexpr bool isValidIndex(std::size_t index) {
  return index < kArchitectureCount;
}

/// @brief Convert an index back into the enum value.
constexpr Architecture fromIndex(std::size_t index) {
  return static_cast<Architecture>(index);
}

/// @brief Return the execution the architecture belongs to.
constexpr execution::Execution executionOf(Architecture arch) {
  const auto execution_index =
      tables::kArchitectureExecutionIndices[toIndex(arch)];
  return execution::fromIndex(execution_index);
}

/// @brief Return the execution-local index (0 == Generic).
constexpr std::uint16_t localIndexOf(Architecture arch) {
  return tables::kArchitectureLocalIndices[toIndex(arch)];
}

/// @brief Return true if the architecture is the Generic entry.
constexpr bool isGeneric(Architecture arch) { return localIndexOf(arch) == 0; }

/// @brief Return the YAML identifier / display name / description.
constexpr std::string_view idOf(Architecture arch) {
  return tables::kArchitectureIds[toIndex(arch)];
}

constexpr std::string_view displayNameOf(Architecture arch) {
  return tables::kArchitectureDisplayNames[toIndex(arch)];
}

constexpr std::string_view descriptionOf(Architecture arch) {
  return tables::kArchitectureDescriptions[toIndex(arch)];
}

constexpr std::size_t countForExecution(execution::Execution execution_id) {
  return tables::kExecutionArchitectureCounts[execution::toIndex(execution_id)];
}

constexpr std::size_t offsetForExecution(execution::Execution execution_id) {
  return tables::kExecutionArchitectureOffsets[execution::toIndex(
      execution_id)];
}

constexpr bool hasLocalIndex(execution::Execution execution_id,
                             std::uint16_t local_index) {
  return local_index < countForExecution(execution_id);
}

/// @brief Construct an architecture from a execution and local index.
constexpr Architecture
fromExecutionAndLocalIndex(execution::Execution execution_id,
                           std::uint16_t local_index) {
  return static_cast<Architecture>(offsetForExecution(execution_id) +
                                   local_index);
}

/// @brief Return a span of every architecture belonging to the execution.
inline constexpr std::span<const Architecture>
architecturesOf(execution::Execution execution_id) {
  const auto offset = offsetForExecution(execution_id);
  const auto count = countForExecution(execution_id);
  return std::span<const Architecture>(kAllArchitectures.data() + offset,
                                       count);
}

/// @brief Check if the architecture has a parent (non-Generic architectures
/// have parents).
constexpr bool hasParent(Architecture arch) {
  return tables::kArchitectureParentIndices[toIndex(arch)] !=
         tables::kInvalidParent;
}

/// @brief Return the parent architecture index, or kInvalidParent if none.
constexpr std::uint16_t parentIndexOf(Architecture arch) {
  return tables::kArchitectureParentIndices[toIndex(arch)];
}

/// @brief Return the parent architecture, or the architecture itself if no
/// parent (Generic).
constexpr Architecture parentOf(Architecture arch) {
  const auto parent_idx = parentIndexOf(arch);
  if (parent_idx == tables::kInvalidParent) {
    return arch; // Generic has no parent, return self
  }
  return fromIndex(parent_idx);
}

/// @brief Return the Generic architecture for the execution of this
/// architecture.
constexpr Architecture genericOf(Architecture arch) {
  return fromExecutionAndLocalIndex(executionOf(arch), 0);
}

/// @brief Invoke callback for each architecture in fallback order.
/// Order: self → parent → parent's parent → ... → Generic
/// Callback signature: void(Architecture)
/// Returns when callback returns false or when Generic is reached.
template <typename Func>
constexpr void forEachFallback(Architecture arch, Func &&func) {
  Architecture current = arch;
  while (true) {
    if constexpr (std::is_invocable_r_v<bool, Func, Architecture>) {
      if (!func(current)) {
        return;
      }
    } else {
      func(current);
    }

    if (!hasParent(current)) {
      return; // Reached Generic
    }
    current = parentOf(current);
  }
}

} // namespace orteaf::internal::architecture
