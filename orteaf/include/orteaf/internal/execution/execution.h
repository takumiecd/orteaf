#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace orteaf::internal::execution {

/// @brief Enumerates the available executions (CPU / CUDA / MPS).
///
/// The ordering matches `configs/execution/executions.yml`, so these values can be used
/// as direct indices into the generated metadata tables.
enum class Execution : std::uint16_t {
#define EXECUTION(ID, DISPLAY_NAME, MODULE_PATH, DESCRIPTION) ID,
#include <orteaf/execution/execution.def>
#undef EXECUTION
    Count,
};

/// @brief Convert a execution to its generated-table index.
constexpr std::size_t toIndex(Execution execution) {
    return static_cast<std::size_t>(execution);
}

inline constexpr std::size_t kExecutionCount = static_cast<std::size_t>(Execution::Count);

}  // namespace orteaf::internal::execution

#include <orteaf/execution/execution_tables.h>

namespace orteaf::internal::execution {

namespace tables = ::orteaf::generated::execution_tables;

static_assert(kExecutionCount == tables::kExecutionCount,
              "Execution enum size must match generated table size");

inline constexpr std::array<std::string_view, kExecutionCount> kExecutionIds = {
#define EXECUTION(ID, DISPLAY_NAME, MODULE_PATH, DESCRIPTION) std::string_view{#ID},
#include <orteaf/execution/execution.def>
#undef EXECUTION
};

inline constexpr std::array<Execution, kExecutionCount> kAllExecutions = {
#define EXECUTION(ID, DISPLAY_NAME, MODULE_PATH, DESCRIPTION) Execution::ID,
#include <orteaf/execution/execution.def>
#undef EXECUTION
};

/// @brief Return whether an index is within range.
constexpr bool isValidIndex(std::size_t index) {
    return index < kExecutionCount;
}

/// @brief Convert an index back into the enum value.
constexpr Execution fromIndex(std::size_t index) {
    return static_cast<Execution>(index);
}

/// @brief Return the YAML identifier (e.g. `"Cuda"`).
constexpr std::string_view idOf(Execution execution) {
    return kExecutionIds[toIndex(execution)];
}

/// @brief Return the display name.
inline constexpr std::string_view displayNameOf(Execution execution) {
    return tables::kExecutionDisplayNames[toIndex(execution)];
}

inline constexpr std::string_view modulePathOf(Execution execution) {
    return tables::kExecutionModulePaths[toIndex(execution)];
}

inline constexpr std::string_view descriptionOf(Execution execution) {
    return tables::kExecutionDescriptions[toIndex(execution)];
}

inline constexpr std::span<const Execution> allExecutions() {
    return std::span<const Execution>(kAllExecutions.data(), kAllExecutions.size());
}

}  // namespace orteaf::internal::execution
