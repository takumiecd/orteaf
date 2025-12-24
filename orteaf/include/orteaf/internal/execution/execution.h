#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace orteaf::internal::execution {

/// @brief Enumerates the available backends (CPU / CUDA / MPS).
///
/// The ordering matches `configs/backend/backends.yml`, so these values can be used
/// as direct indices into the generated metadata tables.
enum class Execution : std::uint16_t {
#define BACKEND(ID, DISPLAY_NAME, MODULE_PATH, DESCRIPTION) ID,
#include <orteaf/backend/backend.def>
#undef BACKEND
    Count,
};

/// @brief Convert a backend to its generated-table index.
constexpr std::size_t toIndex(Execution execution) {
    return static_cast<std::size_t>(execution);
}

inline constexpr std::size_t kBackendCount = static_cast<std::size_t>(Execution::Count);

}  // namespace orteaf::internal::execution

#include <orteaf/backend/backend_tables.h>

namespace orteaf::internal::execution {

namespace tables = ::orteaf::generated::backend_tables;

static_assert(kBackendCount == tables::kBackendCount,
              "Backend enum size must match generated table size");

inline constexpr std::array<std::string_view, kBackendCount> kBackendIds = {
#define BACKEND(ID, DISPLAY_NAME, MODULE_PATH, DESCRIPTION) std::string_view{#ID},
#include <orteaf/backend/backend.def>
#undef BACKEND
};

inline constexpr std::array<Execution, kBackendCount> kAllBackends = {
#define BACKEND(ID, DISPLAY_NAME, MODULE_PATH, DESCRIPTION) Execution::ID,
#include <orteaf/backend/backend.def>
#undef BACKEND
};

/// @brief Return whether an index is within range.
constexpr bool isValidIndex(std::size_t index) {
    return index < kBackendCount;
}

/// @brief Convert an index back into the enum value.
constexpr Execution fromIndex(std::size_t index) {
    return static_cast<Execution>(index);
}

/// @brief Return the YAML identifier (e.g. `"Cuda"`).
constexpr std::string_view idOf(Execution execution) {
    return kBackendIds[toIndex(execution)];
}

/// @brief Return the display name.
inline constexpr std::string_view displayNameOf(Execution execution) {
    return tables::kBackendDisplayNames[toIndex(execution)];
}

inline constexpr std::string_view modulePathOf(Execution execution) {
    return tables::kBackendModulePaths[toIndex(execution)];
}

inline constexpr std::string_view descriptionOf(Execution execution) {
    return tables::kBackendDescriptions[toIndex(execution)];
}

inline constexpr std::span<const Execution> allBackends() {
    return std::span<const Execution>(kAllBackends.data(), kAllBackends.size());
}

}  // namespace orteaf::internal::execution
