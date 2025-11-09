#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace orteaf::internal::backend {

enum class Backend : std::uint16_t {
#define BACKEND(ID, DISPLAY_NAME, MODULE_PATH, DESCRIPTION) ID,
#include <orteaf/backend/backend.def>
#undef BACKEND
    Count,
};

constexpr std::size_t ToIndex(Backend backend) {
    return static_cast<std::size_t>(backend);
}

inline constexpr std::size_t kBackendCount = static_cast<std::size_t>(Backend::Count);

}  // namespace orteaf::internal::backend

#include <orteaf/backend/backend_tables.h>

namespace orteaf::internal::backend {

namespace tables = ::orteaf::generated::backend_tables;

static_assert(kBackendCount == tables::kBackendCount,
              "Backend enum size must match generated table size");

inline constexpr std::array<std::string_view, kBackendCount> kBackendIds = {
#define BACKEND(ID, DISPLAY_NAME, MODULE_PATH, DESCRIPTION) std::string_view{#ID},
#include <orteaf/backend/backend.def>
#undef BACKEND
};

inline constexpr std::array<Backend, kBackendCount> kAllBackends = {
#define BACKEND(ID, DISPLAY_NAME, MODULE_PATH, DESCRIPTION) Backend::ID,
#include <orteaf/backend/backend.def>
#undef BACKEND
};

constexpr bool IsValidIndex(std::size_t index) {
    return index < kBackendCount;
}

constexpr Backend FromIndex(std::size_t index) {
    return static_cast<Backend>(index);
}

constexpr std::string_view IdOf(Backend backend) {
    return kBackendIds[ToIndex(backend)];
}

inline constexpr std::string_view DisplayNameOf(Backend backend) {
    return tables::kBackendDisplayNames[ToIndex(backend)];
}

inline constexpr std::string_view ModulePathOf(Backend backend) {
    return tables::kBackendModulePaths[ToIndex(backend)];
}

inline constexpr std::string_view DescriptionOf(Backend backend) {
    return tables::kBackendDescriptions[ToIndex(backend)];
}

inline constexpr std::span<const Backend> AllBackends() {
    return std::span<const Backend>(kAllBackends.data(), kAllBackends.size());
}

}  // namespace orteaf::internal::backend
