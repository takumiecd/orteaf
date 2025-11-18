// Auto-generated. Do not edit.
#pragma once

#include <array>
#include <cstddef>
#include <string_view>

namespace orteaf::generated::backend_tables {
inline constexpr std::size_t kBackendCount = 3;

inline constexpr std::array<std::string_view, kBackendCount> kBackendDisplayNames = {
    "CUDA",
    "MPS",
    "CPU",
};

inline constexpr std::array<std::string_view, kBackendCount> kBackendModulePaths = {
    "@orteaf/internal/backend/cuda",
    "@orteaf/internal/backend/mps",
    "@orteaf/internal/backend/cpu",
};

inline constexpr std::array<std::string_view, kBackendCount> kBackendDescriptions = {
    "NVIDIA CUDA 実装",
    "macOS/iOS 向け Metal Performance Shaders 実装",
    "汎用 CPU 実装",
};

}  // namespace orteaf::generated::backend_tables
