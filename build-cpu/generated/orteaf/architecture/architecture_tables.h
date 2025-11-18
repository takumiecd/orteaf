// Auto-generated. Do not edit.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace orteaf::generated::architecture_tables {
inline constexpr std::size_t kArchitectureCount = 13;
inline constexpr std::size_t kBackendCount = 3;

inline constexpr std::array<std::uint16_t, kArchitectureCount> kArchitectureBackendIndices = {
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    2,
    2,
    2,
    2,
    2,
};

inline constexpr std::array<std::uint16_t, kArchitectureCount> kArchitectureLocalIndices = {
    0,
    1,
    2,
    3,
    0,
    1,
    2,
    3,
    0,
    1,
    2,
    3,
    4,
};

inline constexpr std::array<std::string_view, kArchitectureCount> kArchitectureIds = {
    "generic",
    "sm80",
    "sm86",
    "sm90",
    "generic",
    "m2",
    "m3",
    "m4",
    "generic",
    "zen4",
    "skylake",
    "apple_m4_pro",
    "intel_comet_lake",
};

inline constexpr std::array<std::string_view, kArchitectureCount> kArchitectureDisplayNames = {
    "Generic CUDA",
    "CUDA SM80",
    "CUDA SM86",
    "CUDA SM90",
    "Generic MPS",
    "Apple M2 GPU",
    "Apple M3 GPU",
    "Apple M4 GPU",
    "Generic CPU",
    "Zen 4 AVX512",
    "Skylake AVX512",
    "Apple M4 Pro",
    "Intel Comet Lake",
};

inline constexpr std::array<std::string_view, kArchitectureCount> kArchitectureDescriptions = {
    "Backend-wide fallback architecture for CUDA",
    "Ampere 世代 GPU (A100 など) 向け最適化",
    "Ampere 世代 GPU (RTX 30 シリーズ) 向け最適化",
    "Hopper 世代 GPU (H100 など) 向け最適化",
    "Backend-wide fallback architecture for MPS",
    "Apple Silicon M2 世代 GPU 向け最適化",
    "Apple Silicon M3 世代 GPU 向け最適化",
    "Apple Silicon M4 世代 GPU 向け最適化",
    "Backend-wide fallback architecture for CPU",
    "Zen 4 / AVX-512 対応 CPU 向け最適化",
    "Intel Skylake-X / AVX-512 対応 CPU 向け最適化",
    "Apple Silicon M4 Pro アーキテクチャ (Avalanche/Blizzard) 向け最適化",
    "Comet Lake ファミリ (Core i9-10850K など) 向けの AVX2 最適化",
};

inline constexpr std::array<std::size_t, 3> kBackendArchitectureCounts = {
    4,
    4,
    5,
};

inline constexpr std::array<std::size_t, 4> kBackendArchitectureOffsets = {
    0,
    4,
    8,
    13,
};

inline constexpr std::array<std::string_view, kArchitectureCount> kArchitectureDetectVendors = {
    std::string_view{""},
    std::string_view{"nvidia"},
    std::string_view{"nvidia"},
    std::string_view{"nvidia"},
    std::string_view{""},
    std::string_view{"apple"},
    std::string_view{"apple"},
    std::string_view{"apple"},
    std::string_view{""},
    std::string_view{"amd"},
    std::string_view{"intel"},
    std::string_view{"apple"},
    std::string_view{"intel"},
};

inline constexpr std::array<int, kArchitectureCount> kArchitectureDetectCpuFamilies = {
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    25,
    6,
    -1,
    6,
};

inline constexpr std::array<int, kArchitectureCount> kArchitectureDetectComputeCapability = {
    -1,
    80,
    86,
    90,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
};

inline constexpr std::array<std::string_view, kArchitectureCount> kArchitectureDetectMetalFamilies = {
    std::string_view{""},
    std::string_view{""},
    std::string_view{""},
    std::string_view{""},
    std::string_view{""},
    std::string_view{"m2"},
    std::string_view{"m3"},
    std::string_view{"m4"},
    std::string_view{""},
    std::string_view{""},
    std::string_view{""},
    std::string_view{""},
    std::string_view{""},
};

inline constexpr std::array<std::size_t, 14> kArchitectureDetectCpuModelOffsets = {
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    3,
};

inline constexpr std::array<int, 3> kArchitectureDetectCpuModels = {
    85,
    165,
    166,
};

inline constexpr std::array<std::size_t, 14> kArchitectureDetectFeatureOffsets = {
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    2,
    2,
    3,
};

inline constexpr std::array<std::string_view, 3> kArchitectureDetectFeatures = {
    std::string_view{"avx512"},
    std::string_view{"avx512"},
    std::string_view{"avx2"},
};

inline constexpr std::array<std::size_t, 14> kArchitectureDetectMachineIdOffsets = {
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    6,
    6,
};

inline constexpr std::array<std::string_view, 6> kArchitectureDetectMachineIds = {
    std::string_view{"Mac15,4"},
    std::string_view{"Mac15,6"},
    std::string_view{"Mac15,7"},
    std::string_view{"Mac16,6"},
    std::string_view{"Mac16,7"},
    std::string_view{"Mac16,8"},
};

}  // namespace orteaf::generated::architecture_tables
