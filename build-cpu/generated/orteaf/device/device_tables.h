// Auto-generated. Do not edit.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace orteaf::generated::device_tables {
inline constexpr std::size_t kDeviceCount = 8;
inline constexpr std::size_t kDeviceDTypeEntryCount = 22;
inline constexpr std::size_t kDeviceOpEntryCount = 30;
inline constexpr std::size_t kDeviceCapabilityEntryCount = 15;

inline constexpr std::array<std::uint16_t, kDeviceCount> kDeviceBackendIndices = {
    0,
    0,
    0,
    1,
    1,
    1,
    2,
    2,
};

inline constexpr std::array<std::uint16_t, kDeviceCount> kDeviceArchitectureLocalIndices = {
    0,
    3,
    2,
    0,
    2,
    3,
    0,
    1,
};

inline constexpr std::array<std::uint64_t, kDeviceCount> kDeviceMemoryMaxBytes = {
    4294967296,
    85899345920,
    8589934592,
    2147483648,
    34359738368,
    25769803776,
    17179869184,
    34359738368,
};

inline constexpr std::array<std::uint64_t, kDeviceCount> kDeviceMemorySharedBytes = {
    49152,
    229376,
    102400,
    0,
    0,
    0,
    0,
    0,
};

inline constexpr std::array<std::string_view, kDeviceCount> kDeviceNotes = {
    "フォールバック用のCUDAデバイス定義",
    "CUDA 12.2 以上の環境を推奨",
    "Desktop Ampere GPU",
    "",
    "",
    "Apple Silicon M4 Pro unified memory GPU",
    "",
    "",
};

inline constexpr std::array<std::size_t, 9> kDeviceDTypeOffsets = {
    0,
    2,
    6,
    10,
    11,
    13,
    15,
    19,
    22,
};

inline constexpr std::array<std::uint16_t, 22> kDeviceDTypeIndices = {
    12,
    11,
    12,
    11,
    9,
    10,
    12,
    11,
    9,
    10,
    12,
    12,
    11,
    12,
    11,
    12,
    13,
    3,
    4,
    12,
    13,
    11,
};

inline constexpr std::array<std::size_t, 9> kDeviceOpOffsets = {
    0,
    3,
    7,
    11,
    13,
    17,
    21,
    25,
    30,
};

inline constexpr std::array<std::uint16_t, 30> kDeviceOpIndices = {
    0,
    1,
    2,
    0,
    1,
    2,
    4,
    0,
    1,
    2,
    4,
    0,
    2,
    0,
    1,
    2,
    4,
    0,
    1,
    2,
    4,
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

inline constexpr std::array<std::size_t, 9> kDeviceCapabilityOffsets = {
    0,
    2,
    5,
    8,
    9,
    11,
    13,
    14,
    15,
};

struct CapabilityKV {
    std::string_view key;
    std::string_view value;
};

inline constexpr std::array<CapabilityKV, kDeviceCapabilityEntryCount> kDeviceCapabilityEntries = {
    CapabilityKV{"compute_capability", "sm50"},
    CapabilityKV{"tensor_cores", "false"},
    CapabilityKV{"compute_capability", "sm90"},
    CapabilityKV{"tensor_cores", "true"},
    CapabilityKV{"pcie_bandwidth_gbps", "128"},
    CapabilityKV{"compute_capability", "sm86"},
    CapabilityKV{"tensor_cores", "true"},
    CapabilityKV{"memory_type", "GDDR6"},
    CapabilityKV{"metal_version", "3.0"},
    CapabilityKV{"metal_version", "3.1"},
    CapabilityKV{"supports_attention", "true"},
    CapabilityKV{"metal_version", "3.2"},
    CapabilityKV{"neural_engine_accel", "true"},
    CapabilityKV{"isa", "SSE4.2"},
    CapabilityKV{"isa", "AVX-512 + BF16"},
};

}  // namespace orteaf::generated::device_tables
