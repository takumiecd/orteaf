#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include <orteaf/internal/dtype/dtype.h>

namespace orteaf::internal::ops {

/// @brief Enumerates operations defined in `configs/ops/ops.yml`.
///
/// The ordering matches `ops.def`, making the enum values stable indices into the
/// generated metadata tables.
enum class Op : std::uint16_t {
#define ORTEAF_OP(ID, DISPLAY_NAME, CATEGORY) ID,
#include <orteaf/ops/ops.def>
#undef ORTEAF_OP
    Count,
};

/// @brief Convert an op to its generated-table index.
constexpr std::size_t ToIndex(Op op) {
    return static_cast<std::size_t>(op);
}

inline constexpr std::size_t kOpCount = static_cast<std::size_t>(Op::Count);

}  // namespace orteaf::internal::ops

#include <orteaf/ops/ops_tables.h>

namespace orteaf::internal::ops {

namespace tables = ::orteaf::generated::ops_tables;

static_assert(kOpCount == tables::kOpCount, "Op enum size must match generated table size");

inline constexpr std::array<std::string_view, kOpCount> kOpIds = {
#define ORTEAF_OP(ID, DISPLAY_NAME, CATEGORY) std::string_view{#ID},
#include <orteaf/ops/ops.def>
#undef ORTEAF_OP
};

inline constexpr std::array<Op, kOpCount> kAllOps = {
#define ORTEAF_OP(ID, DISPLAY_NAME, CATEGORY) Op::ID,
#include <orteaf/ops/ops.def>
#undef ORTEAF_OP
};

using DTypeConstraintMode = tables::DTypeConstraintMode;
using DTypeRuleKind = tables::DTypeRuleKind;
using ComputePolicyKind = tables::ComputePolicyKind;
using InputSpec = tables::InputSpec;
using OutputSpec = tables::OutputSpec;
using AttributeSpec = tables::AttributeSpec;
using ComputePolicySpec = tables::ComputePolicySpec;
using ShapeInferenceSpec = tables::ShapeInferenceSpec;
using MetadataSpec = tables::MetadataSpec;

constexpr bool IsValidIndex(std::size_t index) {
    return index < kOpCount;
}

/// @brief Convert an index back into the enum value.
constexpr Op FromIndex(std::size_t index) {
    return static_cast<Op>(index);
}

/// @brief Return the YAML identifier (e.g. `"MatMul"`).
constexpr std::string_view IdOf(Op op) {
    return kOpIds[ToIndex(op)];
}

/// @brief Return the display name.
inline constexpr std::string_view DisplayNameOf(Op op) {
    return tables::kOpDisplayNames[ToIndex(op)];
}

/// @brief Return the category string.
inline constexpr std::string_view CategoryOf(Op op) {
    return tables::kOpCategories[ToIndex(op)];
}

/// @brief Return the arity (number of inputs).
inline constexpr std::uint32_t ArityOf(Op op) {
    return tables::kOpArity[ToIndex(op)];
}

/// @brief Return the compute-policy specification.
inline constexpr const ComputePolicySpec& ComputePolicyOf(Op op) {
    return tables::kOpComputePolicies[ToIndex(op)];
}

/// @brief Return the shape-inference specification.
inline constexpr const ShapeInferenceSpec& ShapeInferenceOf(Op op) {
    return tables::kOpShapeInference[ToIndex(op)];
}

/// @brief Return the metadata block (tags, aliases, etc.).
inline constexpr const MetadataSpec& MetadataOf(Op op) {
    return tables::kOpMetadata[ToIndex(op)];
}

template <typename T, std::size_t N>
constexpr std::span<const T> Slice(const std::array<T, N>& data, tables::Range range) {
    return std::span<const T>(data.data() + range.offset, range.count);
}

inline constexpr std::span<const InputSpec> InputsOf(Op op) {
    return Slice(tables::kOpInputSpecs, tables::kOpInputRanges[ToIndex(op)]);
}

/// @brief Return the list of output specs.
inline constexpr std::span<const OutputSpec> OutputsOf(Op op) {
    return Slice(tables::kOpOutputSpecs, tables::kOpOutputRanges[ToIndex(op)]);
}

/// @brief Return the list of attribute specs.
inline constexpr std::span<const AttributeSpec> AttributesOf(Op op) {
    return Slice(tables::kOpAttributes, tables::kOpAttributeRanges[ToIndex(op)]);
}

/// @brief Return the metadata tags.
inline constexpr std::span<const std::string_view> TagsOf(Op op) {
    return Slice(tables::kOpMetadataTags, MetadataOf(op).tags);
}

/// @brief Return the metadata aliases.
inline constexpr std::span<const std::string_view> AliasesOf(Op op) {
    return Slice(tables::kOpMetadataAliases, MetadataOf(op).aliases);
}

/// @brief Enumerate every op.
inline constexpr std::span<const Op> AllOps() {
    return std::span<const Op>(kAllOps.data(), kAllOps.size());
}

/// @brief Enumerate every op identifier.
inline constexpr std::span<const std::string_view> AllOpIds() {
    return std::span<const std::string_view>(kOpIds.data(), kOpIds.size());
}

}  // namespace orteaf::internal::ops
