// Auto-generated. Do not edit.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace orteaf::generated::ops_tables {

using ::orteaf::internal::DType;

inline constexpr std::size_t kOpCount = 5;
inline constexpr std::size_t kTotalInputPortCount = 9;
inline constexpr std::size_t kTotalOutputPortCount = 5;
inline constexpr std::size_t kTotalAttributeCount = 5;
inline constexpr std::uint32_t kInvalidIndex = 0xFFFFFFFFu;

enum class DTypeConstraintMode : std::uint8_t { Allow, Deny, Match };
enum class DTypeRuleKind : std::uint8_t { Promote, SameAs, Fixed, Custom };
enum class ComputePolicyKind : std::uint8_t { SameAs, Promote, Fixed, DerivedComputeType, Custom };

struct Range {
    std::size_t offset;
    std::size_t count;
};

struct DTypeConstraintSpec {
    DTypeConstraintMode mode;
    std::uint64_t allow_mask;
    std::uint64_t deny_mask;
    std::uint32_t reference_input;
    bool allow_promotion;
    bool require_same_shape;
};

struct InputSpec {
    std::string_view name;
    std::string_view description;
    DTypeConstraintSpec dtype;
    bool optional;
};

struct OutputSpec {
    std::string_view name;
    std::string_view description;
    DTypeRuleKind kind;
    std::uint64_t input_mask;
    std::uint32_t reference_input;
    std::uint32_t fixed_dtype;
    std::string_view custom_function;
};

struct AttributeSpec {
    std::string_view name;
    std::string_view type;
    std::string_view default_value;
    bool has_default;
    bool required;
    std::string_view description;
};

struct ComputePolicySpec {
    ComputePolicyKind kind;
    std::uint64_t input_mask;
    std::uint32_t reference_input;
    std::uint32_t fixed_dtype;
    std::string_view function;
    std::string_view handler;
};

struct ShapeInferenceSpec {
    std::string_view kind;
    std::string_view function;
    std::string_view description;
};

struct MetadataSpec {
    std::string_view description;
    bool commutative;
    bool differentiable;
    Range tags;
    Range aliases;
};

inline constexpr std::array<std::string_view, kOpCount> kOpIds = {
    "Add",
    "MatMul",
    "Relu",
    "SpikeThreshold",
    "CustomQuantize",
};

inline constexpr std::array<std::string_view, kOpCount> kOpDisplayNames = {
    "Add",
    "MatMul",
    "ReLU",
    "Spike Threshold",
    "Custom Quantize",
};

inline constexpr std::array<std::string_view, kOpCount> kOpCategories = {
    "arithmetic",
    "linear_algebra",
    "activation",
    "spiking",
    "arithmetic",
};

inline constexpr std::array<std::uint32_t, kOpCount> kOpArity = {
    2,
    2,
    1,
    2,
    1,
};

inline constexpr std::array<Range, kOpCount> kOpInputRanges = {
    Range{0, 2},
    Range{2, 3},
    Range{5, 1},
    Range{6, 2},
    Range{8, 1},
};

inline constexpr std::array<InputSpec, kTotalInputPortCount> kOpInputSpecs = {
    InputSpec{"lhs", "Left-hand operand", DTypeConstraintSpec{DTypeConstraintMode::Allow, 0x3FFFULL, 0x0ULL, kInvalidIndex, false, false}, false},
    InputSpec{"rhs", "Right-hand operand", DTypeConstraintSpec{DTypeConstraintMode::Match, 0x3FFFULL, 0x0ULL, 0, true, false}, false},
    InputSpec{"lhs", "Left matrix operand ([..., M, K])", DTypeConstraintSpec{DTypeConstraintMode::Allow, 0x3E00ULL, 0x0ULL, kInvalidIndex, false, false}, false},
    InputSpec{"rhs", "Right matrix operand ([..., K, N])", DTypeConstraintSpec{DTypeConstraintMode::Allow, 0x3E00ULL, 0x0ULL, kInvalidIndex, false, false}, false},
    InputSpec{"bias", "Optional bias vector ([..., N])", DTypeConstraintSpec{DTypeConstraintMode::Allow, 0x3E00ULL, 0x0ULL, kInvalidIndex, false, false}, true},
    InputSpec{"input", "Input tensor", DTypeConstraintSpec{DTypeConstraintMode::Allow, 0x3E00ULL, 0x0ULL, kInvalidIndex, false, false}, false},
    InputSpec{"membrane_potential", "Membrane potential inputs", DTypeConstraintSpec{DTypeConstraintMode::Allow, 0x3E00ULL, 0x0ULL, kInvalidIndex, false, false}, false},
    InputSpec{"threshold", "Firing threshold per neuron", DTypeConstraintSpec{DTypeConstraintMode::Deny, 0x3FFEULL, 0x1ULL, kInvalidIndex, false, false}, false},
    InputSpec{"input", "Tensor to quantize", DTypeConstraintSpec{DTypeConstraintMode::Allow, 0x3E00ULL, 0x0ULL, kInvalidIndex, false, false}, false},
};

inline constexpr std::array<Range, kOpCount> kOpOutputRanges = {
    Range{0, 1},
    Range{1, 1},
    Range{2, 1},
    Range{3, 1},
    Range{4, 1},
};

inline constexpr std::array<OutputSpec, kTotalOutputPortCount> kOpOutputSpecs = {
    OutputSpec{"output", "Elementwise sum respecting broadcasting", DTypeRuleKind::Promote, 0x3ULL, kInvalidIndex, kInvalidIndex, ""},
    OutputSpec{"output", "Matrix product with optional bias add", DTypeRuleKind::SameAs, 0x0ULL, 0, kInvalidIndex, ""},
    OutputSpec{"output", "Rectified tensor", DTypeRuleKind::SameAs, 0x0ULL, 0, kInvalidIndex, ""},
    OutputSpec{"spike", "Binary spike events", DTypeRuleKind::Fixed, 0x0ULL, kInvalidIndex, 0, ""},
    OutputSpec{"quantized", "Quantized tensor using user-provided rule", DTypeRuleKind::Custom, 0x0ULL, kInvalidIndex, kInvalidIndex, "InferQuantizedDType"},
};

inline constexpr std::array<Range, kOpCount> kOpAttributeRanges = {
    Range{0, 1},
    Range{1, 2},
    Range{3, 0},
    Range{3, 0},
    Range{3, 2},
};

inline constexpr std::array<AttributeSpec, kTotalAttributeCount> kOpAttributes = {
    AttributeSpec{"alpha", "float", "1.0", true, false, "Optional scaling factor applied to rhs before addition"},
    AttributeSpec{"transposed_lhs", "bool", "false", true, false, "Interpret lhs as already transposed (K, M)"},
    AttributeSpec{"transposed_rhs", "bool", "false", true, false, "Interpret rhs as already transposed (N, K)"},
    AttributeSpec{"bit_width", "int", "8", true, false, "Number of bits per quantized element"},
    AttributeSpec{"symmetric", "bool", "true", true, false, "Use symmetric quantization range"},
};

inline constexpr std::array<ComputePolicySpec, kOpCount> kOpComputePolicies = {
    ComputePolicySpec{ComputePolicyKind::Promote, 0x3ULL, kInvalidIndex, kInvalidIndex, "", ""},
    ComputePolicySpec{ComputePolicyKind::DerivedComputeType, 0x3ULL, kInvalidIndex, kInvalidIndex, "MatMulAccumulatorType", ""},
    ComputePolicySpec{ComputePolicyKind::SameAs, 0x0ULL, 0, kInvalidIndex, "", ""},
    ComputePolicySpec{ComputePolicyKind::Fixed, 0x0ULL, kInvalidIndex, 0, "", ""},
    ComputePolicySpec{ComputePolicyKind::Custom, 0x0ULL, kInvalidIndex, kInvalidIndex, "", "SelectQuantizedComputeType"},
};

inline constexpr std::array<ShapeInferenceSpec, kOpCount> kOpShapeInference = {
    ShapeInferenceSpec{"broadcast", "", "Applies NumPy-style broadcasting rules"},
    ShapeInferenceSpec{"matmul", "", "Uses standard matrix multiplication dimension rules"},
    ShapeInferenceSpec{"identity", "", ""},
    ShapeInferenceSpec{"elementwise", "", ""},
    ShapeInferenceSpec{"custom", "QuantizeShapeRule", "Consults backend-specific quantization layout"},
};

inline constexpr std::array<MetadataSpec, kOpCount> kOpMetadata = {
    MetadataSpec{"Elementwise addition that supports broadcasting", true, true, Range{0, 2}, Range{0, 2}},
    MetadataSpec{"Batched matrix multiplication with optional bias", false, true, Range{2, 2}, Range{2, 0}},
    MetadataSpec{"Rectified Linear Unit activation", false, true, Range{4, 2}, Range{2, 0}},
    MetadataSpec{"Generates binary spike events when membrane potential exceeds threshold", false, false, Range{6, 2}, Range{2, 0}},
    MetadataSpec{"Delegates quantization specifics to a backend-provided rule", false, false, Range{8, 2}, Range{2, 0}},
};

inline constexpr std::array<std::string_view, 10> kOpMetadataTags = {
    "elementwise",
    "commutative",
    "batched",
    "linear_algebra",
    "elementwise",
    "activation",
    "spiking",
    "non-differentiable",
    "quantization",
    "backend-specific",
};

inline constexpr std::array<std::string_view, 2> kOpMetadataAliases = {
    "add",
    "sum",
};

}  // namespace orteaf::generated::ops_tables
