#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

[[noreturn]] void Fail(const std::string &message) {
  throw std::runtime_error(message);
}

std::string EscapeStringLiteral(std::string_view value) {
  std::string escaped;
  escaped.reserve(value.size() + 2);
  for (char ch : value) {
    switch (ch) {
    case '\\':
      escaped += "\\\\";
      break;
    case '"':
      escaped += "\\\"";
      break;
    case '\n':
      escaped += "\\n";
      break;
    case '\t':
      escaped += "\\t";
      break;
    default:
      escaped += ch;
      break;
    }
  }
  return escaped;
}

std::string FormatBitMask(std::uint64_t mask) {
  std::ostringstream oss;
  oss << "0x" << std::uppercase << std::hex << mask << "ULL";
  return oss.str();
}

void ExpectKeys(const YAML::Node &node, std::string_view context,
                std::initializer_list<std::string_view> allowed) {
  std::unordered_set<std::string_view> allowed_set(allowed.begin(),
                                                   allowed.end());
  for (const auto &kv : node) {
    if (!kv.first.IsScalar()) {
      std::ostringstream oss;
      oss << "Non-scalar key encountered in " << context;
      Fail(oss.str());
    }
    const std::string key = kv.first.as<std::string>();
    if (!allowed_set.count(key)) {
      std::ostringstream oss;
      oss << "Unknown key '" << key << "' in " << context;
      Fail(oss.str());
    }
  }
}

std::optional<std::string> ReadOptionalString(const YAML::Node &node,
                                              std::string_view key,
                                              std::string_view context) {
  const auto value = node[key];
  if (!value) {
    return std::nullopt;
  }
  if (!value.IsScalar()) {
    std::ostringstream oss;
    oss << "Key '" << key << "' must be a scalar in " << context;
    Fail(oss.str());
  }
  return value.as<std::string>();
}

std::string ReadRequiredString(const YAML::Node &node, std::string_view key,
                               std::string_view context) {
  auto value = ReadOptionalString(node, key, context);
  if (!value) {
    std::ostringstream oss;
    oss << "Missing required key '" << key << "' in " << context;
    Fail(oss.str());
  }
  if (value->empty()) {
    std::ostringstream oss;
    oss << "Key '" << key << "' must not be empty in " << context;
    Fail(oss.str());
  }
  return *value;
}

int ReadInt(const YAML::Node &node, std::string_view key,
            std::optional<int> default_value, std::string_view context) {
  const auto value = node[key];
  if (!value) {
    if (default_value) {
      return *default_value;
    }
    std::ostringstream oss;
    oss << "Missing required integer key '" << key << "' in " << context;
    Fail(oss.str());
  }
  if (!value.IsScalar()) {
    std::ostringstream oss;
    oss << "Key '" << key << "' must be a scalar integer in " << context;
    Fail(oss.str());
  }
  try {
    return value.as<int>();
  } catch (const YAML::BadConversion &) {
    std::ostringstream oss;
    oss << "Key '" << key << "' must be a valid integer in " << context;
    Fail(oss.str());
  }
}

bool ReadBool(const YAML::Node &node, std::string_view key, bool default_value,
              std::string_view context) {
  const auto value = node[key];
  if (!value) {
    return default_value;
  }
  if (!value.IsScalar()) {
    std::ostringstream oss;
    oss << "Key '" << key << "' must be a boolean in " << context;
    Fail(oss.str());
  }
  try {
    return value.as<bool>();
  } catch (const YAML::BadConversion &) {
    std::ostringstream oss;
    oss << "Key '" << key << "' must be a valid boolean in " << context;
    Fail(oss.str());
  }
}

std::vector<std::string> ReadStringList(const YAML::Node &node,
                                        std::string_view key,
                                        std::string_view context) {
  const auto value = node[key];
  if (!value) {
    return {};
  }
  if (!value.IsSequence()) {
    std::ostringstream oss;
    oss << "Key '" << key << "' must be a sequence in " << context;
    Fail(oss.str());
  }
  std::vector<std::string> result;
  result.reserve(value.size());
  for (std::size_t i = 0; i < value.size(); ++i) {
    const auto &entry = value[i];
    if (!entry.IsScalar()) {
      std::ostringstream oss;
      oss << "All entries of '" << key << "' must be scalars in " << context;
      Fail(oss.str());
    }
    result.emplace_back(entry.as<std::string>());
  }
  return result;
}

std::optional<std::string> ReadOptionalScalarString(const YAML::Node &node,
                                                    std::string_view key,
                                                    std::string_view context) {
  const auto value = node[key];
  if (!value) {
    return std::nullopt;
  }
  if (!value.IsScalar()) {
    std::ostringstream oss;
    oss << "Key '" << key << "' must be a scalar in " << context;
    Fail(oss.str());
  }
  return value.as<std::string>();
}

struct DTypeCatalog {
  std::vector<std::string> ids;
  std::unordered_map<std::string, std::size_t> id_to_index;
  std::unordered_map<std::string, std::uint64_t> category_masks;
  std::unordered_set<std::string> categories;
  std::uint64_t all_mask = 0;
};

DTypeCatalog LoadDTypeCatalog(const fs::path &dtype_yaml_path) {
  YAML::Node root;
  try {
    root = YAML::LoadFile(dtype_yaml_path.string());
  } catch (const std::exception &e) {
    std::ostringstream oss;
    oss << "Failed to load dtype YAML '" << dtype_yaml_path
        << "': " << e.what();
    Fail(oss.str());
  }
  if (!root || !root.IsMap()) {
    std::ostringstream oss;
    oss << "Root of '" << dtype_yaml_path << "' must be a mapping";
    Fail(oss.str());
  }
  const auto dtypes_node = root["dtypes"];
  if (!dtypes_node || !dtypes_node.IsSequence()) {
    std::ostringstream oss;
    oss << "Missing required sequence key 'dtypes' in '" << dtype_yaml_path
        << "'";
    Fail(oss.str());
  }

  DTypeCatalog catalog;
  catalog.ids.reserve(dtypes_node.size());
  for (std::size_t i = 0; i < dtypes_node.size(); ++i) {
    const auto &node = dtypes_node[i];
    if (!node.IsMap()) {
      std::ostringstream oss;
      oss << "Each dtype entry must be a mapping (index " << i << ")";
      Fail(oss.str());
    }
    const std::string context = "dtypes[" + std::to_string(i) + "]";
    const std::string id = ReadRequiredString(node, "id", context);
    if (!catalog.id_to_index.emplace(id, catalog.ids.size()).second) {
      std::ostringstream oss;
      oss << "Duplicate dtype id '" << id << "'";
      Fail(oss.str());
    }
    catalog.ids.emplace_back(id);
    std::string category = "unknown";
    if (auto cat = ReadOptionalString(node, "category", context)) {
      if (!cat->empty()) {
        category = *cat;
      }
    }
    catalog.categories.insert(category);
    const std::size_t bit_index = catalog.ids.size() - 1;
    if (bit_index >= 64) {
      Fail("Currently only up to 64 dtypes are supported by gen_ops (bitset "
           "limitation)");
    }
    catalog.category_masks[category] |= (1ULL << bit_index);
  }
  if (catalog.ids.empty()) {
    Fail("No dtypes defined in dtype catalog");
  }
  catalog.all_mask = (catalog.ids.size() == 64)
                         ? std::numeric_limits<std::uint64_t>::max()
                         : ((1ULL << catalog.ids.size()) - 1ULL);
  return catalog;
}

struct CatalogInfo {
  std::unordered_set<std::string> op_categories;
  std::unordered_set<std::string> dtype_categories;
  std::unordered_set<std::string> shape_kinds;
};

CatalogInfo ParseCatalogs(const YAML::Node &root) {
  CatalogInfo catalogs;
  const auto catalogs_node = root["catalogs"];
  if (!catalogs_node) {
    return catalogs;
  }
  if (!catalogs_node.IsMap()) {
    Fail("Key 'catalogs' must be a mapping");
  }

  const auto categories_node = catalogs_node["categories"];
  if (categories_node) {
    if (!categories_node.IsMap()) {
      Fail("Key 'catalogs.categories' must be a mapping");
    }
    for (const auto &kv : categories_node) {
      if (!kv.first.IsScalar()) {
        Fail("Category keys must be scalar strings");
      }
      catalogs.op_categories.insert(kv.first.as<std::string>());
    }
  }

  const auto dtype_categories_node = catalogs_node["dtype_categories"];
  if (dtype_categories_node) {
    if (!dtype_categories_node.IsMap()) {
      Fail("Key 'catalogs.dtype_categories' must be a mapping");
    }
    for (const auto &kv : dtype_categories_node) {
      if (!kv.first.IsScalar()) {
        Fail("DType category keys must be scalar strings");
      }
      catalogs.dtype_categories.insert(kv.first.as<std::string>());
    }
  }

  const auto shape_kinds_node = catalogs_node["shape_kinds"];
  if (shape_kinds_node) {
    if (!shape_kinds_node.IsMap()) {
      Fail("Key 'catalogs.shape_kinds' must be a mapping");
    }
    for (const auto &kv : shape_kinds_node) {
      if (!kv.first.IsScalar()) {
        Fail("Shape kind keys must be scalar strings");
      }
      catalogs.shape_kinds.insert(kv.first.as<std::string>());
    }
  }
  return catalogs;
}

enum class DTypeConstraintMode : std::uint8_t { Allow, Deny, Match };
enum class DTypeRuleKind : std::uint8_t { Promote, SameAs, Fixed, Custom };
enum class ComputePolicyKind : std::uint8_t {
  SameAs,
  Promote,
  Fixed,
  DerivedComputeType,
  Custom
};

struct DTypeConstraintDef {
  std::string mode;
  std::vector<std::string> dtypes;
  std::vector<std::string> categories;
  std::string reference;
  bool allow_promotion = false;
  bool require_same_shape = false;
};

struct InputDef {
  std::string name;
  std::string description;
  bool optional = false;
  DTypeConstraintDef constraint;
};

struct OutputDef {
  std::string name;
  std::string description;
  std::string kind;
  std::vector<std::string> inputs;
  std::string input;
  std::string dtype;
  std::string function;
};

struct AttributeDef {
  std::string name;
  std::string type;
  std::optional<std::string> default_value;
  bool required = false;
  std::string description;
};

struct ComputePolicyDef {
  std::string kind;
  std::vector<std::string> inputs;
  std::string input;
  std::string dtype;
  std::string function;
  std::string handler;
};

struct ShapeInferenceDef {
  std::string kind;
  std::string function;
  std::string description;
};

struct MetadataDef {
  std::string description;
  std::vector<std::string> tags;
  std::vector<std::string> aliases;
  bool commutative = false;
  bool differentiable = false;
};

struct OpDefinition {
  std::string id;
  std::string display_name;
  std::string category;
  int arity = 0;
  std::vector<InputDef> inputs;
  std::vector<OutputDef> outputs;
  std::vector<AttributeDef> attributes;
  ComputePolicyDef compute_policy;
  ShapeInferenceDef shape_inference;
  MetadataDef metadata;
};

InputDef ParseInput(const YAML::Node &node, std::string_view context) {
  if (!node.IsMap()) {
    std::ostringstream oss;
    oss << "Each input must be a mapping (" << context << ")";
    Fail(oss.str());
  }
  ExpectKeys(node, context,
             {"name", "description", "dtype_constraints", "optional"});
  InputDef input;
  input.name = ReadRequiredString(node, "name", context);
  if (auto description = ReadOptionalString(node, "description", context)) {
    input.description = *description;
  }
  input.optional = ReadBool(node, "optional", false, context);

  const auto constraint_node = node["dtype_constraints"];
  if (!constraint_node || !constraint_node.IsMap()) {
    std::ostringstream oss;
    oss << "Missing required mapping key 'dtype_constraints' in " << context;
    Fail(oss.str());
  }
  const std::string constraint_context =
      std::string(context) + ".dtype_constraints";
  ExpectKeys(constraint_node, constraint_context,
             {"mode", "dtypes", "categories", "reference", "allow_promotion",
              "require_same_shape"});
  input.constraint.mode =
      ReadRequiredString(constraint_node, "mode", constraint_context);
  input.constraint.dtypes =
      ReadStringList(constraint_node, "dtypes", constraint_context);
  input.constraint.categories =
      ReadStringList(constraint_node, "categories", constraint_context);
  if (auto reference = ReadOptionalString(constraint_node, "reference",
                                          constraint_context)) {
    input.constraint.reference = *reference;
  }
  input.constraint.allow_promotion =
      ReadBool(constraint_node, "allow_promotion", false, constraint_context);
  input.constraint.require_same_shape = ReadBool(
      constraint_node, "require_same_shape", false, constraint_context);
  return input;
}

OutputDef ParseOutput(const YAML::Node &node, std::string_view context) {
  if (!node.IsMap()) {
    std::ostringstream oss;
    oss << "Each output must be a mapping (" << context << ")";
    Fail(oss.str());
  }
  ExpectKeys(node, context, {"name", "description", "dtype_rule"});
  OutputDef output;
  output.name = ReadRequiredString(node, "name", context);
  if (auto description = ReadOptionalString(node, "description", context)) {
    output.description = *description;
  }
  const auto rule_node = node["dtype_rule"];
  if (!rule_node || !rule_node.IsMap()) {
    std::ostringstream oss;
    oss << "Missing required mapping key 'dtype_rule' in " << context;
    Fail(oss.str());
  }
  const std::string rule_context = std::string(context) + ".dtype_rule";
  ExpectKeys(rule_node, rule_context,
             {"kind", "inputs", "input", "dtype", "function"});
  output.kind = ReadRequiredString(rule_node, "kind", rule_context);
  output.inputs = ReadStringList(rule_node, "inputs", rule_context);
  if (auto input = ReadOptionalString(rule_node, "input", rule_context)) {
    output.input = *input;
  }
  if (auto dtype = ReadOptionalString(rule_node, "dtype", rule_context)) {
    output.dtype = *dtype;
  }
  if (auto function = ReadOptionalString(rule_node, "function", rule_context)) {
    output.function = *function;
  }
  return output;
}

AttributeDef ParseAttribute(const YAML::Node &node, std::string_view context) {
  if (!node.IsMap()) {
    std::ostringstream oss;
    oss << "Each attribute must be a mapping (" << context << ")";
    Fail(oss.str());
  }
  ExpectKeys(node, context,
             {"name", "type", "default", "required", "description"});
  AttributeDef attribute;
  attribute.name = ReadRequiredString(node, "name", context);
  attribute.type = ReadRequiredString(node, "type", context);
  if (auto default_value = ReadOptionalScalarString(node, "default", context)) {
    attribute.default_value = *default_value;
  }
  attribute.required = ReadBool(node, "required", false, context);
  if (auto description = ReadOptionalString(node, "description", context)) {
    attribute.description = *description;
  }
  return attribute;
}

ComputePolicyDef ParseComputePolicy(const YAML::Node &node,
                                    std::string_view context) {
  if (!node || !node.IsMap()) {
    std::ostringstream oss;
    oss << "Missing required mapping key 'compute_policy' in " << context;
    Fail(oss.str());
  }
  ExpectKeys(node, context,
             {"kind", "inputs", "input", "dtype", "function", "handler"});
  ComputePolicyDef policy;
  policy.kind = ReadRequiredString(node, "kind", context);
  policy.inputs = ReadStringList(node, "inputs", context);
  if (auto input = ReadOptionalString(node, "input", context)) {
    policy.input = *input;
  }
  if (auto dtype = ReadOptionalString(node, "dtype", context)) {
    policy.dtype = *dtype;
  }
  if (auto function = ReadOptionalString(node, "function", context)) {
    policy.function = *function;
  }
  if (auto handler = ReadOptionalString(node, "handler", context)) {
    policy.handler = *handler;
  }
  return policy;
}

ShapeInferenceDef ParseShapeInference(const YAML::Node &node,
                                      std::string_view context) {
  if (!node || !node.IsMap()) {
    std::ostringstream oss;
    oss << "Missing required mapping key 'shape_inference' in " << context;
    Fail(oss.str());
  }
  ExpectKeys(node, context, {"kind", "function", "description"});
  ShapeInferenceDef shape;
  shape.kind = ReadRequiredString(node, "kind", context);
  if (auto function = ReadOptionalString(node, "function", context)) {
    shape.function = *function;
  }
  if (auto description = ReadOptionalString(node, "description", context)) {
    shape.description = *description;
  }
  return shape;
}

MetadataDef ParseMetadata(const YAML::Node &node, std::string_view context) {
  MetadataDef metadata;
  if (!node) {
    return metadata;
  }
  if (!node.IsMap()) {
    std::ostringstream oss;
    oss << "Key 'metadata' must be a mapping in " << context;
    Fail(oss.str());
  }
  ExpectKeys(
      node, context,
      {"description", "tags", "aliases", "commutative", "differentiable"});
  if (auto description = ReadOptionalString(node, "description", context)) {
    metadata.description = *description;
  }
  metadata.tags = ReadStringList(node, "tags", context);
  metadata.aliases = ReadStringList(node, "aliases", context);
  metadata.commutative = ReadBool(node, "commutative", false, context);
  metadata.differentiable = ReadBool(node, "differentiable", false, context);
  return metadata;
}

struct ParsedOpsFile {
  std::string schema_version;
  CatalogInfo catalogs;
  std::vector<OpDefinition> ops;
};

// Collect all .yml files from a directory (sorted), or return a single file.
std::vector<fs::path> CollectYamlFiles(const fs::path &input_path) {
  if (fs::is_regular_file(input_path)) {
    return {input_path};
  }
  if (!fs::is_directory(input_path)) {
    std::ostringstream oss;
    oss << "Path '" << input_path << "' is neither a file nor a directory";
    Fail(oss.str());
  }
  std::vector<fs::path> files;
  for (const auto &entry : fs::directory_iterator(input_path)) {
    if (entry.is_regular_file() && entry.path().extension() == ".yml") {
      files.push_back(entry.path());
    }
  }
  std::sort(files.begin(), files.end());
  if (files.empty()) {
    std::ostringstream oss;
    oss << "No .yml files found in directory '" << input_path << "'";
    Fail(oss.str());
  }
  return files;
}

ParsedOpsFile ParseOpsFile(const fs::path &ops_input_path) {
  const auto yaml_files = CollectYamlFiles(ops_input_path);

  ParsedOpsFile parsed;
  bool schema_version_set = false;

  for (const auto &yaml_path : yaml_files) {
    YAML::Node root;
    try {
      root = YAML::LoadFile(yaml_path.string());
    } catch (const std::exception &e) {
      std::ostringstream oss;
      oss << "Failed to load ops YAML '" << yaml_path << "': " << e.what();
      Fail(oss.str());
    }
    if (!root || !root.IsMap()) {
      std::ostringstream oss;
      oss << "Root of '" << yaml_path << "' must be a mapping";
      Fail(oss.str());
    }

    // Read schema_version from the first file that has it
    if (!schema_version_set) {
      if (auto sv =
              ReadOptionalString(root, "schema_version", yaml_path.string())) {
        parsed.schema_version = *sv;
        schema_version_set = true;
      }
    }

    // Merge catalogs from any file that has them
    CatalogInfo file_catalogs = ParseCatalogs(root);
    parsed.catalogs.op_categories.insert(file_catalogs.op_categories.begin(),
                                         file_catalogs.op_categories.end());
    parsed.catalogs.dtype_categories.insert(
        file_catalogs.dtype_categories.begin(),
        file_catalogs.dtype_categories.end());
    parsed.catalogs.shape_kinds.insert(file_catalogs.shape_kinds.begin(),
                                       file_catalogs.shape_kinds.end());

    // Merge ops from any file that has them
    const auto ops_node = root["ops"];
    if (!ops_node) {
      continue; // Skip files without 'ops' key (e.g. catalogs.yml)
    }
    if (!ops_node.IsSequence()) {
      std::ostringstream oss;
      oss << "Key 'ops' must be a sequence in '" << yaml_path << "'";
      Fail(oss.str());
    }
    for (std::size_t i = 0; i < ops_node.size(); ++i) {
      const auto &node = ops_node[i];
      const std::string context =
          yaml_path.filename().string() + ":ops[" + std::to_string(i) + "]";
      if (!node.IsMap()) {
        std::ostringstream oss;
        oss << "Each op entry must be a mapping (" << context << ")";
        Fail(oss.str());
      }
      ExpectKeys(node, context,
                 {"id", "display_name", "category", "arity", "inputs",
                  "outputs", "attributes", "compute_policy", "shape_inference",
                  "metadata"});
      OpDefinition op;
      op.id = ReadRequiredString(node, "id", context);
      if (auto display_name =
              ReadOptionalString(node, "display_name", context)) {
        op.display_name = *display_name;
      }
      op.category = ReadRequiredString(node, "category", context);
      op.arity = ReadInt(node, "arity", std::nullopt, context);

      const auto inputs_node = node["inputs"];
      if (!inputs_node || !inputs_node.IsSequence()) {
        std::ostringstream oss;
        oss << "Missing required sequence key 'inputs' in " << context;
        Fail(oss.str());
      }
      op.inputs.reserve(inputs_node.size());
      for (std::size_t j = 0; j < inputs_node.size(); ++j) {
        op.inputs.emplace_back(ParseInput(
            inputs_node[j], context + ".inputs[" + std::to_string(j) + "]"));
      }

      const auto outputs_node = node["outputs"];
      if (!outputs_node || !outputs_node.IsSequence()) {
        std::ostringstream oss;
        oss << "Missing required sequence key 'outputs' in " << context;
        Fail(oss.str());
      }
      op.outputs.reserve(outputs_node.size());
      for (std::size_t j = 0; j < outputs_node.size(); ++j) {
        op.outputs.emplace_back(ParseOutput(
            outputs_node[j], context + ".outputs[" + std::to_string(j) + "]"));
      }

      const auto attributes_node = node["attributes"];
      if (attributes_node) {
        if (!attributes_node.IsSequence()) {
          std::ostringstream oss;
          oss << "Key 'attributes' must be a sequence in " << context;
          Fail(oss.str());
        }
        op.attributes.reserve(attributes_node.size());
        for (std::size_t j = 0; j < attributes_node.size(); ++j) {
          op.attributes.emplace_back(
              ParseAttribute(attributes_node[j], context + ".attributes[" +
                                                     std::to_string(j) + "]"));
        }
      }

      op.compute_policy = ParseComputePolicy(node["compute_policy"],
                                             context + ".compute_policy");
      op.shape_inference = ParseShapeInference(node["shape_inference"],
                                               context + ".shape_inference");
      op.metadata = ParseMetadata(node["metadata"], context + ".metadata");

      parsed.ops.emplace_back(std::move(op));
    }
  }

  if (!schema_version_set) {
    Fail("No schema_version found in any YAML file");
  }

  return parsed;
}

DTypeConstraintMode ParseConstraintMode(const std::string &mode,
                                        std::string_view context) {
  if (mode == "allow") {
    return DTypeConstraintMode::Allow;
  }
  if (mode == "deny") {
    return DTypeConstraintMode::Deny;
  }
  if (mode == "match") {
    return DTypeConstraintMode::Match;
  }
  std::ostringstream oss;
  oss << "Unsupported dtype constraint mode '" << mode << "' in " << context;
  Fail(oss.str());
}

DTypeRuleKind ParseDTypeRuleKind(const std::string &kind,
                                 std::string_view context) {
  if (kind == "promote") {
    return DTypeRuleKind::Promote;
  }
  if (kind == "same_as") {
    return DTypeRuleKind::SameAs;
  }
  if (kind == "fixed") {
    return DTypeRuleKind::Fixed;
  }
  if (kind == "custom") {
    return DTypeRuleKind::Custom;
  }
  std::ostringstream oss;
  oss << "Unsupported dtype_rule kind '" << kind << "' in " << context;
  Fail(oss.str());
}

ComputePolicyKind ParseComputePolicyKind(const std::string &kind,
                                         std::string_view context) {
  if (kind == "same_as") {
    return ComputePolicyKind::SameAs;
  }
  if (kind == "promote") {
    return ComputePolicyKind::Promote;
  }
  if (kind == "fixed") {
    return ComputePolicyKind::Fixed;
  }
  if (kind == "derived_compute_type") {
    return ComputePolicyKind::DerivedComputeType;
  }
  if (kind == "custom") {
    return ComputePolicyKind::Custom;
  }
  std::ostringstream oss;
  oss << "Unsupported compute_policy kind '" << kind << "' in " << context;
  Fail(oss.str());
}

std::uint32_t ResolveDTypeIndex(const std::string &dtype_id,
                                const DTypeCatalog &catalog,
                                std::string_view context) {
  auto it = catalog.id_to_index.find(dtype_id);
  if (it == catalog.id_to_index.end()) {
    std::ostringstream oss;
    oss << "Unknown dtype '" << dtype_id << "' referenced in " << context;
    Fail(oss.str());
  }
  return static_cast<std::uint32_t>(it->second);
}

std::uint64_t MaskForDTypes(const std::vector<std::string> &dtypes,
                            const DTypeCatalog &catalog,
                            std::string_view context) {
  std::uint64_t mask = 0;
  for (const auto &dtype : dtypes) {
    mask |= (1ULL << ResolveDTypeIndex(dtype, catalog, context));
  }
  return mask;
}

std::uint64_t MaskForCategories(const std::vector<std::string> &categories,
                                const DTypeCatalog &catalog,
                                std::string_view context) {
  std::uint64_t mask = 0;
  for (const auto &category : categories) {
    auto it = catalog.category_masks.find(category);
    if (it == catalog.category_masks.end()) {
      std::ostringstream oss;
      oss << "Unknown dtype category '" << category << "' referenced in "
          << context;
      Fail(oss.str());
    }
    mask |= it->second;
  }
  return mask;
}

bool IsInteger(const std::string &value) {
  if (value.empty()) {
    return false;
  }
  std::size_t idx = 0;
  if (value[0] == '-' || value[0] == '+') {
    idx = 1;
  }
  if (idx >= value.size()) {
    return false;
  }
  for (; idx < value.size(); ++idx) {
    if (!std::isdigit(static_cast<unsigned char>(value[idx]))) {
      return false;
    }
  }
  return true;
}

bool IsFloatingPoint(const std::string &value) {
  if (value.empty()) {
    return false;
  }
  bool seen_digit = false;
  bool seen_dot = false;
  std::size_t idx = 0;
  if (value[0] == '-' || value[0] == '+') {
    idx = 1;
  }
  for (; idx < value.size(); ++idx) {
    char ch = value[idx];
    if (std::isdigit(static_cast<unsigned char>(ch))) {
      seen_digit = true;
    } else if (ch == '.' && !seen_dot) {
      seen_dot = true;
    } else {
      return false;
    }
  }
  return seen_digit;
}

struct ResolvedInput {
  std::string name;
  std::string description;
  bool optional = false;
  DTypeConstraintMode mode = DTypeConstraintMode::Allow;
  std::uint64_t allow_mask = 0;
  std::uint64_t deny_mask = 0;
  std::uint32_t reference_input = std::numeric_limits<std::uint32_t>::max();
  bool allow_promotion = false;
  bool require_same_shape = false;
};

struct ResolvedOutput {
  std::string name;
  std::string description;
  DTypeRuleKind kind = DTypeRuleKind::Promote;
  std::uint64_t input_mask = 0;
  std::uint32_t reference_input = std::numeric_limits<std::uint32_t>::max();
  std::uint32_t fixed_dtype = std::numeric_limits<std::uint32_t>::max();
  std::string custom_function;
};

struct ResolvedAttribute {
  std::string name;
  std::string type;
  std::optional<std::string> default_value;
  bool required = false;
  std::string description;
};

struct ResolvedComputePolicy {
  ComputePolicyKind kind = ComputePolicyKind::SameAs;
  std::uint64_t input_mask = 0;
  std::uint32_t reference_input = std::numeric_limits<std::uint32_t>::max();
  std::uint32_t fixed_dtype = std::numeric_limits<std::uint32_t>::max();
  std::string function;
  std::string handler;
};

struct ResolvedShapeInference {
  std::string kind;
  std::string function;
  std::string description;
};

struct ResolvedMetadata {
  std::string description;
  bool commutative = false;
  bool differentiable = false;
  std::vector<std::string> tags;
  std::vector<std::string> aliases;
};

struct ResolvedOp {
  std::string id;
  std::string display_name;
  std::string category;
  std::uint32_t arity = 0;
  std::vector<ResolvedInput> inputs;
  std::vector<ResolvedOutput> outputs;
  std::vector<ResolvedAttribute> attributes;
  ResolvedComputePolicy compute_policy;
  ResolvedShapeInference shape_inference;
  ResolvedMetadata metadata;
};

void ValidateAttributeDefault(const ResolvedAttribute &attribute,
                              std::string_view context) {
  if (!attribute.default_value) {
    return;
  }
  const std::string &value = *attribute.default_value;
  const std::string &type = attribute.type;
  auto fail_type = [&](const std::string &message) {
    std::ostringstream oss;
    oss << "Attribute '" << attribute.name << "' default mismatch in "
        << context << ": " << message;
    Fail(oss.str());
  };
  if (type == "int") {
    if (!IsInteger(value)) {
      fail_type("expected integer default but got '" + value + "'");
    }
  } else if (type == "float") {
    if (!IsInteger(value) && !IsFloatingPoint(value)) {
      fail_type("expected numeric default but got '" + value + "'");
    }
  } else if (type == "bool") {
    if (!(value == "true" || value == "false" || value == "True" ||
          value == "False")) {
      fail_type("expected boolean default but got '" + value + "'");
    }
  } else if (type == "string") {
    return;
  } else {
    // Unknown types are not validated to allow future extensions.
  }
}

std::vector<ResolvedOp> ResolveOps(const ParsedOpsFile &parsed,
                                   const DTypeCatalog &dtype_catalog) {
  if (parsed.schema_version != "1.0") {
    std::ostringstream oss;
    oss << "Unsupported schema_version '" << parsed.schema_version
        << "', expected '1.0'";
    Fail(oss.str());
  }

  std::vector<ResolvedOp> resolved;
  resolved.reserve(parsed.ops.size());
  std::unordered_set<std::string> seen_ids;

  for (std::size_t op_index = 0; op_index < parsed.ops.size(); ++op_index) {
    const auto &op = parsed.ops[op_index];
    const std::string context = "ops[" + std::to_string(op_index) + "]";

    if (!seen_ids.insert(op.id).second) {
      std::ostringstream oss;
      oss << "Duplicate op id '" << op.id << "'";
      Fail(oss.str());
    }

    if (!parsed.catalogs.op_categories.empty() &&
        !parsed.catalogs.op_categories.count(op.category)) {
      std::ostringstream oss;
      oss << "Op '" << op.id << "' references unknown category '" << op.category
          << "'";
      Fail(oss.str());
    }

    if (op.inputs.size() > 64) {
      std::ostringstream oss;
      oss << "Op '" << op.id << "' declares " << op.inputs.size()
          << " inputs, which exceeds the 64-input mask limitation";
      Fail(oss.str());
    }

    ResolvedOp resolved_op;
    resolved_op.id = op.id;
    resolved_op.display_name =
        op.display_name.empty() ? op.id : op.display_name;
    resolved_op.category = op.category;

    std::unordered_map<std::string, std::size_t> input_indices;
    input_indices.reserve(op.inputs.size());
    std::uint32_t required_count = 0;
    for (std::size_t i = 0; i < op.inputs.size(); ++i) {
      const auto &input = op.inputs[i];
      const std::string input_context =
          context + ".inputs[" + std::to_string(i) + "]";
      if (!input_indices.emplace(input.name, i).second) {
        std::ostringstream oss;
        oss << "Duplicate input name '" << input.name << "' in " << context;
        Fail(oss.str());
      }
      if (!input.optional) {
        ++required_count;
      }
    }

    if (op.arity < 0) {
      std::ostringstream oss;
      oss << "Op '" << op.id << "' has negative arity";
      Fail(oss.str());
    }
    if (static_cast<std::uint32_t>(op.arity) != required_count) {
      std::ostringstream oss;
      oss << "Op '" << op.id << "' arity mismatch: declared " << op.arity
          << " but found " << required_count << " required inputs";
      Fail(oss.str());
    }
    resolved_op.arity = static_cast<std::uint32_t>(op.arity);

    resolved_op.inputs.reserve(op.inputs.size());
    for (std::size_t i = 0; i < op.inputs.size(); ++i) {
      const auto &input = op.inputs[i];
      const std::string input_context =
          context + ".inputs[" + std::to_string(i) + "]";
      ResolvedInput resolved_input;
      resolved_input.name = input.name;
      resolved_input.description = input.description;
      resolved_input.optional = input.optional;

      resolved_input.mode = ParseConstraintMode(
          input.constraint.mode, input_context + ".dtype_constraints.mode");
      std::uint64_t dtype_mask =
          MaskForDTypes(input.constraint.dtypes, dtype_catalog,
                        input_context + ".dtype_constraints.dtypes");
      std::uint64_t category_mask =
          MaskForCategories(input.constraint.categories, dtype_catalog,
                            input_context + ".dtype_constraints.categories");

      switch (resolved_input.mode) {
      case DTypeConstraintMode::Allow: {
        resolved_input.allow_mask = dtype_mask | category_mask;
        if (resolved_input.allow_mask == 0) {
          resolved_input.allow_mask = dtype_catalog.all_mask;
        }
        resolved_input.deny_mask = 0;
        resolved_input.reference_input =
            std::numeric_limits<std::uint32_t>::max();
        resolved_input.allow_promotion = false;
        resolved_input.require_same_shape = false;
        break;
      }
      case DTypeConstraintMode::Deny: {
        resolved_input.deny_mask = dtype_mask | category_mask;
        resolved_input.allow_mask =
            dtype_catalog.all_mask & ~resolved_input.deny_mask;
        resolved_input.reference_input =
            std::numeric_limits<std::uint32_t>::max();
        resolved_input.allow_promotion = false;
        resolved_input.require_same_shape = false;
        break;
      }
      case DTypeConstraintMode::Match: {
        if (input.constraint.reference.empty()) {
          std::ostringstream oss;
          oss << "Constraint mode 'match' requires 'reference' in "
              << input_context;
          Fail(oss.str());
        }
        auto it = input_indices.find(input.constraint.reference);
        if (it == input_indices.end()) {
          std::ostringstream oss;
          oss << "Constraint mode 'match' references unknown input '"
              << input.constraint.reference << "' in " << input_context;
          Fail(oss.str());
        }
        resolved_input.reference_input = static_cast<std::uint32_t>(it->second);
        resolved_input.allow_mask = dtype_catalog.all_mask;
        resolved_input.deny_mask = 0;
        resolved_input.allow_promotion = input.constraint.allow_promotion;
        resolved_input.require_same_shape = input.constraint.require_same_shape;
        break;
      }
      }

      resolved_op.inputs.emplace_back(std::move(resolved_input));
    }

    resolved_op.outputs.reserve(op.outputs.size());
    std::unordered_set<std::string> output_names;
    for (std::size_t i = 0; i < op.outputs.size(); ++i) {
      const auto &output = op.outputs[i];
      const std::string output_context =
          context + ".outputs[" + std::to_string(i) + "]";
      if (!output_names.insert(output.name).second) {
        std::ostringstream oss;
        oss << "Duplicate output name '" << output.name << "' in " << context;
        Fail(oss.str());
      }
      ResolvedOutput resolved_output;
      resolved_output.name = output.name;
      resolved_output.description = output.description;
      resolved_output.kind =
          ParseDTypeRuleKind(output.kind, output_context + ".dtype_rule.kind");

      auto resolve_input_list = [&](const std::vector<std::string> &names,
                                    std::string_view field_context) {
        if (names.empty()) {
          std::ostringstream oss;
          oss << "Field '" << field_context << "' must list at least one input";
          Fail(oss.str());
        }
        std::uint64_t mask = 0;
        for (const auto &name : names) {
          auto it = input_indices.find(name);
          if (it == input_indices.end()) {
            std::ostringstream oss;
            oss << "Unknown input '" << name << "' referenced in "
                << field_context;
            Fail(oss.str());
          }
          mask |= (1ULL << it->second);
        }
        return mask;
      };

      switch (resolved_output.kind) {
      case DTypeRuleKind::Promote:
        resolved_output.input_mask = resolve_input_list(
            output.inputs, output_context + ".dtype_rule.inputs");
        resolved_output.reference_input =
            std::numeric_limits<std::uint32_t>::max();
        resolved_output.fixed_dtype = std::numeric_limits<std::uint32_t>::max();
        break;
      case DTypeRuleKind::SameAs: {
        if (output.input.empty()) {
          std::ostringstream oss;
          oss << "dtype_rule 'same_as' requires 'input' in " << output_context;
          Fail(oss.str());
        }
        auto it = input_indices.find(output.input);
        if (it == input_indices.end()) {
          std::ostringstream oss;
          oss << "dtype_rule 'same_as' references unknown input '"
              << output.input << "' in " << output_context;
          Fail(oss.str());
        }
        resolved_output.reference_input =
            static_cast<std::uint32_t>(it->second);
        resolved_output.input_mask = 0;
        resolved_output.fixed_dtype = std::numeric_limits<std::uint32_t>::max();
        break;
      }
      case DTypeRuleKind::Fixed:
        if (output.dtype.empty()) {
          std::ostringstream oss;
          oss << "dtype_rule 'fixed' requires 'dtype' in " << output_context;
          Fail(oss.str());
        }
        resolved_output.fixed_dtype = ResolveDTypeIndex(
            output.dtype, dtype_catalog, output_context + ".dtype_rule.dtype");
        resolved_output.input_mask = 0;
        resolved_output.reference_input =
            std::numeric_limits<std::uint32_t>::max();
        break;
      case DTypeRuleKind::Custom:
        if (output.function.empty()) {
          std::ostringstream oss;
          oss << "dtype_rule 'custom' requires 'function' in "
              << output_context;
          Fail(oss.str());
        }
        resolved_output.custom_function = output.function;
        resolved_output.input_mask = 0;
        resolved_output.reference_input =
            std::numeric_limits<std::uint32_t>::max();
        resolved_output.fixed_dtype = std::numeric_limits<std::uint32_t>::max();
        break;
      }
      resolved_op.outputs.emplace_back(std::move(resolved_output));
    }

    resolved_op.attributes.reserve(op.attributes.size());
    std::unordered_set<std::string> attribute_names;
    for (std::size_t i = 0; i < op.attributes.size(); ++i) {
      const auto &attribute = op.attributes[i];
      const std::string attribute_context =
          context + ".attributes[" + std::to_string(i) + "]";
      if (!attribute_names.insert(attribute.name).second) {
        std::ostringstream oss;
        oss << "Duplicate attribute name '" << attribute.name << "' in "
            << context;
        Fail(oss.str());
      }
      ResolvedAttribute resolved_attribute;
      resolved_attribute.name = attribute.name;
      resolved_attribute.type = attribute.type;
      resolved_attribute.default_value = attribute.default_value;
      resolved_attribute.required = attribute.required;
      resolved_attribute.description = attribute.description;
      ValidateAttributeDefault(resolved_attribute, attribute_context);
      resolved_op.attributes.emplace_back(std::move(resolved_attribute));
    }

    resolved_op.compute_policy.kind = ParseComputePolicyKind(
        op.compute_policy.kind, context + ".compute_policy.kind");
    auto resolve_policy_inputs =
        [&](const std::vector<std::string> &names,
            std::string_view field_context) -> std::uint64_t {
      if (names.empty()) {
        return 0;
      }
      std::uint64_t mask = 0;
      for (const auto &name : names) {
        auto it = input_indices.find(name);
        if (it == input_indices.end()) {
          std::ostringstream oss;
          oss << "Unknown input '" << name << "' referenced in "
              << field_context;
          Fail(oss.str());
        }
        mask |= (1ULL << it->second);
      }
      return mask;
    };

    switch (resolved_op.compute_policy.kind) {
    case ComputePolicyKind::SameAs: {
      if (op.compute_policy.input.empty()) {
        std::ostringstream oss;
        oss << "compute_policy 'same_as' requires 'input' in " << context
            << ".compute_policy";
        Fail(oss.str());
      }
      auto it = input_indices.find(op.compute_policy.input);
      if (it == input_indices.end()) {
        std::ostringstream oss;
        oss << "compute_policy 'same_as' references unknown input '"
            << op.compute_policy.input << "'";
        Fail(oss.str());
      }
      resolved_op.compute_policy.reference_input =
          static_cast<std::uint32_t>(it->second);
      break;
    }
    case ComputePolicyKind::Promote:
      resolved_op.compute_policy.input_mask = resolve_policy_inputs(
          op.compute_policy.inputs, context + ".compute_policy.inputs");
      if (resolved_op.compute_policy.input_mask == 0) {
        std::ostringstream oss;
        oss << "compute_policy 'promote' must specify non-empty 'inputs' in "
            << context;
        Fail(oss.str());
      }
      break;
    case ComputePolicyKind::Fixed:
      if (op.compute_policy.dtype.empty()) {
        std::ostringstream oss;
        oss << "compute_policy 'fixed' requires 'dtype' in " << context;
        Fail(oss.str());
      }
      resolved_op.compute_policy.fixed_dtype =
          ResolveDTypeIndex(op.compute_policy.dtype, dtype_catalog,
                            std::string(context) + ".compute_policy.dtype");
      break;
    case ComputePolicyKind::DerivedComputeType:
      if (op.compute_policy.function.empty()) {
        std::ostringstream oss;
        oss << "compute_policy 'derived_compute_type' requires 'function' in "
            << context;
        Fail(oss.str());
      }
      resolved_op.compute_policy.function = op.compute_policy.function;
      resolved_op.compute_policy.input_mask = resolve_policy_inputs(
          op.compute_policy.inputs,
          std::string(context) + ".compute_policy.inputs");
      break;
    case ComputePolicyKind::Custom:
      if (op.compute_policy.handler.empty()) {
        std::ostringstream oss;
        oss << "compute_policy 'custom' requires 'handler' in " << context;
        Fail(oss.str());
      }
      resolved_op.compute_policy.handler = op.compute_policy.handler;
      resolved_op.compute_policy.function = op.compute_policy.function;
      resolved_op.compute_policy.input_mask = resolve_policy_inputs(
          op.compute_policy.inputs,
          std::string(context) + ".compute_policy.inputs");
      break;
    }

    resolved_op.shape_inference.kind = op.shape_inference.kind;
    resolved_op.shape_inference.function = op.shape_inference.function;
    resolved_op.shape_inference.description = op.shape_inference.description;
    if (!parsed.catalogs.shape_kinds.empty() &&
        !parsed.catalogs.shape_kinds.count(op.shape_inference.kind)) {
      std::ostringstream oss;
      oss << "Op '" << op.id << "' references unknown shape_inference.kind '"
          << op.shape_inference.kind << "'";
      Fail(oss.str());
    }

    resolved_op.metadata.description = op.metadata.description;
    resolved_op.metadata.commutative = op.metadata.commutative;
    resolved_op.metadata.differentiable = op.metadata.differentiable;
    resolved_op.metadata.tags = op.metadata.tags;
    resolved_op.metadata.aliases = op.metadata.aliases;

    resolved.emplace_back(std::move(resolved_op));
  }

  return resolved;
}

struct GeneratedData {
  std::string ops_def;
  std::string ops_tables_header;
};

GeneratedData GenerateOutputs(const std::vector<ResolvedOp> &ops,
                              const DTypeCatalog &dtype_catalog) {
  (void)dtype_catalog;
  GeneratedData generated;
  std::ostringstream def_stream;
  def_stream << "// Auto-generated. Do not edit.\n";
  for (const auto &op : ops) {
    def_stream << "ORTEAF_OP(" << op.id << ", \""
               << EscapeStringLiteral(op.display_name) << "\", \""
               << EscapeStringLiteral(op.category) << "\")\n";
  }
  generated.ops_def = def_stream.str();

  const std::size_t op_count = ops.size();
  std::size_t total_inputs = 0;
  std::size_t total_outputs = 0;
  std::size_t total_attributes = 0;
  std::size_t total_tags = 0;
  std::size_t total_aliases = 0;

  for (const auto &op : ops) {
    total_inputs += op.inputs.size();
    total_outputs += op.outputs.size();
    total_attributes += op.attributes.size();
    total_tags += op.metadata.tags.size();
    total_aliases += op.metadata.aliases.size();
  }

  std::ostringstream header_stream;
  header_stream << "// Auto-generated. Do not edit.\n";
  header_stream << "#pragma once\n\n";
  header_stream << "#include <array>\n";
  header_stream << "#include <cstddef>\n";
  header_stream << "#include <cstdint>\n";
  header_stream << "#include <string_view>\n\n";
  header_stream << "namespace orteaf::generated::ops_tables {\n\n";
  header_stream << "using ::orteaf::internal::DType;\n\n";
  header_stream << "inline constexpr std::size_t kOpCount = " << op_count
                << ";\n";
  header_stream << "inline constexpr std::size_t kTotalInputPortCount = "
                << total_inputs << ";\n";
  header_stream << "inline constexpr std::size_t kTotalOutputPortCount = "
                << total_outputs << ";\n";
  header_stream << "inline constexpr std::size_t kTotalAttributeCount = "
                << total_attributes << ";\n";
  header_stream
      << "inline constexpr std::uint32_t kInvalidIndex = 0xFFFFFFFFu;\n\n";

  header_stream << "enum class DTypeConstraintMode : std::uint8_t { Allow, "
                   "Deny, Match };\n";
  header_stream << "enum class DTypeRuleKind : std::uint8_t { Promote, SameAs, "
                   "Fixed, Custom };\n";
  header_stream << "enum class ComputePolicyKind : std::uint8_t { SameAs, "
                   "Promote, Fixed, DerivedComputeType, Custom };\n\n";

  header_stream << "struct Range {\n";
  header_stream << "    std::size_t offset;\n";
  header_stream << "    std::size_t count;\n";
  header_stream << "};\n\n";

  header_stream << "struct DTypeConstraintSpec {\n";
  header_stream << "    DTypeConstraintMode mode;\n";
  header_stream << "    std::uint64_t allow_mask;\n";
  header_stream << "    std::uint64_t deny_mask;\n";
  header_stream << "    std::uint32_t reference_input;\n";
  header_stream << "    bool allow_promotion;\n";
  header_stream << "    bool require_same_shape;\n";
  header_stream << "};\n\n";

  header_stream << "struct InputSpec {\n";
  header_stream << "    std::string_view name;\n";
  header_stream << "    std::string_view description;\n";
  header_stream << "    DTypeConstraintSpec dtype;\n";
  header_stream << "    bool optional;\n";
  header_stream << "};\n\n";

  header_stream << "struct OutputSpec {\n";
  header_stream << "    std::string_view name;\n";
  header_stream << "    std::string_view description;\n";
  header_stream << "    DTypeRuleKind kind;\n";
  header_stream << "    std::uint64_t input_mask;\n";
  header_stream << "    std::uint32_t reference_input;\n";
  header_stream << "    std::uint32_t fixed_dtype;\n";
  header_stream << "    std::string_view custom_function;\n";
  header_stream << "};\n\n";

  header_stream << "struct AttributeSpec {\n";
  header_stream << "    std::string_view name;\n";
  header_stream << "    std::string_view type;\n";
  header_stream << "    std::string_view default_value;\n";
  header_stream << "    bool has_default;\n";
  header_stream << "    bool required;\n";
  header_stream << "    std::string_view description;\n";
  header_stream << "};\n\n";

  header_stream << "struct ComputePolicySpec {\n";
  header_stream << "    ComputePolicyKind kind;\n";
  header_stream << "    std::uint64_t input_mask;\n";
  header_stream << "    std::uint32_t reference_input;\n";
  header_stream << "    std::uint32_t fixed_dtype;\n";
  header_stream << "    std::string_view function;\n";
  header_stream << "    std::string_view handler;\n";
  header_stream << "};\n\n";

  header_stream << "struct ShapeInferenceSpec {\n";
  header_stream << "    std::string_view kind;\n";
  header_stream << "    std::string_view function;\n";
  header_stream << "    std::string_view description;\n";
  header_stream << "};\n\n";

  header_stream << "struct MetadataSpec {\n";
  header_stream << "    std::string_view description;\n";
  header_stream << "    bool commutative;\n";
  header_stream << "    bool differentiable;\n";
  header_stream << "    Range tags;\n";
  header_stream << "    Range aliases;\n";
  header_stream << "};\n\n";

  header_stream
      << "inline constexpr std::array<std::string_view, kOpCount> kOpIds = {\n";
  for (const auto &op : ops) {
    header_stream << "    \"" << EscapeStringLiteral(op.id) << "\",\n";
  }
  header_stream << "};\n\n";

  header_stream << "inline constexpr std::array<std::string_view, kOpCount> "
                   "kOpDisplayNames = {\n";
  for (const auto &op : ops) {
    header_stream << "    \"" << EscapeStringLiteral(op.display_name)
                  << "\",\n";
  }
  header_stream << "};\n\n";

  header_stream << "inline constexpr std::array<std::string_view, kOpCount> "
                   "kOpCategories = {\n";
  for (const auto &op : ops) {
    header_stream << "    \"" << EscapeStringLiteral(op.category) << "\",\n";
  }
  header_stream << "};\n\n";

  header_stream
      << "inline constexpr std::array<std::uint32_t, kOpCount> kOpArity = {\n";
  for (const auto &op : ops) {
    header_stream << "    " << op.arity << ",\n";
  }
  header_stream << "};\n\n";

  header_stream
      << "inline constexpr std::array<Range, kOpCount> kOpInputRanges = {\n";
  std::size_t input_offset = 0;
  for (const auto &op : ops) {
    header_stream << "    Range{" << input_offset << ", " << op.inputs.size()
                  << "},\n";
    input_offset += op.inputs.size();
  }
  header_stream << "};\n\n";

  if (total_inputs == 0) {
    header_stream
        << "inline constexpr std::array<InputSpec, 0> kOpInputSpecs = {};\n\n";
  } else {
    header_stream << "inline constexpr std::array<InputSpec, "
                     "kTotalInputPortCount> kOpInputSpecs = {\n";
    for (const auto &op : ops) {
      for (const auto &input : op.inputs) {
        header_stream << "    InputSpec{\"" << EscapeStringLiteral(input.name)
                      << "\", \"" << EscapeStringLiteral(input.description)
                      << "\", "
                      << "DTypeConstraintSpec{DTypeConstraintMode::";
        switch (input.mode) {
        case DTypeConstraintMode::Allow:
          header_stream << "Allow";
          break;
        case DTypeConstraintMode::Deny:
          header_stream << "Deny";
          break;
        case DTypeConstraintMode::Match:
          header_stream << "Match";
          break;
        }
        header_stream << ", " << FormatBitMask(input.allow_mask) << ", "
                      << FormatBitMask(input.deny_mask) << ", ";
        if (input.reference_input ==
            std::numeric_limits<std::uint32_t>::max()) {
          header_stream << "kInvalidIndex";
        } else {
          header_stream << input.reference_input;
        }
        header_stream << ", " << (input.allow_promotion ? "true" : "false")
                      << ", " << (input.require_same_shape ? "true" : "false")
                      << "}, " << (input.optional ? "true" : "false") << "},\n";
      }
    }
    header_stream << "};\n\n";
  }

  header_stream
      << "inline constexpr std::array<Range, kOpCount> kOpOutputRanges = {\n";
  std::size_t output_offset = 0;
  for (const auto &op : ops) {
    header_stream << "    Range{" << output_offset << ", " << op.outputs.size()
                  << "},\n";
    output_offset += op.outputs.size();
  }
  header_stream << "};\n\n";

  if (total_outputs == 0) {
    header_stream << "inline constexpr std::array<OutputSpec, 0> "
                     "kOpOutputSpecs = {};\n\n";
  } else {
    header_stream << "inline constexpr std::array<OutputSpec, "
                     "kTotalOutputPortCount> kOpOutputSpecs = {\n";
    for (const auto &op : ops) {
      for (const auto &output : op.outputs) {
        header_stream << "    OutputSpec{\"" << EscapeStringLiteral(output.name)
                      << "\", \"" << EscapeStringLiteral(output.description)
                      << "\", DTypeRuleKind::";
        switch (output.kind) {
        case DTypeRuleKind::Promote:
          header_stream << "Promote";
          break;
        case DTypeRuleKind::SameAs:
          header_stream << "SameAs";
          break;
        case DTypeRuleKind::Fixed:
          header_stream << "Fixed";
          break;
        case DTypeRuleKind::Custom:
          header_stream << "Custom";
          break;
        }
        header_stream << ", " << FormatBitMask(output.input_mask) << ", ";
        if (output.reference_input ==
            std::numeric_limits<std::uint32_t>::max()) {
          header_stream << "kInvalidIndex";
        } else {
          header_stream << output.reference_input;
        }
        header_stream << ", ";
        if (output.fixed_dtype == std::numeric_limits<std::uint32_t>::max()) {
          header_stream << "kInvalidIndex";
        } else {
          header_stream << output.fixed_dtype;
        }
        header_stream << ", \"" << EscapeStringLiteral(output.custom_function)
                      << "\"},\n";
      }
    }
    header_stream << "};\n\n";
  }

  header_stream << "inline constexpr std::array<Range, kOpCount> "
                   "kOpAttributeRanges = {\n";
  std::size_t attribute_offset = 0;
  for (const auto &op : ops) {
    header_stream << "    Range{" << attribute_offset << ", "
                  << op.attributes.size() << "},\n";
    attribute_offset += op.attributes.size();
  }
  header_stream << "};\n\n";

  if (total_attributes == 0) {
    header_stream << "inline constexpr std::array<AttributeSpec, 0> "
                     "kOpAttributes = {};\n\n";
  } else {
    header_stream << "inline constexpr std::array<AttributeSpec, "
                     "kTotalAttributeCount> kOpAttributes = {\n";
    for (const auto &op : ops) {
      for (const auto &attribute : op.attributes) {
        header_stream << "    AttributeSpec{\""
                      << EscapeStringLiteral(attribute.name) << "\", \""
                      << EscapeStringLiteral(attribute.type) << "\", \""
                      << EscapeStringLiteral(
                             attribute.default_value.value_or(""))
                      << "\", " << (attribute.default_value ? "true" : "false")
                      << ", " << (attribute.required ? "true" : "false")
                      << ", \"" << EscapeStringLiteral(attribute.description)
                      << "\"},\n";
      }
    }
    header_stream << "};\n\n";
  }

  header_stream << "inline constexpr std::array<ComputePolicySpec, kOpCount> "
                   "kOpComputePolicies = {\n";
  for (const auto &op : ops) {
    header_stream << "    ComputePolicySpec{ComputePolicyKind::";
    switch (op.compute_policy.kind) {
    case ComputePolicyKind::SameAs:
      header_stream << "SameAs";
      break;
    case ComputePolicyKind::Promote:
      header_stream << "Promote";
      break;
    case ComputePolicyKind::Fixed:
      header_stream << "Fixed";
      break;
    case ComputePolicyKind::DerivedComputeType:
      header_stream << "DerivedComputeType";
      break;
    case ComputePolicyKind::Custom:
      header_stream << "Custom";
      break;
    }
    header_stream << ", " << FormatBitMask(op.compute_policy.input_mask)
                  << ", ";
    if (op.compute_policy.reference_input ==
        std::numeric_limits<std::uint32_t>::max()) {
      header_stream << "kInvalidIndex";
    } else {
      header_stream << op.compute_policy.reference_input;
    }
    header_stream << ", ";
    if (op.compute_policy.fixed_dtype ==
        std::numeric_limits<std::uint32_t>::max()) {
      header_stream << "kInvalidIndex";
    } else {
      header_stream << op.compute_policy.fixed_dtype;
    }
    header_stream << ", \"" << EscapeStringLiteral(op.compute_policy.function)
                  << "\", \"" << EscapeStringLiteral(op.compute_policy.handler)
                  << "\"},\n";
  }
  header_stream << "};\n\n";

  header_stream << "inline constexpr std::array<ShapeInferenceSpec, kOpCount> "
                   "kOpShapeInference = {\n";
  for (const auto &op : ops) {
    header_stream << "    ShapeInferenceSpec{\""
                  << EscapeStringLiteral(op.shape_inference.kind) << "\", \""
                  << EscapeStringLiteral(op.shape_inference.function)
                  << "\", \""
                  << EscapeStringLiteral(op.shape_inference.description)
                  << "\"},\n";
  }
  header_stream << "};\n\n";

  std::vector<std::string> metadata_tags;
  metadata_tags.reserve(total_tags);
  std::vector<std::string> metadata_aliases;
  metadata_aliases.reserve(total_aliases);

  header_stream << "inline constexpr std::array<MetadataSpec, kOpCount> "
                   "kOpMetadata = {\n";
  std::size_t tag_offset = 0;
  std::size_t alias_offset = 0;
  for (const auto &op : ops) {
    header_stream << "    MetadataSpec{\""
                  << EscapeStringLiteral(op.metadata.description) << "\", "
                  << (op.metadata.commutative ? "true" : "false") << ", "
                  << (op.metadata.differentiable ? "true" : "false") << ", "
                  << "Range{" << tag_offset << ", " << op.metadata.tags.size()
                  << "}, "
                  << "Range{" << alias_offset << ", "
                  << op.metadata.aliases.size() << "}},\n";
    tag_offset += op.metadata.tags.size();
    alias_offset += op.metadata.aliases.size();
    metadata_tags.insert(metadata_tags.end(), op.metadata.tags.begin(),
                         op.metadata.tags.end());
    metadata_aliases.insert(metadata_aliases.end(), op.metadata.aliases.begin(),
                            op.metadata.aliases.end());
  }
  header_stream << "};\n\n";

  if (metadata_tags.empty()) {
    header_stream << "inline constexpr std::array<std::string_view, 0> "
                     "kOpMetadataTags = {};\n\n";
  } else {
    header_stream << "inline constexpr std::array<std::string_view, "
                  << metadata_tags.size() << "> kOpMetadataTags = {\n";
    for (const auto &tag : metadata_tags) {
      header_stream << "    \"" << EscapeStringLiteral(tag) << "\",\n";
    }
    header_stream << "};\n\n";
  }

  if (metadata_aliases.empty()) {
    header_stream << "inline constexpr std::array<std::string_view, 0> "
                     "kOpMetadataAliases = {};\n\n";
  } else {
    header_stream << "inline constexpr std::array<std::string_view, "
                  << metadata_aliases.size() << "> kOpMetadataAliases = {\n";
    for (const auto &alias : metadata_aliases) {
      header_stream << "    \"" << EscapeStringLiteral(alias) << "\",\n";
    }
    header_stream << "};\n\n";
  }

  header_stream << "}  // namespace orteaf::generated::ops_tables\n";

  generated.ops_tables_header = header_stream.str();
  return generated;
}

void WriteFile(const fs::path &path, const std::string &content) {
  std::ofstream file(path, std::ios::binary);
  if (!file.is_open()) {
    std::ostringstream oss;
    oss << "Failed to open file '" << path << "' for writing";
    Fail(oss.str());
  }
  file << content;
  if (!file.good()) {
    std::ostringstream oss;
    oss << "Failed to write file '" << path << "'";
    Fail(oss.str());
  }
}

} // namespace

int main(int argc, char **argv) try {
  if (argc != 3) {
    std::cerr << "Usage: gen_ops <ops_dir_or_file> <output_dir>\n";
    return 1;
  }

  const fs::path input_path = argv[1];
  const fs::path output_dir = argv[2];
  if (!fs::exists(input_path)) {
    std::cerr << "Input path does not exist: " << input_path << "\n";
    return 1;
  }

  // Derive dtype path: if input is a directory (e.g. configs/ops/),
  // go up one level to configs/ then into dtype/dtypes.yml.
  // If input is a file (e.g. configs/ops/ops.yml), go up two levels.
  fs::path dtype_yaml;
  if (fs::is_directory(input_path)) {
    dtype_yaml = input_path.parent_path() / "dtype" / "dtypes.yml";
  } else {
    dtype_yaml =
        input_path.parent_path().parent_path() / "dtype" / "dtypes.yml";
  }
  if (!fs::exists(dtype_yaml)) {
    std::cerr << "Required dtype catalog not found at " << dtype_yaml << "\n";
    return 1;
  }

  ParsedOpsFile parsed = ParseOpsFile(input_path);
  DTypeCatalog dtype_catalog = LoadDTypeCatalog(dtype_yaml);
  std::vector<ResolvedOp> resolved = ResolveOps(parsed, dtype_catalog);
  GeneratedData generated = GenerateOutputs(resolved, dtype_catalog);

  std::error_code ec;
  fs::create_directories(output_dir, ec);
  if (ec) {
    std::cerr << "Failed to create output directory '" << output_dir
              << "': " << ec.message() << "\n";
    return 1;
  }

  WriteFile(output_dir / "ops.def", generated.ops_def);
  WriteFile(output_dir / "ops_tables.h", generated.ops_tables_header);

  return 0;
} catch (const std::exception &e) {
  std::cerr << "gen_ops error: " << e.what() << "\n";
  return 1;
}
