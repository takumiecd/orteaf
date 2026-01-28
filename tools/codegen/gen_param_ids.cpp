#include <yaml-cpp/yaml.h>

#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
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

std::string ReadString(const YAML::Node &node, std::string_view key,
                       bool required, std::string_view context) {
  const auto value = node[key];
  if (!value) {
    if (required) {
      std::ostringstream oss;
      oss << "Missing required key '" << key << "' in " << context;
      Fail(oss.str());
    }
    return {};
  }
  if (!value.IsScalar()) {
    std::ostringstream oss;
    oss << "Key '" << key << "' must be a scalar in " << context;
    Fail(oss.str());
  }
  return value.as<std::string>();
}

int ReadInt(const YAML::Node &node, std::string_view key, bool required,
            std::string_view context) {
  const auto value = node[key];
  if (!value) {
    if (required) {
      std::ostringstream oss;
      oss << "Missing required key '" << key << "' in " << context;
      Fail(oss.str());
    }
    return -1;
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

struct ParamDefinition {
  std::string id;
  int value;
  std::string cpp_type;
  std::string description;
};

struct ParsedConfig {
  std::string schema_version;
  std::vector<ParamDefinition> params;
};

ParsedConfig ParseConfig(const fs::path &yaml_path) {
  YAML::Node root;
  try {
    root = YAML::LoadFile(yaml_path.string());
  } catch (const std::exception &e) {
    std::ostringstream oss;
    oss << "Failed to load YAML '" << yaml_path << "': " << e.what();
    Fail(oss.str());
  }

  if (!root || !root.IsMap()) {
    std::ostringstream oss;
    oss << "Root of '" << yaml_path << "' must be a mapping";
    Fail(oss.str());
  }

  ParsedConfig config;
  const auto schema_version_node = root["schema_version"];
  if (!schema_version_node || !schema_version_node.IsScalar()) {
    Fail("Missing required scalar key 'schema_version'");
  }
  config.schema_version = schema_version_node.as<std::string>();

  const auto params_node = root["params"];
  if (!params_node || !params_node.IsSequence()) {
    Fail("Missing required sequence key 'params'");
  }

  std::unordered_set<std::string> seen_ids;
  std::unordered_set<int> seen_values;

  config.params.reserve(params_node.size());
  for (std::size_t idx = 0; idx < params_node.size(); ++idx) {
    const auto &node = params_node[idx];
    if (!node.IsMap()) {
      std::ostringstream oss;
      oss << "Each param entry must be a mapping (index " << idx << ")";
      Fail(oss.str());
    }
    const std::string context = "params[" + std::to_string(idx) + "]";

    ParamDefinition param;
    param.id = ReadString(node, "id", true, context);
    if (param.id.empty()) {
      std::ostringstream oss;
      oss << "Key 'id' must not be empty (" << context << ")";
      Fail(oss.str());
    }
    if (!seen_ids.insert(param.id).second) {
      std::ostringstream oss;
      oss << "Duplicate param id '" << param.id << "'";
      Fail(oss.str());
    }

    // Value is auto-assigned by order (explicit values are not allowed)
    if (node["value"]) {
      std::ostringstream oss;
      oss << "Key 'value' is not allowed for params (" << context
          << "). Values are auto-assigned by order.";
      Fail(oss.str());
    }
    param.value = static_cast<int>(idx);

    // Check for duplicate values
    if (!seen_values.insert(param.value).second) {
      std::ostringstream oss;
      oss << "Duplicate param value " << param.value;
      Fail(oss.str());
    }

    param.cpp_type = ReadString(node, "cpp_type", true, context);
    if (param.cpp_type.empty()) {
      std::ostringstream oss;
      oss << "Key 'cpp_type' must not be empty (" << context << ")";
      Fail(oss.str());
    }

    param.description = ReadString(node, "description", false, context);
    if (param.description.empty()) {
      param.description = param.id;
    }

    config.params.emplace_back(std::move(param));
  }

  return config;
}

struct ResolvedConfig {
  std::vector<ParamDefinition> params;
  std::unordered_map<std::string, std::size_t> id_to_index;
  std::unordered_map<int, std::size_t> value_to_index;
};

ResolvedConfig ResolveConfig(ParsedConfig config) {
  if (config.schema_version != "1.0") {
    std::ostringstream oss;
    oss << "Unsupported schema_version '" << config.schema_version
        << "', expected '1.0'";
    Fail(oss.str());
  }

  ResolvedConfig resolved;
  resolved.id_to_index.reserve(config.params.size());
  resolved.value_to_index.reserve(config.params.size());

  for (std::size_t i = 0; i < config.params.size(); ++i) {
    resolved.id_to_index.emplace(config.params[i].id, i);
    resolved.value_to_index.emplace(config.params[i].value, i);
  }

  resolved.params = std::move(config.params);
  return resolved;
}

struct GeneratedData {
  std::string param_id_def;
  std::string param_id_tables_header;
};

GeneratedData GenerateOutputs(const ResolvedConfig &resolved) {
  const auto param_count = resolved.params.size();
  if (param_count == 0) {
    Fail("No params defined");
  }

  // Generate .def file
  std::ostringstream def_stream;
  def_stream << "// Auto-generated. Do not edit.\n";
  for (const auto &param : resolved.params) {
    def_stream << "PARAM_ID(" << param.id << ", " << param.value << ")\n";
  }

  // Generate tables header
  std::ostringstream header_stream;
  header_stream << "// Auto-generated. Do not edit.\n";
  header_stream << "#pragma once\n\n";
  header_stream << "#include <cstddef>\n";
  header_stream << "#include <cstdint>\n";
  header_stream << "#include <string_view>\n";
  header_stream << "#include <variant>\n";
  header_stream << "#include <orteaf/internal/base/array_view.h>\n\n";
  header_stream << "namespace orteaf::internal::kernel {\n";
  header_stream << "enum class ParamId : std::uint64_t;\n";
  header_stream << "}  // namespace orteaf::internal::kernel\n\n";
  header_stream << "namespace orteaf::generated::param_id_tables {\n";
  header_stream << "using ::orteaf::internal::kernel::ParamId;\n\n";
  header_stream << "inline constexpr std::size_t kParamIdCount = "
                << param_count << ";\n\n";

  // Unified ParamInfo template
  header_stream << "// Parameter information for each ParamId\n";
  header_stream << "template <ParamId ID>\n";
  header_stream << "struct ParamInfo;\n\n";

  for (const auto &param : resolved.params) {
    header_stream << "template <>\n";
    header_stream << "struct ParamInfo<ParamId::" << param.id << "> {\n";
    header_stream << "    using Type = " << param.cpp_type << ";\n";
    header_stream << "    static constexpr std::string_view kTypeName = \""
                  << EscapeStringLiteral(param.cpp_type) << "\";\n";
    header_stream << "    static constexpr std::string_view kDescription = \""
                  << EscapeStringLiteral(param.description) << "\";\n";
    header_stream << "};\n\n";
  }

  header_stream << "template <ParamId ID>\n";
  header_stream << "using ParamValueTypeT = typename ParamInfo<ID>::Type;\n\n";

  // Collect unique C++ types (excluding "Tensor")
  std::unordered_set<std::string> unique_types;
  for (const auto &param : resolved.params) {
    if (param.cpp_type != "Tensor") {
      unique_types.insert(param.cpp_type);
    }
  }

  // Generate ParamValue variant
  header_stream << "// Type-erased parameter value variant\n";

  header_stream << "using ParamValue = std::variant<\n";

  // Add primitive types
  std::vector<std::string> variant_types;
  if (unique_types.count("float"))
    variant_types.push_back("  float");
  if (unique_types.count("double"))
    variant_types.push_back("  double");
  if (unique_types.count("int"))
    variant_types.push_back("  int");
  if (unique_types.count("std::size_t"))
    variant_types.push_back("  std::size_t");
  if (unique_types.count("std::uint32_t"))
    variant_types.push_back("  std::uint32_t");
  if (unique_types.count("std::int64_t"))
    variant_types.push_back("  std::int64_t");
  if (unique_types.count("void*"))
    variant_types.push_back("  void*");

  // Add ArrayView types for common numeric types
  variant_types.push_back("  ::orteaf::internal::base::ArrayView<const int>");
  variant_types.push_back(
      "  ::orteaf::internal::base::ArrayView<const std::size_t>");
  variant_types.push_back(
      "  ::orteaf::internal::base::ArrayView<const std::int64_t>");
  variant_types.push_back("  ::orteaf::internal::base::ArrayView<const float>");
  variant_types.push_back(
      "  ::orteaf::internal::base::ArrayView<const double>");

  // Output variant types
  for (std::size_t i = 0; i < variant_types.size(); ++i) {
    header_stream << variant_types[i];
    if (i < variant_types.size() - 1) {
      header_stream << ",\n";
    }
  }
  header_stream << "\n>;\n\n";

  header_stream << "}  // namespace orteaf::generated::param_id_tables\n";

  GeneratedData generated;
  generated.param_id_def = def_stream.str();
  generated.param_id_tables_header = header_stream.str();
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
    std::cerr << "Usage: gen_param_ids <param_ids.yml> <output_dir>\n";
    return 1;
  }

  const fs::path input_path = argv[1];
  const fs::path output_dir = argv[2];
  if (!fs::exists(input_path)) {
    std::cerr << "Input file does not exist: " << input_path << "\n";
    return 1;
  }

  ParsedConfig parsed = ParseConfig(input_path);
  ResolvedConfig resolved = ResolveConfig(std::move(parsed));
  GeneratedData generated = GenerateOutputs(resolved);

  std::error_code ec;
  fs::create_directories(output_dir, ec);
  if (ec) {
    std::cerr << "Failed to create output directory '" << output_dir
              << "': " << ec.message() << "\n";
    return 1;
  }

  WriteFile(output_dir / "param_id.def", generated.param_id_def);
  WriteFile(output_dir / "param_id_tables.h", generated.param_id_tables_header);

  return 0;
} catch (const std::exception &e) {
  std::cerr << "gen_param_ids error: " << e.what() << "\n";
  return 1;
}
