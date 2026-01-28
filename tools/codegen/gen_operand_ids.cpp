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

enum class Access {
  None,
  Read,
  Write,
  ReadWrite,
};

Access ParseAccess(std::string_view access_str, std::string_view context) {
  if (access_str == "None") {
    return Access::None;
  } else if (access_str == "Read") {
    return Access::Read;
  } else if (access_str == "Write") {
    return Access::Write;
  } else if (access_str == "ReadWrite") {
    return Access::ReadWrite;
  } else {
    std::ostringstream oss;
    oss << "Invalid access value '" << access_str << "' in " << context
        << ". Expected: None, Read, Write, or ReadWrite";
    Fail(oss.str());
  }
}

std::string_view AccessToString(Access access) {
  switch (access) {
  case Access::None:
    return "None";
  case Access::Read:
    return "Read";
  case Access::Write:
    return "Write";
  case Access::ReadWrite:
    return "ReadWrite";
  }
  return "None";
}

struct OperandDefinition {
  std::string id;
  int value;
  Access access;
  std::string description;
};

struct ParsedConfig {
  std::string schema_version;
  std::vector<OperandDefinition> operands;
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

  const auto operands_node = root["operands"];
  if (!operands_node || !operands_node.IsSequence()) {
    Fail("Missing required sequence key 'operands'");
  }

  std::unordered_set<std::string> seen_ids;
  std::unordered_set<int> seen_values;

  config.operands.reserve(operands_node.size());
  for (std::size_t idx = 0; idx < operands_node.size(); ++idx) {
    const auto &node = operands_node[idx];
    if (!node.IsMap()) {
      std::ostringstream oss;
      oss << "Each operand entry must be a mapping (index " << idx << ")";
      Fail(oss.str());
    }
    const std::string context = "operands[" + std::to_string(idx) + "]";

    OperandDefinition operand;
    operand.id = ReadString(node, "id", true, context);
    if (operand.id.empty()) {
      std::ostringstream oss;
      oss << "Key 'id' must not be empty (" << context << ")";
      Fail(oss.str());
    }
    if (!seen_ids.insert(operand.id).second) {
      std::ostringstream oss;
      oss << "Duplicate operand id '" << operand.id << "'";
      Fail(oss.str());
    }

    // Value is auto-assigned by order (explicit values are not allowed)
    if (node["value"]) {
      std::ostringstream oss;
      oss << "Key 'value' is not allowed for operands (" << context
          << "). Values are auto-assigned by order.";
      Fail(oss.str());
    }
    operand.value = static_cast<int>(idx);

    // Check for duplicate values
    if (!seen_values.insert(operand.value).second) {
      std::ostringstream oss;
      oss << "Duplicate operand value " << operand.value;
      Fail(oss.str());
    }

    // Parse access
    const std::string access_str = ReadString(node, "access", true, context);
    if (access_str.empty()) {
      std::ostringstream oss;
      oss << "Key 'access' must not be empty (" << context << ")";
      Fail(oss.str());
    }
    operand.access = ParseAccess(access_str, context);

    operand.description = ReadString(node, "description", false, context);
    if (operand.description.empty()) {
      operand.description = operand.id;
    }

    config.operands.emplace_back(std::move(operand));
  }

  return config;
}

struct ResolvedConfig {
  std::vector<OperandDefinition> operands;
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
  resolved.id_to_index.reserve(config.operands.size());
  resolved.value_to_index.reserve(config.operands.size());

  for (std::size_t i = 0; i < config.operands.size(); ++i) {
    resolved.id_to_index.emplace(config.operands[i].id, i);
    resolved.value_to_index.emplace(config.operands[i].value, i);
  }

  resolved.operands = std::move(config.operands);
  return resolved;
}

struct GeneratedData {
  std::string operand_id_def;
  std::string operand_id_tables_header;
};

GeneratedData GenerateOutputs(const ResolvedConfig &resolved) {
  const auto operand_count = resolved.operands.size();
  if (operand_count == 0) {
    Fail("No operands defined");
  }

  // Generate .def file
  std::ostringstream def_stream;
  def_stream << "// Auto-generated. Do not edit.\n";
  for (const auto &operand : resolved.operands) {
    def_stream << "OPERAND_ID(" << operand.id << ", " << operand.value << ")\n";
  }

  // Generate tables header
  std::ostringstream header_stream;
  header_stream << "// Auto-generated. Do not edit.\n";
  header_stream << "#pragma once\n\n";
  header_stream << "#include <array>\n";
  header_stream << "#include <cstddef>\n";
  header_stream << "#include <cstdint>\n";
  header_stream << "#include <string_view>\n\n";
  header_stream << "#include <orteaf/internal/kernel/core/access.h>\n\n";
  header_stream << "namespace orteaf::internal::kernel {\n";
  header_stream << "enum class OperandId : std::uint64_t;\n";
  header_stream << "}  // namespace orteaf::internal::kernel\n\n";
  header_stream << "namespace orteaf::generated::operand_id_tables {\n";
  header_stream << "using ::orteaf::internal::kernel::Access;\n";
  header_stream << "using ::orteaf::internal::kernel::OperandId;\n\n";
  header_stream << "inline constexpr std::size_t kOperandIdCount = "
                << operand_count << ";\n\n";

  // Description table
  header_stream << "inline constexpr std::array<std::string_view, "
                   "kOperandIdCount> kOperandIdDescriptions = {\n";
  for (const auto &operand : resolved.operands) {
    header_stream << "    \"" << EscapeStringLiteral(operand.description)
                  << "\",\n";
  }
  header_stream << "};\n\n";

  // Access table
  header_stream << "inline constexpr std::array<Access, "
                   "kOperandIdCount> kOperandIdAccesses = {\n";
  for (const auto &operand : resolved.operands) {
    header_stream << "    Access::" << AccessToString(operand.access) << ",\n";
  }
  header_stream << "};\n\n";

  // Type info (template specializations for each OperandId)
  header_stream << "// Type info for each OperandId\n";
  header_stream << "template <OperandId ID>\n";
  header_stream << "struct OperandTypeInfo;\n\n";

  for (const auto &operand : resolved.operands) {
    header_stream << "template <>\n";
    header_stream << "struct OperandTypeInfo<OperandId::" << operand.id
                  << "> {\n";
    header_stream << "    static constexpr Access kAccess = Access::"
                  << AccessToString(operand.access) << ";\n";
    header_stream << "    static constexpr std::string_view kDescription = \""
                  << EscapeStringLiteral(operand.description) << "\";\n";
    header_stream << "};\n\n";
  }

  header_stream << "}  // namespace orteaf::generated::operand_id_tables\n";

  GeneratedData generated;
  generated.operand_id_def = def_stream.str();
  generated.operand_id_tables_header = header_stream.str();
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
    std::cerr << "Usage: gen_operand_ids <operand_ids.yml> <output_dir>\n";
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

  WriteFile(output_dir / "operand_id.def", generated.operand_id_def);
  WriteFile(output_dir / "operand_id_tables.h",
            generated.operand_id_tables_header);

  return 0;
} catch (const std::exception &e) {
  std::cerr << "gen_operand_ids error: " << e.what() << "\n";
  return 1;
}
