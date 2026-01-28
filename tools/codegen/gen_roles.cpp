#include <yaml-cpp/yaml.h>

#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
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

struct RoleDefinition {
  std::string id;
  int value;
  std::string description;
};

struct ParsedConfig {
  std::string schema_version;
  std::vector<RoleDefinition> roles;
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

  const auto roles_node = root["roles"];
  if (!roles_node || !roles_node.IsSequence()) {
    Fail("Missing required sequence key 'roles'");
  }

  std::unordered_set<std::string> seen_ids;
  std::unordered_set<int> seen_values;

  config.roles.reserve(roles_node.size());
  for (std::size_t idx = 0; idx < roles_node.size(); ++idx) {
    const auto &node = roles_node[idx];
    if (!node.IsMap()) {
      std::ostringstream oss;
      oss << "Each role entry must be a mapping (index " << idx << ")";
      Fail(oss.str());
    }
    const std::string context = "roles[" + std::to_string(idx) + "]";

    RoleDefinition role;
    role.id = ReadString(node, "id", true, context);
    if (role.id.empty()) {
      std::ostringstream oss;
      oss << "Key 'id' must not be empty (" << context << ")";
      Fail(oss.str());
    }
    if (!seen_ids.insert(role.id).second) {
      std::ostringstream oss;
      oss << "Duplicate role id '" << role.id << "'";
      Fail(oss.str());
    }

    if (node["value"]) {
      std::ostringstream oss;
      oss << "Key 'value' is not allowed for roles (" << context
          << "). Values are auto-assigned by order.";
      Fail(oss.str());
    }
    role.value = static_cast<int>(idx);
    if (!seen_values.insert(role.value).second) {
      std::ostringstream oss;
      oss << "Duplicate role value " << role.value;
      Fail(oss.str());
    }

    role.description = ReadString(node, "description", false, context);

    config.roles.push_back(std::move(role));
  }

  return config;
}

struct GeneratedFiles {
  std::string role_def;
  std::string role_tables_header;
};

GeneratedFiles GenerateFiles(const ParsedConfig &config) {
  GeneratedFiles generated;
  const std::size_t role_count = config.roles.size();

  std::ostringstream def_stream;
  def_stream << "// Auto-generated. Do not edit.\n";
  for (const auto &role : config.roles) {
    def_stream << "ROLE(" << role.id << ", " << role.value << ")\n";
  }

  std::ostringstream header_stream;
  header_stream << "// Auto-generated. Do not edit.\n";
  header_stream << "#pragma once\n\n";
  header_stream << "#include <array>\n";
  header_stream << "#include <cstddef>\n";
  header_stream << "#include <cstdint>\n";
  header_stream << "#include <string_view>\n\n";
  header_stream << "namespace orteaf::internal::kernel {\n";
  header_stream << "enum class Role : std::uint8_t;\n";
  header_stream << "}  // namespace orteaf::internal::kernel\n\n";
  header_stream << "namespace orteaf::generated::role_tables {\n";
  header_stream << "using ::orteaf::internal::kernel::Role;\n\n";
  header_stream << "inline constexpr std::size_t kRoleCount = "
                << role_count << ";\n\n";

  header_stream << "inline constexpr std::array<std::string_view, "
                   "kRoleCount> kRoleDescriptions = {\n";
  for (const auto &role : config.roles) {
    header_stream << "    \"" << EscapeStringLiteral(role.description)
                  << "\",\n";
  }
  header_stream << "};\n\n";

  header_stream << "// Type info for each Role\n";
  header_stream << "template <Role Role>\n";
  header_stream << "struct RoleInfo;\n\n";

  for (const auto &role : config.roles) {
    header_stream << "template <>\n";
    header_stream << "struct RoleInfo<Role::" << role.id
                  << "> {\n";
    header_stream << "    static constexpr std::string_view kDescription = \""
                  << EscapeStringLiteral(role.description) << "\";\n";
    header_stream << "};\n\n";
  }

  header_stream << "}  // namespace orteaf::generated::role_tables\n";

  generated.role_def = def_stream.str();
  generated.role_tables_header = header_stream.str();
  return generated;
}

void WriteFile(const fs::path &path, const std::string &content) {
  fs::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    std::ostringstream oss;
    oss << "Failed to open output file '" << path << "'";
    Fail(oss.str());
  }
  out << content;
  if (!out) {
    std::ostringstream oss;
    oss << "Failed to write output file '" << path << "'";
    Fail(oss.str());
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: gen_roles <roles.yml> <output_dir>\n";
    return 1;
  }

  const fs::path yaml_path = argv[1];
  const fs::path output_dir = argv[2];

  try {
    auto config = ParseConfig(yaml_path);
    auto generated = GenerateFiles(config);

    WriteFile(output_dir / "role.def", generated.role_def);
    WriteFile(output_dir / "role_tables.h",
              generated.role_tables_header);
  } catch (const std::exception &e) {
    std::cerr << "gen_roles error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
