#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

[[noreturn]] void Fail(const std::string &message) {
  throw std::runtime_error(message);
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

std::string EscapeStringLiteral(std::string_view value) {
  std::string escaped;
  escaped.reserve(value.size() + 4);
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

std::string ReadRequiredString(const YAML::Node &node, std::string_view key,
                               std::string_view context) {
  const auto value = node[key];
  if (!value) {
    std::ostringstream oss;
    oss << "Missing required key '" << key << "' in " << context;
    Fail(oss.str());
  }
  if (!value.IsScalar()) {
    std::ostringstream oss;
    oss << "Key '" << key << "' must be a scalar in " << context;
    Fail(oss.str());
  }
  const std::string result = value.as<std::string>();
  if (result.empty()) {
    std::ostringstream oss;
    oss << "Key '" << key << "' must not be empty in " << context;
    Fail(oss.str());
  }
  return result;
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

bool LooksLikeIdentifier(std::string_view value) {
  if (value.empty()) {
    return false;
  }
  const unsigned char first = static_cast<unsigned char>(value.front());
  if (!(std::isalpha(first) || value.front() == '_')) {
    return false;
  }
  for (const char ch : value) {
    const unsigned char uch = static_cast<unsigned char>(ch);
    if (!(std::isalnum(uch) || ch == '_')) {
      return false;
    }
  }
  return true;
}

int ReadScalarInt(const YAML::Node &node, std::string_view key,
                  std::string_view context) {
  const auto value = node[key];
  if (!value) {
    std::ostringstream oss;
    oss << "Missing required integer key '" << key << "' in " << context;
    Fail(oss.str());
  }
  if (!value.IsScalar()) {
    std::ostringstream oss;
    oss << "Key '" << key << "' must be an integer scalar in " << context;
    Fail(oss.str());
  }
  return value.as<int>();
}

std::optional<int> ReadOptionalInt(const YAML::Node &node, std::string_view key,
                                   std::string_view context) {
  const auto value = node[key];
  if (!value) {
    return std::nullopt;
  }
  if (!value.IsScalar()) {
    std::ostringstream oss;
    oss << "Key '" << key << "' must be an integer scalar in " << context;
    Fail(oss.str());
  }
  return value.as<int>();
}

std::vector<int> ReadOptionalIntList(const YAML::Node &node,
                                     std::string_view key,
                                     std::string_view context) {
  const auto value = node[key];
  if (!value) {
    return {};
  }
  std::vector<int> result;
  if (value.IsSequence()) {
    result.reserve(value.size());
    for (std::size_t i = 0; i < value.size(); ++i) {
      const auto &item = value[i];
      if (!item.IsScalar()) {
        std::ostringstream oss;
        oss << "Entry '" << key << "[" << i << "]' must be an integer in "
            << context;
        Fail(oss.str());
      }
      result.push_back(item.as<int>());
    }
  } else if (value.IsScalar()) {
    result.push_back(value.as<int>());
  } else {
    std::ostringstream oss;
    oss << "Key '" << key << "' must be an integer or sequence in " << context;
    Fail(oss.str());
  }
  return result;
}

std::vector<std::string> ReadOptionalStringList(const YAML::Node &node,
                                                std::string_view key,
                                                std::string_view context) {
  const auto value = node[key];
  if (!value) {
    return {};
  }
  std::vector<std::string> result;
  if (value.IsSequence()) {
    result.reserve(value.size());
    for (std::size_t i = 0; i < value.size(); ++i) {
      const auto &item = value[i];
      if (!item.IsScalar()) {
        std::ostringstream oss;
        oss << "Entry '" << key << "[" << i << "]' must be a string in "
            << context;
        Fail(oss.str());
      }
      auto str = item.as<std::string>();
      if (str.empty()) {
        std::ostringstream oss;
        oss << "Entry '" << key << "[" << i << "]' must not be empty in "
            << context;
        Fail(oss.str());
      }
      result.push_back(std::move(str));
    }
  } else if (value.IsScalar()) {
    auto str = value.as<std::string>();
    if (str.empty()) {
      std::ostringstream oss;
      oss << "Key '" << key << "' must not be empty in " << context;
      Fail(oss.str());
    }
    result.push_back(std::move(str));
  } else {
    std::ostringstream oss;
    oss << "Key '" << key << "' must be a string or sequence in " << context;
    Fail(oss.str());
  }
  return result;
}

struct ExecutionInfo {
  std::string id;
  std::string display_name;
};

struct DetectSpec {
  std::string vendor;
  std::optional<int> family;
  std::vector<int> models;
  std::vector<std::string> features;
  std::vector<std::string> machine_ids;
  std::optional<int> compute_capability;
  std::optional<std::string> metal_family;
};

struct ArchitectureInput {
  std::string id;
  std::string execution_id;
  std::string display_name;
  std::string description;
  std::optional<std::string> parent;
  std::optional<DetectSpec> detect;
};

std::optional<DetectSpec> ParseDetectSpec(const YAML::Node &node,
                                          std::string_view context) {
  if (!node) {
    return std::nullopt;
  }
  if (!node.IsMap()) {
    std::ostringstream oss;
    oss << "Detect block in " << context << " must be a mapping";
    Fail(oss.str());
  }
  DetectSpec spec;
  ExpectKeys(node, context,
             {"vendor", "family", "model", "features", "machine_ids",
              "compute_capability", "metal_family"});
  if (const auto vendor = ReadOptionalString(node, "vendor", context); vendor) {
    spec.vendor = *vendor;
  }
  if (const auto family = ReadOptionalInt(node, "family", context); family) {
    spec.family = *family;
  }
  spec.models = ReadOptionalIntList(node, "model", context);
  spec.features = ReadOptionalStringList(node, "features", context);
  spec.machine_ids = ReadOptionalStringList(node, "machine_ids", context);
  if (const auto cc = ReadOptionalInt(node, "compute_capability", context);
      cc) {
    spec.compute_capability = *cc;
  }
  if (const auto metal = ReadOptionalString(node, "metal_family", context);
      metal) {
    spec.metal_family = *metal;
  }

  if (spec.vendor.empty() && !spec.family && spec.models.empty() &&
      spec.features.empty() && spec.machine_ids.empty() &&
      !spec.compute_capability && !spec.metal_family) {
    return std::nullopt;
  }
  return spec;
}

std::vector<ExecutionInfo>
ParseExecutionConfig(const fs::path &execution_yaml_path) {
  YAML::Node root;
  try {
    root = YAML::LoadFile(execution_yaml_path.string());
  } catch (const std::exception &e) {
    std::ostringstream oss;
    oss << "Failed to load execution YAML '" << execution_yaml_path
        << "': " << e.what();
    Fail(oss.str());
  }

  if (!root || !root.IsMap()) {
    Fail("Execution YAML root must be a mapping");
  }

  const auto executions_node = root["executions"];
  if (!executions_node || !executions_node.IsSequence()) {
    Fail("Execution YAML must contain a sequence 'executions'");
  }

  std::vector<ExecutionInfo> executions;
  executions.reserve(executions_node.size());
  std::unordered_set<std::string> seen;

  for (std::size_t i = 0; i < executions_node.size(); ++i) {
    const auto &node = executions_node[i];
    if (!node.IsMap()) {
      std::ostringstream oss;
      oss << "Each execution entry must be a mapping (index " << i << ")";
      Fail(oss.str());
    }
    const std::string context = "executions[" + std::to_string(i) + "]";

    ExecutionInfo info;
    info.id = ReadRequiredString(node, "id", context);
    if (!LooksLikeIdentifier(info.id)) {
      std::ostringstream oss;
      oss << "Execution id '" << info.id << "' is not a valid identifier ("
          << context << ")";
      Fail(oss.str());
    }
    if (!seen.insert(info.id).second) {
      std::ostringstream oss;
      oss << "Duplicate execution id '" << info.id << "'";
      Fail(oss.str());
    }
    info.display_name = ReadRequiredString(node, "display_name", context);
    executions.push_back(std::move(info));
  }

  if (executions.empty()) {
    Fail("At least one execution must be defined");
  }

  return executions;
}

std::vector<ArchitectureInput> ParseArchitectureConfig(
    const fs::path &architecture_yaml_path,
    const std::unordered_set<std::string> &valid_executions) {
  YAML::Node root;
  try {
    root = YAML::LoadFile(architecture_yaml_path.string());
  } catch (const std::exception &e) {
    std::ostringstream oss;
    oss << "Failed to load architecture YAML '" << architecture_yaml_path
        << "': " << e.what();
    Fail(oss.str());
  }

  if (!root || !root.IsMap()) {
    Fail("Architecture YAML root must be a mapping");
  }

  const auto schema_node = root["schema_version"];
  if (!schema_node || !schema_node.IsScalar()) {
    Fail("Missing required scalar key 'schema_version' in architecture YAML");
  }

  const auto architectures_node = root["architectures"];
  if (!architectures_node || !architectures_node.IsSequence()) {
    Fail("Missing required sequence key 'architectures'");
  }

  std::vector<ArchitectureInput> architectures;
  architectures.reserve(architectures_node.size());
  std::unordered_map<std::string, std::unordered_set<std::string>>
      seen_per_execution;

  for (std::size_t i = 0; i < architectures_node.size(); ++i) {
    const auto &node = architectures_node[i];
    if (!node.IsMap()) {
      std::ostringstream oss;
      oss << "Each architecture entry must be a mapping (index " << i << ")";
      Fail(oss.str());
    }
    const std::string context = "architectures[" + std::to_string(i) + "]";

    ArchitectureInput input;
    input.id = ReadRequiredString(node, "id", context);
    if (!LooksLikeIdentifier(input.id)) {
      std::ostringstream oss;
      oss << "Architecture id '" << input.id << "' is not a valid identifier ("
          << context << ")";
      Fail(oss.str());
    }

    input.execution_id = ReadRequiredString(node, "execution", context);
    if (!valid_executions.count(input.execution_id)) {
      std::ostringstream oss;
      oss << "Architecture '" << input.id << "' references unknown execution '"
          << input.execution_id << "'";
      Fail(oss.str());
    }

    if (!seen_per_execution[input.execution_id].insert(input.id).second) {
      std::ostringstream oss;
      oss << "Duplicate architecture id '" << input.id << "' for execution '"
          << input.execution_id << "'";
      Fail(oss.str());
    }

    input.display_name = ReadRequiredString(node, "display_name", context);

    const auto metadata_node = node["metadata"];
    if (metadata_node) {
      if (!metadata_node.IsMap()) {
        std::ostringstream oss;
        oss << "Metadata for " << context << " must be a mapping";
        Fail(oss.str());
      }
      const std::string metadata_context = context + ".metadata";
      ExpectKeys(metadata_node, metadata_context,
                 {"description", "detect", "parent"});
      if (const auto parent =
              ReadOptionalString(metadata_node, "parent", metadata_context);
          parent) {
        input.parent = *parent;
      }
      if (const auto desc = ReadOptionalString(metadata_node, "description",
                                               metadata_context);
          desc) {
        input.description = *desc;
      }
      if (const auto detect_node = metadata_node["detect"]) {
        const std::string detect_context = metadata_context + ".detect";
        input.detect = ParseDetectSpec(detect_node, detect_context);
      }
    }

    architectures.push_back(std::move(input));
  }

  return architectures;
}

struct ResolvedArchitecture {
  std::string enum_name;
  std::string architecture_id;
  std::string display_name;
  std::string description;
  std::size_t execution_index;
  std::uint16_t local_index;
  std::optional<std::string>
      parent_id; // Raw parent from YAML (to resolve later)
  std::optional<DetectSpec> detect;
};

struct GeneratedData {
  std::string architecture_def;
  std::string architecture_tables_header;
};

GeneratedData
GenerateOutputs(const std::vector<ExecutionInfo> &executions,
                const std::vector<ArchitectureInput> &architectures) {
  // Group user-defined architectures by execution.
  std::unordered_map<std::string, std::vector<ArchitectureInput>> by_execution;
  by_execution.reserve(executions.size());
  for (const auto &arch : architectures) {
    by_execution[arch.execution_id].push_back(arch);
  }

  // Prepare resolved list in execution order.
  std::vector<ResolvedArchitecture> resolved;
  resolved.reserve(executions.size() + architectures.size());

  std::vector<std::size_t> execution_offsets;
  execution_offsets.reserve(executions.size() + 1);
  std::vector<std::size_t> execution_counts(executions.size(), 0);

  for (std::size_t execution_index = 0; execution_index < executions.size();
       ++execution_index) {
    const auto &execution = executions[execution_index];
    execution_offsets.push_back(resolved.size());

    // Build the reserved generic ID for this execution (e.g., "CpuGeneric").
    const std::string reserved_generic_id = execution.id + "Generic";
    std::string reserved_generic_lower = reserved_generic_id;
    std::transform(reserved_generic_lower.begin(), reserved_generic_lower.end(),
                   reserved_generic_lower.begin(), [](unsigned char ch) {
                     return static_cast<char>(std::tolower(ch));
                   });

    const auto duplicate_it = by_execution.find(execution.id);
    if (duplicate_it != by_execution.end()) {
      const auto &entries = duplicate_it->second;
      if (std::any_of(
              entries.begin(), entries.end(),
              [&reserved_generic_lower](const ArchitectureInput &input) {
                std::string lower = input.id;
                std::transform(lower.begin(), lower.end(), lower.begin(),
                               [](unsigned char ch) {
                                 return static_cast<char>(std::tolower(ch));
                               });
                return lower == reserved_generic_lower;
              })) {
        std::ostringstream oss;
        oss << "Execution '" << execution.id << "' defines architecture id '"
            << reserved_generic_id
            << "', which is reserved for the auto-generated fallback";
        Fail(oss.str());
      }
    }

    auto make_enum_name = [](const std::string &execution_id,
                             const std::string &arch_id) {
      std::ostringstream oss;
      oss << execution_id << arch_id;
      return oss.str();
    };

    // Auto-insert generic entry (local index 0).
    ResolvedArchitecture generic;
    generic.enum_name = make_enum_name(execution.id, "Generic");
    generic.architecture_id = execution.id + "Generic";
    generic.display_name = "Generic " + execution.display_name;
    generic.description =
        "Execution-wide fallback architecture for " + execution.display_name;
    generic.execution_index = execution_index;
    generic.local_index = 0;
    generic.detect = std::nullopt;
    resolved.push_back(std::move(generic));

    // Append user entries honoring input order.
    const auto by_execution_it = by_execution.find(execution.id);
    if (by_execution_it != by_execution.end()) {
      const auto &entries = by_execution_it->second;
      std::uint16_t local_index = 1;
      for (const auto &entry : entries) {
        ResolvedArchitecture resolved_entry;
        resolved_entry.enum_name = make_enum_name(execution.id, entry.id);
        resolved_entry.architecture_id = entry.id;
        resolved_entry.display_name = entry.display_name;
        resolved_entry.description = entry.description;
        resolved_entry.execution_index = execution_index;
        resolved_entry.local_index = local_index++;
        resolved_entry.detect = entry.detect;
        resolved_entry.parent_id =
            entry.parent; // Will be resolved to index later
        resolved.push_back(std::move(resolved_entry));
      }
      execution_counts[execution_index] = entries.size() + 1; // +1 for generic
    } else {
      execution_counts[execution_index] = 1; // generic only
    }
  }
  execution_offsets.push_back(resolved.size());

  // Build architecture_id -> global index lookup, scoped by execution.
  // Key: (execution_index, architecture_id) -> global index
  std::unordered_map<std::string, std::size_t> arch_id_to_global_index;
  for (std::size_t i = 0; i < resolved.size(); ++i) {
    // Use "execution_id:arch_id" as key to scope by execution
    const std::string key = executions[resolved[i].execution_index].id + ":" +
                            resolved[i].architecture_id;
    arch_id_to_global_index[key] = i;
  }

  // Resolve parent_id to parent_index.
  // kInvalidParent (0xFFFF) means no parent (Generic architectures).
  constexpr std::uint16_t kInvalidParent = 0xFFFF;
  std::vector<std::uint16_t> parent_indices(resolved.size(), kInvalidParent);

  for (std::size_t i = 0; i < resolved.size(); ++i) {
    const auto &arch = resolved[i];

    if (arch.local_index == 0) {
      // Generic architectures have no parent
      parent_indices[i] = kInvalidParent;
      continue;
    }

    // User-defined architectures MUST have explicit parent
    if (!arch.parent_id) {
      std::ostringstream oss;
      oss << "Architecture '" << arch.architecture_id
          << "' is missing required 'parent' field in metadata. "
          << "Use 'parent: \"" << executions[arch.execution_index].id
          << "Generic\"' to derive from the base architecture.";
      Fail(oss.str());
    }

    // Resolve explicit parent
    const std::string parent_key =
        executions[arch.execution_index].id + ":" + *arch.parent_id;
    const auto parent_it = arch_id_to_global_index.find(parent_key);
    if (parent_it == arch_id_to_global_index.end()) {
      std::ostringstream oss;
      oss << "Architecture '" << arch.architecture_id
          << "' references unknown parent '" << *arch.parent_id
          << "' in execution '" << executions[arch.execution_index].id << "'";
      Fail(oss.str());
    }
    parent_indices[i] = static_cast<std::uint16_t>(parent_it->second);
  }

  // Cycle detection using DFS
  {
    enum class State { Unvisited, Visiting, Visited };
    std::vector<State> states(resolved.size(), State::Unvisited);

    std::function<void(std::size_t, std::vector<std::size_t> &)> detect_cycle =
        [&](std::size_t idx, std::vector<std::size_t> &path) {
          if (states[idx] == State::Visited) {
            return;
          }
          if (states[idx] == State::Visiting) {
            // Found cycle - build error message
            std::ostringstream oss;
            oss << "Cycle detected in architecture parent chain: ";
            bool in_cycle = false;
            for (std::size_t i = 0; i < path.size(); ++i) {
              if (path[i] == idx) {
                in_cycle = true;
              }
              if (in_cycle) {
                oss << resolved[path[i]].architecture_id;
                if (i + 1 < path.size()) {
                  oss << " -> ";
                }
              }
            }
            oss << " -> " << resolved[idx].architecture_id;
            Fail(oss.str());
          }

          states[idx] = State::Visiting;
          path.push_back(idx);

          const auto parent_idx = parent_indices[idx];
          if (parent_idx != kInvalidParent) {
            detect_cycle(parent_idx, path);
          }

          path.pop_back();
          states[idx] = State::Visited;
        };

    for (std::size_t i = 0; i < resolved.size(); ++i) {
      std::vector<std::size_t> path;
      detect_cycle(i, path);
    }
  }

  GeneratedData generated;

  {
    std::ostringstream def_stream;
    def_stream << "// Auto-generated. Do not edit.\n";
    for (const auto &arch : resolved) {
      def_stream << "ARCHITECTURE(" << arch.enum_name
                 << ", execution::Execution::"
                 << executions[arch.execution_index].id << ", "
                 << arch.local_index << ", \""
                 << EscapeStringLiteral(arch.architecture_id) << "\", \""
                 << EscapeStringLiteral(arch.display_name) << "\", \""
                 << EscapeStringLiteral(arch.description) << "\")\n";
    }
    generated.architecture_def = def_stream.str();
  }

  {
    std::ostringstream header_stream;
    header_stream << "// Auto-generated. Do not edit.\n";
    header_stream << "#pragma once\n\n";
    header_stream << "#include <array>\n";
    header_stream << "#include <cstddef>\n";
    header_stream << "#include <cstdint>\n";
    header_stream << "#include <string_view>\n\n";
    header_stream << "namespace orteaf::generated::architecture_tables {\n";
    header_stream << "inline constexpr std::size_t kArchitectureCount = "
                  << resolved.size() << ";\n";
    header_stream << "inline constexpr std::size_t kExecutionCount = "
                  << executions.size() << ";\n\n";

    auto emit_array = [&](const std::string_view decl, auto value_fn) {
      header_stream << "inline constexpr " << decl << " = {\n";
      for (const auto &arch : resolved) {
        header_stream << "    " << value_fn(arch) << ",\n";
      }
      header_stream << "};\n\n";
    };

    emit_array("std::array<std::uint16_t, kArchitectureCount> "
               "kArchitectureExecutionIndices",
               [](const ResolvedArchitecture &arch) {
                 return std::to_string(arch.execution_index);
               });
    emit_array("std::array<std::uint16_t, kArchitectureCount> "
               "kArchitectureLocalIndices",
               [](const ResolvedArchitecture &arch) {
                 return std::to_string(arch.local_index);
               });
    emit_array(
        "std::array<std::string_view, kArchitectureCount> kArchitectureIds",
        [](const ResolvedArchitecture &arch) {
          std::ostringstream oss;
          oss << '"' << EscapeStringLiteral(arch.architecture_id) << '"';
          return oss.str();
        });
    emit_array("std::array<std::string_view, kArchitectureCount> "
               "kArchitectureDisplayNames",
               [](const ResolvedArchitecture &arch) {
                 std::ostringstream oss;
                 oss << '"' << EscapeStringLiteral(arch.display_name) << '"';
                 return oss.str();
               });
    emit_array("std::array<std::string_view, kArchitectureCount> "
               "kArchitectureDescriptions",
               [](const ResolvedArchitecture &arch) {
                 std::ostringstream oss;
                 oss << '"' << EscapeStringLiteral(arch.description) << '"';
                 return oss.str();
               });

    // Parent indices table
    header_stream
        << "inline constexpr std::uint16_t kInvalidParent = 0xFFFF;\n\n";
    header_stream
        << "inline constexpr std::array<std::uint16_t, kArchitectureCount> "
           "kArchitectureParentIndices = {\n";
    for (const auto parent_idx : parent_indices) {
      if (parent_idx == kInvalidParent) {
        header_stream << "    kInvalidParent,\n";
      } else {
        header_stream << "    " << parent_idx << ",\n";
      }
    }
    header_stream << "};\n\n";

    header_stream << "inline constexpr std::array<std::size_t, "
                  << executions.size()
                  << "> kExecutionArchitectureCounts = {\n";
    for (const auto count : execution_counts) {
      header_stream << "    " << count << ",\n";
    }
    header_stream << "};\n\n";

    header_stream << "inline constexpr std::array<std::size_t, "
                  << (executions.size() + 1)
                  << "> kExecutionArchitectureOffsets = {\n";
    for (const auto offset : execution_offsets) {
      header_stream << "    " << offset << ",\n";
    }
    header_stream << "};\n\n";

    const std::size_t arch_count = resolved.size();
    std::vector<std::string> detect_vendors(arch_count);
    std::vector<int> detect_cpu_families(arch_count, -1);
    std::vector<int> detect_compute_caps(arch_count, -1);
    std::vector<std::string> detect_metal_families(arch_count);
    std::vector<std::size_t> cpu_model_offsets;
    std::vector<int> cpu_model_entries;
    std::vector<std::size_t> feature_offsets;
    std::vector<std::string> feature_entries;
    std::vector<std::size_t> machine_id_offsets;
    std::vector<std::string> machine_id_entries;

    cpu_model_offsets.reserve(arch_count + 1);
    feature_offsets.reserve(arch_count + 1);
    machine_id_offsets.reserve(arch_count + 1);

    for (std::size_t idx = 0; idx < arch_count; ++idx) {
      cpu_model_offsets.push_back(cpu_model_entries.size());
      feature_offsets.push_back(feature_entries.size());
      machine_id_offsets.push_back(machine_id_entries.size());

      const auto &detect = resolved[idx].detect;
      if (!detect) {
        continue;
      }
      if (!detect->vendor.empty()) {
        detect_vendors[idx] = detect->vendor;
      }
      if (detect->family) {
        detect_cpu_families[idx] = *detect->family;
      }
      if (detect->compute_capability) {
        detect_compute_caps[idx] = *detect->compute_capability;
      }
      if (detect->metal_family) {
        detect_metal_families[idx] = *detect->metal_family;
      }
      cpu_model_entries.insert(cpu_model_entries.end(), detect->models.begin(),
                               detect->models.end());
      feature_entries.insert(feature_entries.end(), detect->features.begin(),
                             detect->features.end());
      machine_id_entries.insert(machine_id_entries.end(),
                                detect->machine_ids.begin(),
                                detect->machine_ids.end());
    }

    cpu_model_offsets.push_back(cpu_model_entries.size());
    feature_offsets.push_back(feature_entries.size());
    machine_id_offsets.push_back(machine_id_entries.size());

    header_stream << "inline constexpr std::array<std::string_view, "
                     "kArchitectureCount> kArchitectureDetectVendors = {\n";
    for (const auto &vendor : detect_vendors) {
      header_stream << "    std::string_view{\"" << EscapeStringLiteral(vendor)
                    << "\"},\n";
    }
    header_stream << "};\n\n";

    header_stream << "inline constexpr std::array<int, kArchitectureCount> "
                     "kArchitectureDetectCpuFamilies = {\n";
    for (const auto family : detect_cpu_families) {
      header_stream << "    " << family << ",\n";
    }
    header_stream << "};\n\n";

    header_stream << "inline constexpr std::array<int, kArchitectureCount> "
                     "kArchitectureDetectComputeCapability = {\n";
    for (const auto cc : detect_compute_caps) {
      header_stream << "    " << cc << ",\n";
    }
    header_stream << "};\n\n";

    header_stream
        << "inline constexpr std::array<std::string_view, kArchitectureCount> "
           "kArchitectureDetectMetalFamilies = {\n";
    for (const auto &metal : detect_metal_families) {
      header_stream << "    std::string_view{\"" << EscapeStringLiteral(metal)
                    << "\"},\n";
    }
    header_stream << "};\n\n";

    header_stream << "inline constexpr std::array<std::size_t, "
                  << cpu_model_offsets.size()
                  << "> kArchitectureDetectCpuModelOffsets = {\n";
    for (const auto offset : cpu_model_offsets) {
      header_stream << "    " << offset << ",\n";
    }
    header_stream << "};\n\n";

    header_stream << "inline constexpr std::array<int, "
                  << cpu_model_entries.size()
                  << "> kArchitectureDetectCpuModels = {\n";
    for (const auto value : cpu_model_entries) {
      header_stream << "    " << value << ",\n";
    }
    header_stream << "};\n\n";

    header_stream << "inline constexpr std::array<std::size_t, "
                  << feature_offsets.size()
                  << "> kArchitectureDetectFeatureOffsets = {\n";
    for (const auto offset : feature_offsets) {
      header_stream << "    " << offset << ",\n";
    }
    header_stream << "};\n\n";

    header_stream << "inline constexpr std::array<std::string_view, "
                  << feature_entries.size()
                  << "> kArchitectureDetectFeatures = {\n";
    for (const auto &feature : feature_entries) {
      header_stream << "    std::string_view{\"" << EscapeStringLiteral(feature)
                    << "\"},\n";
    }
    header_stream << "};\n\n";

    header_stream << "inline constexpr std::array<std::size_t, "
                  << machine_id_offsets.size()
                  << "> kArchitectureDetectMachineIdOffsets = {\n";
    for (const auto offset : machine_id_offsets) {
      header_stream << "    " << offset << ",\n";
    }
    header_stream << "};\n\n";

    header_stream << "inline constexpr std::array<std::string_view, "
                  << machine_id_entries.size()
                  << "> kArchitectureDetectMachineIds = {\n";
    for (const auto &machine : machine_id_entries) {
      header_stream << "    std::string_view{\"" << EscapeStringLiteral(machine)
                    << "\"},\n";
    }
    header_stream << "};\n\n";

    header_stream << "}  // namespace orteaf::generated::architecture_tables\n";

    generated.architecture_tables_header = header_stream.str();
  }

  return generated;
}

void WriteFile(const fs::path &path, const std::string &content) {
  std::error_code ec;
  fs::create_directories(path.parent_path(), ec);
  if (ec) {
    std::ostringstream oss;
    oss << "Failed to create directories for '" << path
        << "': " << ec.message();
    Fail(oss.str());
  }

  std::ofstream file(path, std::ios::binary);
  if (!file.is_open()) {
    std::ostringstream oss;
    oss << "Failed to open output file '" << path << "'";
    Fail(oss.str());
  }
  file << content;
  if (!file.good()) {
    std::ostringstream oss;
    oss << "Failed to write output file '" << path << "'";
    Fail(oss.str());
  }
}

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "Usage: gen_architectures <architectures.yml> "
                 "<executions.yml> <output_dir>\n";
    return 1;
  }

  const fs::path architecture_yaml = argv[1];
  const fs::path execution_yaml = argv[2];
  const fs::path output_dir = argv[3];

  try {
    const auto executions = ParseExecutionConfig(execution_yaml);
    std::unordered_set<std::string> execution_ids;
    execution_ids.reserve(executions.size());
    for (const auto &execution : executions) {
      execution_ids.insert(execution.id);
    }

    const auto architectures =
        ParseArchitectureConfig(architecture_yaml, execution_ids);
    const auto generated = GenerateOutputs(executions, architectures);

    WriteFile(output_dir / "architecture.def", generated.architecture_def);
    WriteFile(output_dir / "architecture_tables.h",
              generated.architecture_tables_header);
  } catch (const std::exception &e) {
    std::cerr << "[gen_architectures] Error: " << e.what() << '\n';
    return 1;
  }

  return 0;
}
