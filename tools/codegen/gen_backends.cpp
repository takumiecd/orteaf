#include <yaml-cpp/yaml.h>

#include <cctype>
#include <initializer_list>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <optional>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

[[noreturn]] void Fail(const std::string& message) {
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

void ExpectKeys(const YAML::Node& node, std::string_view context,
                std::initializer_list<std::string_view> allowed) {
    std::unordered_set<std::string_view> allowed_set(allowed.begin(), allowed.end());
    for (const auto& kv : node) {
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

std::string ReadRequiredString(const YAML::Node& node, std::string_view key, std::string_view context) {
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

std::optional<std::string> ReadOptionalString(const YAML::Node& node, std::string_view key, std::string_view context) {
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
    if (!(std::isalpha(static_cast<unsigned char>(value.front())) || value.front() == '_')) {
        return false;
    }
    for (char ch : value) {
        if (!(std::isalnum(static_cast<unsigned char>(ch)) || ch == '_')) {
            return false;
        }
    }
    return true;
}

struct BackendDefinition {
    std::string id;
    std::string display_name;
    std::string module_path;
    std::string description;
};

struct ParsedConfig {
    std::string schema_version;
    std::vector<BackendDefinition> backends;
};

ParsedConfig ParseConfig(const fs::path& yaml_path) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(yaml_path.string());
    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "Failed to load YAML '" << yaml_path << "': " << e.what();
        Fail(oss.str());
    }

    if (!root || !root.IsMap()) {
        Fail("Root of backend YAML must be a mapping");
    }

    const auto schema_node = root["schema_version"];
    if (!schema_node || !schema_node.IsScalar()) {
        Fail("Missing required scalar key 'schema_version'");
    }

    ParsedConfig parsed;
    parsed.schema_version = schema_node.as<std::string>();

    const auto backends_node = root["backends"];
    if (!backends_node || !backends_node.IsSequence()) {
        Fail("Missing required sequence key 'backends'");
    }

    parsed.backends.reserve(backends_node.size());
    std::unordered_set<std::string> seen_ids;

    for (std::size_t index = 0; index < backends_node.size(); ++index) {
        const auto& node = backends_node[index];
        if (!node.IsMap()) {
            std::ostringstream oss;
            oss << "Each backend entry must be a mapping (index " << index << ")";
            Fail(oss.str());
        }
        const std::string context = "backends[" + std::to_string(index) + "]";

        BackendDefinition backend;
        backend.id = ReadRequiredString(node, "id", context);
        if (!LooksLikeIdentifier(backend.id)) {
            std::ostringstream oss;
            oss << "Backend id '" << backend.id << "' is not a valid identifier (" << context << ")";
            Fail(oss.str());
        }
        if (!seen_ids.insert(backend.id).second) {
            std::ostringstream oss;
            oss << "Duplicate backend id '" << backend.id << "'";
            Fail(oss.str());
        }

        backend.display_name = ReadRequiredString(node, "display_name", context);
        backend.module_path = ReadRequiredString(node, "module_path", context);

        const auto metadata_node = node["metadata"];
        if (metadata_node) {
            if (!metadata_node.IsMap()) {
                std::ostringstream oss;
                oss << "Metadata for " << context << " must be a mapping";
                Fail(oss.str());
            }
            const std::string metadata_context = context + " metadata";
            ExpectKeys(metadata_node, metadata_context, {"description"});
            const auto description =
                ReadOptionalString(metadata_node, "description", metadata_context);
            if (description) {
                backend.description = *description;
            }
        }

        parsed.backends.push_back(std::move(backend));
    }

    return parsed;
}

struct GeneratedData {
    std::string backend_def;
    std::string backend_tables_header;
};

GeneratedData GenerateOutputs(const ParsedConfig& config) {
    GeneratedData generated;

    {
        std::ostringstream def_stream;
        def_stream << "// Auto-generated. Do not edit.\n";
        for (const auto& backend : config.backends) {
            def_stream << "BACKEND(" << backend.id << ", \""
                       << EscapeStringLiteral(backend.display_name) << "\", \""
                       << EscapeStringLiteral(backend.module_path) << "\", \""
                       << EscapeStringLiteral(backend.description) << "\")\n";
        }
        generated.backend_def = def_stream.str();
    }

    {
        std::ostringstream header_stream;
        header_stream << "// Auto-generated. Do not edit.\n";
        header_stream << "#pragma once\n\n";
        header_stream << "#include <array>\n";
        header_stream << "#include <cstddef>\n";
        header_stream << "#include <string_view>\n\n";
        header_stream << "namespace orteaf::generated::execution_tables {\n";
        header_stream << "inline constexpr std::size_t kBackendCount = " << config.backends.size() << ";\n\n";

        auto emit_string_array = [&](const std::string_view name, auto getter) {
            header_stream << "inline constexpr std::array<std::string_view, kBackendCount> " << name << " = {\n";
            for (const auto& backend : config.backends) {
                header_stream << "    \"" << getter(backend) << "\",\n";
            }
            header_stream << "};\n\n";
        };

        emit_string_array("kBackendDisplayNames",
                          [](const BackendDefinition& backend) { return EscapeStringLiteral(backend.display_name); });
        emit_string_array("kBackendModulePaths",
                          [](const BackendDefinition& backend) { return EscapeStringLiteral(backend.module_path); });
        emit_string_array("kBackendDescriptions",
                          [](const BackendDefinition& backend) { return EscapeStringLiteral(backend.description); });

        header_stream << "}  // namespace orteaf::generated::execution_tables\n";
        generated.backend_tables_header = header_stream.str();
    }

    return generated;
}

void WriteFile(const fs::path& path, const std::string& content) {
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

int main(int argc, char** argv) try {
    if (argc != 3) {
        std::cerr << "Usage: gen_backends <backends.yml> <output_dir>\n";
        return 1;
    }

    const fs::path input_path = argv[1];
    const fs::path output_dir = argv[2];
    if (!fs::exists(input_path)) {
        std::cerr << "Input file does not exist: " << input_path << "\n";
        return 1;
    }

    const ParsedConfig parsed = ParseConfig(input_path);
    const GeneratedData generated = GenerateOutputs(parsed);

    std::error_code ec;
    fs::create_directories(output_dir, ec);
    if (ec) {
        std::cerr << "Failed to create output directory '" << output_dir << "': " << ec.message() << "\n";
        return 1;
    }

    WriteFile(output_dir / "execution.def", generated.backend_def);
    WriteFile(output_dir / "execution_tables.h", generated.backend_tables_header);

    return 0;
} catch (const std::exception& e) {
    std::cerr << "gen_backends error: " << e.what() << "\n";
    return 1;
}
