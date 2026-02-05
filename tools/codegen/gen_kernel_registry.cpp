#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct KernelEntry {
  std::string symbol;
  std::string guard;

  bool operator<(const KernelEntry &other) const {
    if (guard != other.guard) {
      return guard < other.guard;
    }
    return symbol < other.symbol;
  }
};

bool hasKernelExtension(const fs::path &path) {
  const auto ext = path.extension().string();
  return ext == ".cpp" || ext == ".cc" || ext == ".cxx" || ext == ".mm" ||
         ext == ".cu";
}

std::string trim(std::string_view value) {
  std::size_t begin = 0;
  while (begin < value.size() &&
         std::isspace(static_cast<unsigned char>(value[begin])) != 0) {
    ++begin;
  }
  std::size_t end = value.size();
  while (end > begin &&
         std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
    --end;
  }
  return std::string(value.substr(begin, end - begin));
}

std::string detectGuard(const fs::path &path) {
  for (const auto &part : path) {
    const auto name = part.string();
    if (name == "cpu") {
      return "ORTEAF_ENABLE_CPU";
    }
    if (name == "cuda") {
      return "ORTEAF_ENABLE_CUDA";
    }
    if (name == "mps") {
      return "ORTEAF_ENABLE_MPS";
    }
  }
  return {};
}

std::vector<std::string> splitSymbol(std::string_view symbol) {
  std::vector<std::string> parts;
  std::size_t start = 0;
  while (start < symbol.size()) {
    const auto pos = symbol.find("::", start);
    if (pos == std::string::npos) {
      parts.emplace_back(symbol.substr(start));
      break;
    }
    parts.emplace_back(symbol.substr(start, pos - start));
    start = pos + 2;
  }
  return parts;
}

std::string stripLeadingColons(std::string_view symbol) {
  std::size_t offset = 0;
  while (offset + 1 < symbol.size() && symbol[offset] == ':' &&
         symbol[offset + 1] == ':') {
    offset += 2;
  }
  return std::string(symbol.substr(offset));
}

void emitForwardDecl(std::ostream &out, const std::string &symbol) {
  const auto cleaned = stripLeadingColons(symbol);
  const auto parts = splitSymbol(cleaned);
  if (parts.empty()) {
    return;
  }
  if (parts.size() == 1) {
    out << "void " << parts[0] << "();\n";
    return;
  }
  for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
    out << "namespace " << parts[i] << " { ";
  }
  out << "void " << parts.back() << "();";
  for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
    out << " }";
  }
  out << "\n";
}

void extractKernelRegistrars(const std::string &content,
                             const std::string &guard,
                             std::set<KernelEntry> &entries) {
  const std::string marker = "ORTEAF_REGISTER_KERNEL";
  std::size_t pos = 0;
  while (true) {
    pos = content.find(marker, pos);
    if (pos == std::string::npos) {
      break;
    }
    const auto open = content.find('(', pos + marker.size());
    if (open == std::string::npos) {
      break;
    }
    std::size_t depth = 1;
    std::size_t close = open + 1;
    for (; close < content.size(); ++close) {
      if (content[close] == '(') {
        ++depth;
      } else if (content[close] == ')') {
        if (--depth == 0) {
          break;
        }
      }
    }
    if (depth != 0 || close >= content.size()) {
      break;
    }
    auto symbol = trim(std::string_view(content).substr(open + 1,
                                                        close - open - 1));
    if (!symbol.empty() && symbol.back() == ';') {
      symbol.pop_back();
      symbol = trim(symbol);
    }
    if (!symbol.empty()) {
      entries.insert(KernelEntry{symbol, guard});
    }
    pos = close + 1;
  }
}

int run(const fs::path &source_root, const fs::path &output_dir) {
  const fs::path kernel_root =
      source_root / "src" / "extension" / "kernel";
  std::set<KernelEntry> entries;

  if (fs::exists(kernel_root)) {
    for (const auto &entry : fs::recursive_directory_iterator(kernel_root)) {
      if (!entry.is_regular_file()) {
        continue;
      }
      if (!hasKernelExtension(entry.path())) {
        continue;
      }
      std::ifstream input(entry.path());
      if (!input) {
        std::cerr << "Failed to read " << entry.path() << "\n";
        return 1;
      }
      std::string content((std::istreambuf_iterator<char>(input)),
                          std::istreambuf_iterator<char>());
      extractKernelRegistrars(content, detectGuard(entry.path()), entries);
    }
  }

  std::map<std::string, std::vector<std::string>> grouped;
  for (const auto &entry : entries) {
    grouped[entry.guard].push_back(entry.symbol);
  }
  for (auto &group : grouped) {
    auto &symbols = group.second;
    std::sort(symbols.begin(), symbols.end());
    symbols.erase(std::unique(symbols.begin(), symbols.end()), symbols.end());
  }

  fs::create_directories(output_dir);
  const auto output_path = output_dir / "kernel_registry.cpp";
  std::ofstream out(output_path);
  if (!out) {
    std::cerr << "Failed to write " << output_path << "\n";
    return 1;
  }

  out << "// Auto-generated by gen_kernel_registry.cpp. Do not edit.\n";
  out << "#include \"orteaf/internal/kernel/registry/kernel_generated_registry.h\"\n\n";

  for (const auto &group : grouped) {
    if (!group.first.empty()) {
      out << "#if " << group.first << "\n";
    }
    for (const auto &symbol : group.second) {
      emitForwardDecl(out, symbol);
    }
    if (!group.first.empty()) {
      out << "#endif\n";
    }
    if (!group.second.empty()) {
      out << "\n";
    }
  }

  out << "namespace orteaf::internal::kernel::registry {\n\n";
  out << "void registerAllGeneratedKernels() {\n";
  for (const auto &group : grouped) {
    if (!group.first.empty()) {
      out << "#if " << group.first << "\n";
    }
    for (const auto &symbol : group.second) {
      out << "  " << symbol << "();\n";
    }
    if (!group.first.empty()) {
      out << "#endif\n";
    }
  }
  out << "}\n\n";
  out << "} // namespace orteaf::internal::kernel::registry\n";

  return 0;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: gen_kernel_registry <source_root> <output_dir>\n";
    return 1;
  }
  return run(fs::path(argv[1]), fs::path(argv[2]));
}
