#pragma once

#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

namespace orteaf::codegen {

inline void InsertIncludeIfTypeMatches(std::unordered_set<std::string> &includes,
                                       const std::string &type,
                                       const char *needle,
                                       const char *header) {
  if (type.find(needle) != std::string::npos) {
    includes.insert(header);
  }
}

inline std::vector<std::string>
CollectRequiredIncludes(const std::vector<std::string> &types) {
  std::unordered_set<std::string> includes;
  includes.reserve(types.size());

  for (const auto &type : types) {
    InsertIncludeIfTypeMatches(includes, type, "ArrayView<",
                               "orteaf/internal/base/array_view.h");
    InsertIncludeIfTypeMatches(includes, type, "SmallVector<",
                               "orteaf/internal/base/small_vector.h");
    InsertIncludeIfTypeMatches(includes, type, "InlineVector<",
                               "orteaf/internal/base/inline_vector.h");
    InsertIncludeIfTypeMatches(includes, type, "std::array", "array");
    InsertIncludeIfTypeMatches(includes, type, "std::optional", "optional");
    InsertIncludeIfTypeMatches(includes, type, "std::span", "span");
    InsertIncludeIfTypeMatches(includes, type, "std::string",
                               "string");
    InsertIncludeIfTypeMatches(includes, type, "std::vector",
                               "vector");
  }

  std::vector<std::string> sorted(includes.begin(), includes.end());
  std::sort(sorted.begin(), sorted.end());
  return sorted;
}

} // namespace orteaf::codegen
