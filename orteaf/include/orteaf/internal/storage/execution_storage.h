#pragma once

#include <orteaf/internal/execution/execution.h>

namespace orteaf::internal::storage {
template <execution::Execution B> class ExecutionStorage {
public:
  ExecutionStorage() = default;
};
} // namespace orteaf::internal::storage
