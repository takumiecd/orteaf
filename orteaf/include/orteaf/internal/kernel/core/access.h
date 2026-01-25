#pragma once

#include <cstdint>

namespace orteaf::internal::kernel {

/**
 * @brief Access mode for kernel buffer bindings.
 *
 * Specifies how a kernel will access a buffer during execution.
 */
enum class Access : uint8_t {
  None = 0,      ///< Unused slot / invalid
  Read = 1,      ///< Read-only access
  Write = 2,     ///< Write-only access
  ReadWrite = 3, ///< Read and write access
};

} // namespace orteaf::internal::kernel
