#pragma once

#include "orteaf/internal/architecture/architecture.h"

#include <cstdint>
#include <string_view>

namespace orteaf::internal::architecture {

/**
 * @brief Select the CUDA architecture that matches the provided compute capability.
 *
 * The generated architecture metadata keeps compute capability hints per entry, so the
 * detector walks each CUDA backend row and returns the first match that shares the requested
 * capability and (optionally) vendor. Undefined capabilities (negative values) skip the check,
 * allowing the generic architecture to be returned instead.
 *
 * @param compute_capability Numeric compute capability expressed as `major*10 + minor` (e.g., 80 for SM80).
 * @param vendor_hint Optional vendor hint, defaults to `"nvidia"`, that filters tables with stricter matches.
 * @return The matching CUDA `Architecture`, or `Architecture::cuda_generic` when no table entry matches.
 */
Architecture detectCudaArchitecture(int compute_capability, std::string_view vendor_hint = "nvidia");

/**
 * @brief Enumerate CUDA devices and detect the architecture for the selected index.
 *
 * When CUDA is enabled, the helper queries `backend::cuda` for the device count, extracts
 * the compute capability and vendor for the requested device, and then delegates to
 * `detectCudaArchitecture`. Out-of-range indices or missing CUDA support result in
 * `Architecture::cuda_generic`.
 *
 * @param device_index Zero-based CUDA device index.
 * @return The detected CUDA `Architecture`, or `Architecture::cuda_generic` when enumeration fails.
 */
Architecture detectCudaArchitectureForDeviceIndex(std::uint32_t device_index);

} // namespace orteaf::internal::architecture
