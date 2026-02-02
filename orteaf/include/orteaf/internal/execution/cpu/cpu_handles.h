#pragma once

#include <orteaf/internal/base/handle.h>

namespace orteaf::internal::execution::cpu {

struct CpuDeviceTag {};
struct CpuStreamTag {};
struct CpuContextTag {};
struct CpuBufferTag {};
struct CpuEventTag {};
struct CpuFenceTag {};
struct CpuKernelBaseTag {};
struct CpuKernelMetadataTag {};

using CpuDeviceHandle =
    ::orteaf::internal::base::Handle<CpuDeviceTag, uint32_t, void>;
using CpuStreamHandle =
    ::orteaf::internal::base::Handle<CpuStreamTag, uint32_t, uint8_t>;
using CpuContextHandle =
    ::orteaf::internal::base::Handle<CpuContextTag, uint32_t, uint8_t>;
using CpuBufferHandle =
    ::orteaf::internal::base::Handle<CpuBufferTag, uint32_t, uint32_t>;
using CpuBufferViewHandle =
    ::orteaf::internal::base::Handle<CpuBufferTag, uint32_t, void>;
using CpuEventHandle =
    ::orteaf::internal::base::Handle<CpuEventTag, uint32_t, uint8_t>;
using CpuFenceHandle =
    ::orteaf::internal::base::Handle<CpuFenceTag, uint32_t, uint8_t>;
using CpuKernelBaseHandle =
    ::orteaf::internal::base::Handle<CpuKernelBaseTag, uint32_t, void>;
using CpuKernelMetadataHandle =
    ::orteaf::internal::base::Handle<CpuKernelMetadataTag, uint32_t, void>;

static_assert(std::is_trivially_copyable_v<CpuDeviceHandle>);
static_assert(std::is_trivially_copyable_v<CpuBufferHandle>);
static_assert(std::is_trivially_copyable_v<CpuBufferViewHandle>);
static_assert(std::is_trivially_copyable_v<CpuEventHandle>);
static_assert(std::is_trivially_copyable_v<CpuFenceHandle>);
static_assert(std::is_trivially_copyable_v<CpuKernelBaseHandle>);
static_assert(std::is_trivially_copyable_v<CpuKernelMetadataHandle>);

} // namespace orteaf::internal::execution::cpu
