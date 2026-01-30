#pragma once

#include <orteaf/internal/base/handle.h>

namespace orteaf::internal::execution::mps {

struct MpsDeviceTag {};
struct MpsCommandQueueTag {};
struct MpsLibraryTag {};
struct MpsFunctionTag {};
struct MpsHeapTag {};
struct MpsBufferTag {};
struct MpsGraphTag {};
struct MpsEventTag {};
struct MpsFenceTag {};
struct MpsKernelBaseTag {};
struct MpsKernelMetadataTag {};

using MpsDeviceHandle = ::orteaf::internal::base::Handle<MpsDeviceTag, uint32_t, void>;
using MpsCommandQueueHandle =
    ::orteaf::internal::base::Handle<MpsCommandQueueTag, uint32_t, void>;
using MpsLibraryHandle = ::orteaf::internal::base::Handle<MpsLibraryTag, uint32_t, void>;
using MpsFunctionHandle = ::orteaf::internal::base::Handle<MpsFunctionTag, uint32_t, void>;
using MpsHeapHandle = ::orteaf::internal::base::Handle<MpsHeapTag, uint32_t, void>;
using MpsBufferHandle = ::orteaf::internal::base::Handle<MpsBufferTag, uint32_t, uint32_t>;
using MpsBufferViewHandle = ::orteaf::internal::base::Handle<MpsBufferTag, uint32_t, void>;
using MpsGraphHandle = ::orteaf::internal::base::Handle<MpsGraphTag, uint32_t, uint8_t>;
using MpsEventHandle = ::orteaf::internal::base::Handle<MpsEventTag, uint32_t, uint8_t>;
using MpsFenceHandle = ::orteaf::internal::base::Handle<MpsFenceTag, uint32_t, uint8_t>;
using MpsKernelBaseHandle = ::orteaf::internal::base::Handle<MpsKernelBaseTag, uint32_t, void>;
using MpsKernelMetadataHandle =
    ::orteaf::internal::base::Handle<MpsKernelMetadataTag, uint32_t, void>;

static_assert(std::is_trivially_copyable_v<MpsDeviceHandle>);
static_assert(std::is_trivially_copyable_v<MpsBufferHandle>);
static_assert(std::is_trivially_copyable_v<MpsBufferViewHandle>);
static_assert(std::is_trivially_copyable_v<MpsEventHandle>);
static_assert(std::is_trivially_copyable_v<MpsFenceHandle>);

} // namespace orteaf::internal::execution::mps
