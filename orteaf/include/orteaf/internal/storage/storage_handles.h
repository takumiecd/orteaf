#pragma once

#include <cstdint>
#include <type_traits>

#include <orteaf/internal/base/handle.h>

namespace orteaf::internal::storage {

struct StorageTag {};

using StorageHandle = ::orteaf::internal::base::Handle<StorageTag, uint32_t, uint32_t>;

static_assert(std::is_trivially_copyable_v<StorageHandle>);

} // namespace orteaf::internal::storage
