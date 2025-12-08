#pragma once

namespace orteaf::internal::runtime::cpu::resource {

// CPU tokens are no-ops; they exist so the allocator interface can stay consistent.
struct FenceToken {};
struct ReuseToken {};

}  // namespace orteaf::internal::runtime::cpu::resource