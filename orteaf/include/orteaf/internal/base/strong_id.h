#pragma once

#include <cstdint>
#include <compare>
#include <type_traits>

namespace orteaf::internal::base {

template <class Tag, class T = uint8_t>
struct StrongId {
    using underlying_type = T;

    T value{};

    constexpr StrongId() = default;
    explicit constexpr StrongId(T v) noexcept : value(v) {}

    constexpr auto operator<=>(const StrongId&) const = default;
    explicit constexpr operator T() const noexcept { return value; }

    static constexpr StrongId invalid() noexcept { return StrongId{static_cast<T>(~T{})}; }
    constexpr bool isValid() const noexcept { return value != static_cast<T>(~T{}); }
};

struct DeviceTag {};
struct StreamTag {};
struct ContextTag {};
struct CommandQueueTag {};
struct LibraryTag {};
struct FunctionTag {};
struct HeapTag {};
struct BufferTag {};

using DeviceId  = StrongId<DeviceTag, uint32_t>;
using StreamId  = StrongId<StreamTag, uint8_t>;
using ContextId = StrongId<ContextTag, uint8_t>;
using CommandQueueId = StrongId<CommandQueueTag, uint32_t>;
using LibraryId = StrongId<LibraryTag, uint32_t>;
using FunctionId = StrongId<FunctionTag, uint32_t>;
using HeapId = StrongId<HeapTag, uint32_t>;
using BufferId = StrongId<BufferTag, uint32_t>;

static_assert(sizeof(DeviceId) == sizeof(uint32_t));
static_assert(sizeof(StreamId) == sizeof(uint8_t));
static_assert(sizeof(ContextId) == sizeof(uint8_t));
static_assert(sizeof(CommandQueueId) == sizeof(uint32_t));
static_assert(sizeof(LibraryId) == sizeof(uint32_t));
static_assert(sizeof(FunctionId) == sizeof(uint32_t));
static_assert(sizeof(HeapId) == sizeof(uint32_t));
static_assert(sizeof(BufferId) == sizeof(uint32_t));
static_assert(std::is_trivially_copyable_v<DeviceId>);

} // namespace orteaf::internal::base
