#include "orteaf/internal/base/handle.h"

#include <gtest/gtest.h>

namespace base = orteaf::internal::base;

namespace {
struct StreamTag {};
struct ContextTag {};
struct DeviceTag {};

using StreamHandle = base::Handle<StreamTag, uint32_t, uint8_t>;
using ContextHandle = base::Handle<ContextTag, uint32_t, uint8_t>;
using DeviceHandle = base::Handle<DeviceTag, uint32_t, void>;
} // namespace

TEST(Handle, BasicComparisonAndConversion) {
    StreamHandle stream1{3};
    StreamHandle stream2{3};
    StreamHandle stream3{4};

    EXPECT_EQ(stream1, stream2);
    EXPECT_NE(stream1, stream3);
    EXPECT_EQ(static_cast<uint32_t>(stream1), 3u); // Handle casts to index type (uint32_t)
    EXPECT_LT(stream1, stream3);
    EXPECT_TRUE(stream1.isValid());
}

TEST(Handle, InvalidHelper) {
    constexpr auto bad = ContextHandle::invalid();
    EXPECT_FALSE(bad.isValid());
    EXPECT_EQ(static_cast<uint32_t>(bad), ContextHandle::invalid_index());
}

TEST(Handle, DeviceTypeIsIndependent) {
    DeviceHandle device{0};
    StreamHandle stream{0};
    // Ensure there is no implicit conversion between different Handle tags.
    static_assert(!std::is_convertible_v<StreamHandle, DeviceHandle>);
    static_assert(!std::is_convertible_v<DeviceHandle, StreamHandle>);
    EXPECT_EQ(static_cast<uint32_t>(device), 0u);
    EXPECT_EQ(static_cast<uint32_t>(stream), 0u);
}
