#include "orteaf/internal/runtime/strong_id.h"

#include <gtest/gtest.h>

namespace runtime = orteaf::internal::runtime;

TEST(StrongId, BasicComparisonAndConversion) {
    runtime::StreamId stream1{3};
    runtime::StreamId stream2{3};
    runtime::StreamId stream3{4};

    EXPECT_EQ(stream1, stream2);
    EXPECT_NE(stream1, stream3);
    EXPECT_EQ(static_cast<uint8_t>(stream1), 3u);
    EXPECT_LT(stream1, stream3);
    EXPECT_TRUE(stream1.isValid());
}

TEST(StrongId, InvalidHelper) {
    constexpr auto bad = runtime::ContextId::invalid();
    EXPECT_FALSE(bad.isValid());
    EXPECT_EQ(static_cast<uint8_t>(bad), static_cast<uint8_t>(~0));
}

TEST(StrongId, DeviceTypeIsIndependent) {
    runtime::DeviceId device{0};
    runtime::StreamId stream{0};
    // Ensure there is no implicit conversion between different StrongId tags.
    static_assert(!std::is_convertible_v<runtime::StreamId, runtime::DeviceId>);
    static_assert(!std::is_convertible_v<runtime::DeviceId, runtime::StreamId>);
    EXPECT_EQ(static_cast<uint32_t>(device), 0u);
    EXPECT_EQ(static_cast<uint8_t>(stream), 0u);
}
