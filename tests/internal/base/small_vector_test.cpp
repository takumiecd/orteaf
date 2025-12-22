/**
 * @file
 * @brief SmallVectorのインライン/ヒープ遷移と例外安全性を中心に挙動を検証するユニットテスト。
 *
 * - インライン容量内/外での push/pop/assign/resize 動作とイテレータの整合性。
 * - `swap`/ムーブ代入によるストレージ移転、ストレージ復帰（assignで再びinlineに戻るケース）を検証。
 * - 例外を投げる要素型での emplace/resize を通じ、強い例外保証とライブインスタンス管理が機能するかを確認。
 */
#include "orteaf/internal/base/small_vector.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace orteaf::internal::base {
namespace {

struct CountingPayload {
    static inline int live_instances = 0;
    static inline int moves = 0;
    static inline int copies = 0;

    int value = 0;

    CountingPayload() : value(0) { ++live_instances; }
    explicit CountingPayload(int v) : value(v) { ++live_instances; }
    CountingPayload(const CountingPayload& other) : value(other.value) {
        ++live_instances;
        ++copies;
    }
    CountingPayload(CountingPayload&& other) noexcept : value(other.value) {
        ++live_instances;
        ++moves;
    }
    ~CountingPayload() { --live_instances; }

    CountingPayload& operator=(const CountingPayload&) = default;
    CountingPayload& operator=(CountingPayload&&) = default;

    static void Reset() {
        live_instances = 0;
        moves = 0;
        copies = 0;
    }
};

struct ThrowingPayload {
    static inline int throws_before = 0;
    static inline int live_instances = 0;

    ThrowingPayload() {
        if (throws_before == 0) {
            throw std::runtime_error("ctor failure");
        }
        --throws_before;
        ++live_instances;
    }
    ThrowingPayload(const ThrowingPayload&) {
        if (throws_before == 0) {
            throw std::runtime_error("copy failure");
        }
        --throws_before;
        ++live_instances;
    }
    ~ThrowingPayload() { --live_instances; }
};

}  // namespace

/** @test SmallVectorTest.DefaultStateUsesInlineBuffer
 *  @brief Ensures inline-vs-heap transitions behave correctly relative to capacity.
 */
TEST(SmallVectorTest, DefaultStateUsesInlineBuffer) {
    SmallVector<int, 2> vec;
    EXPECT_TRUE(vec.empty());
    EXPECT_EQ(vec.size(), 0u);
    EXPECT_EQ(vec.capacity(), 2u);

    int* inline_ptr = vec.data();
    vec.pushBack(1);
    vec.pushBack(2);
    EXPECT_EQ(vec.data(), inline_ptr) << "within inline capacity";

    vec.pushBack(3);
    EXPECT_NE(vec.data(), inline_ptr) << "should have moved to heap storage";
}

/** @test SmallVectorTest.PushPopAndIteratorsWork
 *  @brief Confirms push/emplace/pop and range iteration produce expected sequence.
 */
TEST(SmallVectorTest, PushPopAndIteratorsWork) {
    SmallVector<int, 3> vec;
    vec.pushBack(5);
    vec.emplaceBack(6);
    vec.pushBack(7);
    EXPECT_EQ(vec.front(), 5);
    EXPECT_EQ(vec.back(), 7);

    int sum = 0;
    for (int value : vec) sum += value;
    EXPECT_EQ(sum, 18);

    vec.popBack();
    EXPECT_EQ(vec.size(), 2u);
    EXPECT_EQ(vec.back(), 6);
}

/** @test SmallVectorTest.AssignAndReserveManageStorageTransitions
 *  @brief Validates assign shifts between heap and inline storage depending on count.
 */
TEST(SmallVectorTest, AssignAndReserveManageStorageTransitions) {
    SmallVector<int, 2> vec;
    int* inline_ptr = vec.data();
    using Vec = SmallVector<int, 2>;
    vec.assign(Vec::size_type{5}, 9);
    EXPECT_EQ(vec.size(), 5u);
    EXPECT_NE(vec.data(), inline_ptr);

    vec.assign(Vec::size_type{1}, 42);
    EXPECT_EQ(vec.size(), 1u);
    EXPECT_EQ(vec.data(), inline_ptr) << "assign with small count returns to inline storage";
}

/** @test SmallVectorTest.ResizeAndClearAdjustSize
 *  @brief Verifies resize and clear adjust size while preserving expected defaults.
 */
TEST(SmallVectorTest, ResizeAndClearAdjustSize) {
    SmallVector<int, 2> vec;
    vec.resize(3, 8);
    EXPECT_EQ(vec.size(), 3u);
    EXPECT_EQ(vec[2], 8);

    vec.resize(1);
    EXPECT_EQ(vec.size(), 1u);

    vec.clear();
    EXPECT_TRUE(vec.empty());
}

/** @test SmallVectorTest.AtThrowsOnOutOfRange
 *  @brief Confirms at() throws when index is out of bounds.
 */
TEST(SmallVectorTest, AtThrowsOnOutOfRange) {
    SmallVector<int, 2> vec;
    vec.pushBack(1);
    EXPECT_THROW(vec.at(1), std::out_of_range);
    EXPECT_EQ(vec.at(0), 1);
}

/** @test SmallVectorTest.ReserveAndShrinkToFit
 *  @brief Validates reserve grows capacity and shrinkToFit releases unused space.
 */
TEST(SmallVectorTest, ReserveAndShrinkToFit) {
    SmallVector<int, 2> vec;
    const int* inline_ptr = vec.data();
    vec.reserve(8);
    EXPECT_GE(vec.capacity(), 8u);
    vec.resize(3, 7);
    EXPECT_NE(vec.data(), inline_ptr);
    vec.shrinkToFit();
    EXPECT_EQ(vec.size(), 3u);
    EXPECT_LE(vec.capacity(), 3u);
}

/** @test SmallVectorTest.InsertEraseEmplace
 *  @brief Exercises insert/emplace/erase operations and ordering.
 */
TEST(SmallVectorTest, InsertEraseEmplace) {
    SmallVector<int, 2> vec;
    vec.pushBack(1);
    vec.pushBack(3);

    vec.insert(vec.begin() + 1, 2);
    ASSERT_EQ(vec.size(), 3u);
    EXPECT_EQ(vec[0], 1);
    EXPECT_EQ(vec[1], 2);
    EXPECT_EQ(vec[2], 3);

    vec.emplace(vec.begin() + 3, 4);
    EXPECT_EQ(vec.back(), 4);

    vec.erase(vec.begin() + 1);
    ASSERT_EQ(vec.size(), 3u);
    EXPECT_EQ(vec[0], 1);
    EXPECT_EQ(vec[1], 3);
    EXPECT_EQ(vec[2], 4);

    vec.insert(vec.begin(), 2, 9);
    ASSERT_EQ(vec.size(), 5u);
    EXPECT_EQ(vec[0], 9);
    EXPECT_EQ(vec[1], 9);

    vec.erase(vec.begin(), vec.begin() + 2);
    ASSERT_EQ(vec.size(), 3u);
    EXPECT_EQ(vec[0], 1);
}

/** @test SmallVectorTest.RangeInsertMaintainsOrder
 *  @brief Ensures range insert preserves element order.
 */
TEST(SmallVectorTest, RangeInsertMaintainsOrder) {
    SmallVector<int, 2> vec;
    vec.pushBack(1);
    vec.pushBack(4);
    std::vector<int> values{2, 3};

    vec.insert(vec.begin() + 1, values.begin(), values.end());
    ASSERT_EQ(vec.size(), 4u);
    EXPECT_EQ(vec[0], 1);
    EXPECT_EQ(vec[1], 2);
    EXPECT_EQ(vec[2], 3);
    EXPECT_EQ(vec[3], 4);
}

/** @test SmallVectorTest.ConstIteration
 *  @brief Verifies const iteration covers all elements.
 */
TEST(SmallVectorTest, ConstIteration) {
    SmallVector<int, 2> vec;
    vec.pushBack(2);
    vec.pushBack(4);
    const SmallVector<int, 2>& cvec = vec;
    int sum = 0;
    for (auto it = cvec.cbegin(); it != cvec.cend(); ++it) {
        sum += *it;
    }
    EXPECT_EQ(sum, 6);
}

/** @test SmallVectorTest.SwapAndMovePreserveElements
 *  @brief Ensures swap and move operations relocate elements without loss.
 */
TEST(SmallVectorTest, SwapAndMovePreserveElements) {
    CountingPayload::Reset();
    SmallVector<CountingPayload, 2> a;
    a.emplaceBack(CountingPayload{1});
    SmallVector<CountingPayload, 2> b;
    b.emplaceBack(CountingPayload{2});
    b.emplaceBack(CountingPayload{3});

    swap(a, b);
    EXPECT_EQ(a.size(), 2u);
    EXPECT_EQ(b.size(), 1u);

    SmallVector<CountingPayload, 2> moved(std::move(a));
    EXPECT_EQ(moved.size(), 2u);
    EXPECT_EQ(a.size(), 0u);

    SmallVector<CountingPayload, 2> assigned;
    assigned = std::move(b);
    EXPECT_EQ(assigned.size(), 1u);
    EXPECT_EQ(b.size(), 0u);
}

/** @test SmallVectorTest.ExceptionSafetyDuringInsertion
 *  @brief Confirms emplaceBack exceptions keep the vector unchanged.
 */
TEST(SmallVectorTest, ExceptionSafetyDuringInsertion) {
    SmallVector<ThrowingPayload, 1> vec;
    ThrowingPayload::throws_before = 0;
    EXPECT_THROW(vec.emplaceBack(), std::runtime_error);
    EXPECT_EQ(vec.size(), 0u);
    EXPECT_EQ(ThrowingPayload::live_instances, 0);
}

/** @test SmallVectorTest.ExceptionSafetyDuringResize
 *  @brief Verifies resize exceptions rollback both size and live instance counts.
 */
TEST(SmallVectorTest, ExceptionSafetyDuringResize) {
    SmallVector<ThrowingPayload, 1> vec;
    ThrowingPayload::throws_before = 1;
    EXPECT_THROW(vec.resize(2), std::runtime_error);
    EXPECT_EQ(vec.size(), 0u);
    EXPECT_EQ(ThrowingPayload::live_instances, 0);
}

}  // namespace orteaf::internal::base
