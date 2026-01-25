#include "orteaf/internal/base/lru_list.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <type_traits>
#include <vector>

namespace base = orteaf::internal::base;

namespace {

static_assert(!std::is_move_constructible_v<base::LruList<int>>);
static_assert(!std::is_move_assignable_v<base::LruList<int>>);

// ============================================================
// LruNode tests
// ============================================================

TEST(LruNodeTest, DefaultConstruction) {
  base::LruNode<int> node;
  EXPECT_EQ(node.key, 0);
  EXPECT_EQ(node.prev, nullptr);
  EXPECT_EQ(node.next, nullptr);
}

TEST(LruNodeTest, ConstructionWithKey) {
  base::LruNode<int> node(42);
  EXPECT_EQ(node.key, 42);
  EXPECT_EQ(node.prev, nullptr);
  EXPECT_EQ(node.next, nullptr);
}

// ============================================================
// LruList basic tests
// ============================================================

TEST(LruListTest, DefaultConstruction) {
  base::LruList<int> list;
  EXPECT_TRUE(list.empty());
  EXPECT_EQ(list.size(), 0u);
  EXPECT_EQ(list.front(), nullptr);
  EXPECT_EQ(list.back(), nullptr);
}

TEST(LruListTest, PushFrontSingleNode) {
  base::LruList<int> list;
  base::LruNode<int> node(1);

  list.pushFront(&node);

  EXPECT_FALSE(list.empty());
  EXPECT_EQ(list.size(), 1u);
  EXPECT_EQ(list.front(), &node);
  EXPECT_EQ(list.back(), &node);
}

TEST(LruListTest, PushFrontMultipleNodes) {
  base::LruList<int> list;
  base::LruNode<int> node1(1);
  base::LruNode<int> node2(2);
  base::LruNode<int> node3(3);

  list.pushFront(&node1);
  list.pushFront(&node2);
  list.pushFront(&node3);

  EXPECT_EQ(list.size(), 3u);
  // Most recently added is at front
  EXPECT_EQ(list.front(), &node3);
  EXPECT_EQ(list.front()->key, 3);
  // Least recently added is at back
  EXPECT_EQ(list.back(), &node1);
  EXPECT_EQ(list.back()->key, 1);
}

// ============================================================
// LruList popBack tests
// ============================================================

TEST(LruListTest, PopBackEmpty) {
  base::LruList<int> list;
  EXPECT_EQ(list.popBack(), nullptr);
}

TEST(LruListTest, PopBackSingleNode) {
  base::LruList<int> list;
  base::LruNode<int> node(1);

  list.pushFront(&node);
  auto *popped = list.popBack();

  EXPECT_EQ(popped, &node);
  EXPECT_TRUE(list.empty());
  EXPECT_EQ(list.size(), 0u);
}

TEST(LruListTest, PopBackMultipleNodes) {
  base::LruList<int> list;
  base::LruNode<int> node1(1);
  base::LruNode<int> node2(2);
  base::LruNode<int> node3(3);

  list.pushFront(&node1);
  list.pushFront(&node2);
  list.pushFront(&node3);

  // Pop should return LRU (oldest = node1)
  auto *popped1 = list.popBack();
  EXPECT_EQ(popped1->key, 1);
  EXPECT_EQ(list.size(), 2u);

  auto *popped2 = list.popBack();
  EXPECT_EQ(popped2->key, 2);
  EXPECT_EQ(list.size(), 1u);

  auto *popped3 = list.popBack();
  EXPECT_EQ(popped3->key, 3);
  EXPECT_TRUE(list.empty());
}

// ============================================================
// LruList touch tests
// ============================================================

TEST(LruListTest, TouchMovesToFront) {
  base::LruList<int> list;
  base::LruNode<int> node1(1);
  base::LruNode<int> node2(2);
  base::LruNode<int> node3(3);

  list.pushFront(&node1);
  list.pushFront(&node2);
  list.pushFront(&node3);

  // Order: 3 -> 2 -> 1 (front to back)
  EXPECT_EQ(list.front()->key, 3);
  EXPECT_EQ(list.back()->key, 1);

  // Touch node1 (LRU) - should move to front
  list.touch(&node1);

  EXPECT_EQ(list.front()->key, 1);
  EXPECT_EQ(list.back()->key, 2);
  EXPECT_EQ(list.size(), 3u);
}

TEST(LruListTest, TouchAlreadyAtFront) {
  base::LruList<int> list;
  base::LruNode<int> node1(1);
  base::LruNode<int> node2(2);

  list.pushFront(&node1);
  list.pushFront(&node2);

  // Touch front node - should be no-op
  list.touch(&node2);

  EXPECT_EQ(list.front()->key, 2);
  EXPECT_EQ(list.back()->key, 1);
  EXPECT_EQ(list.size(), 2u);
}

TEST(LruListTest, TouchMiddleNode) {
  base::LruList<int> list;
  base::LruNode<int> node1(1);
  base::LruNode<int> node2(2);
  base::LruNode<int> node3(3);

  list.pushFront(&node1);
  list.pushFront(&node2);
  list.pushFront(&node3);

  // Order: 3 -> 2 -> 1
  // Touch middle node (2)
  list.touch(&node2);

  // New order: 2 -> 3 -> 1
  EXPECT_EQ(list.front()->key, 2);
  EXPECT_EQ(list.back()->key, 1);
}

TEST(LruListTest, TouchNullptr) {
  base::LruList<int> list;
  base::LruNode<int> node(1);
  list.pushFront(&node);

  // Should not crash
  list.touch(nullptr);

  EXPECT_EQ(list.size(), 1u);
  EXPECT_EQ(list.front(), &node);
}

// ============================================================
// LruList remove tests
// ============================================================

TEST(LruListTest, RemoveFront) {
  base::LruList<int> list;
  base::LruNode<int> node1(1);
  base::LruNode<int> node2(2);

  list.pushFront(&node1);
  list.pushFront(&node2);

  list.remove(&node2);

  EXPECT_EQ(list.size(), 1u);
  EXPECT_EQ(list.front(), &node1);
  EXPECT_EQ(list.back(), &node1);
}

TEST(LruListTest, RemoveBack) {
  base::LruList<int> list;
  base::LruNode<int> node1(1);
  base::LruNode<int> node2(2);

  list.pushFront(&node1);
  list.pushFront(&node2);

  list.remove(&node1);

  EXPECT_EQ(list.size(), 1u);
  EXPECT_EQ(list.front(), &node2);
  EXPECT_EQ(list.back(), &node2);
}

TEST(LruListTest, RemoveMiddle) {
  base::LruList<int> list;
  base::LruNode<int> node1(1);
  base::LruNode<int> node2(2);
  base::LruNode<int> node3(3);

  list.pushFront(&node1);
  list.pushFront(&node2);
  list.pushFront(&node3);

  list.remove(&node2);

  EXPECT_EQ(list.size(), 2u);
  EXPECT_EQ(list.front()->key, 3);
  EXPECT_EQ(list.back()->key, 1);
}

// ============================================================
// LruList clear tests
// ============================================================

TEST(LruListTest, Clear) {
  base::LruList<int> list;
  base::LruNode<int> node1(1);
  base::LruNode<int> node2(2);

  list.pushFront(&node1);
  list.pushFront(&node2);

  list.clear();

  EXPECT_TRUE(list.empty());
  EXPECT_EQ(list.size(), 0u);
  EXPECT_EQ(list.front(), nullptr);
  EXPECT_EQ(list.back(), nullptr);
}

// ============================================================
// LruList LRU eviction pattern test
// ============================================================

TEST(LruListTest, LruEvictionPattern) {
  base::LruList<int> list;
  std::vector<base::LruNode<int>> nodes;
  nodes.reserve(5);
  for (int i = 0; i < 5; ++i) {
    nodes.emplace_back(i);
  }

  // Insert all: 0, 1, 2, 3, 4
  for (auto &node : nodes) {
    list.pushFront(&node);
  }
  // Order: 4 -> 3 -> 2 -> 1 -> 0 (front to back)

  // Access pattern: touch 0 and 2
  list.touch(&nodes[0]);
  list.touch(&nodes[2]);
  // New order: 2 -> 0 -> 4 -> 3 -> 1 (front to back)

  // Evict 3 (simulate capacity limit)
  std::vector<int> evicted;
  for (int i = 0; i < 3; ++i) {
    auto *node = list.popBack();
    evicted.push_back(node->key);
  }

  // Should evict in LRU order: 1, 3, 4
  EXPECT_EQ(evicted[0], 1);
  EXPECT_EQ(evicted[1], 3);
  EXPECT_EQ(evicted[2], 4);

  // Remaining: 2 -> 0
  EXPECT_EQ(list.size(), 2u);
  EXPECT_EQ(list.front()->key, 2);
  EXPECT_EQ(list.back()->key, 0);
}

} // namespace
