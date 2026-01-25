#pragma once

#include <cstddef>

namespace orteaf::internal::base {

/**
 * @brief Intrusive LRU node for doubly-linked list.
 *
 * Each node contains prev/next pointers and a key for identification.
 * The node does not own the value - it's meant to be embedded or associated
 * with external storage.
 *
 * @tparam Key Key type for node identification
 */
template <typename Key> struct LruNode {
  Key key{};
  LruNode *prev{nullptr};
  LruNode *next{nullptr};

  constexpr LruNode() = default;
  constexpr explicit LruNode(Key k) : key(k) {}

  LruNode(const LruNode &) = delete;
  LruNode &operator=(const LruNode &) = delete;
  LruNode(LruNode &&) = default;
  LruNode &operator=(LruNode &&) = default;
};

/**
 * @brief Intrusive LRU doubly-linked list.
 *
 * Maintains nodes in LRU order with O(1) operations:
 * - touch(): Move node to front (most recently used)
 * - pushFront(): Insert new node at front
 * - popBack(): Remove and return least recently used node
 * - remove(): Remove specific node from list
 *
 * Uses a sentinel node for simplified boundary handling.
 *
 * @tparam Key Key type for node identification
 */
template <typename Key> class LruList {
public:
  using Node = LruNode<Key>;

  constexpr LruList() noexcept {
    // Initialize sentinel: points to itself (empty list)
    sentinel_.prev = &sentinel_;
    sentinel_.next = &sentinel_;
  }

  LruList(const LruList &) = delete;
  LruList &operator=(const LruList &) = delete;
  LruList(LruList &&) = delete;
  LruList &operator=(LruList &&) = delete;

  /**
   * @brief Check if list is empty.
   */
  [[nodiscard]] constexpr bool empty() const noexcept {
    return sentinel_.next == &sentinel_;
  }

  /**
   * @brief Get current size.
   */
  [[nodiscard]] constexpr std::size_t size() const noexcept { return size_; }

  /**
   * @brief Move node to front (mark as most recently used).
   *
   * O(1) operation.
   *
   * @param node Node to move to front
   */
  void touch(Node *node) noexcept {
    if (node == nullptr || node == &sentinel_) {
      return;
    }
    // Already at front?
    if (sentinel_.next == node) {
      return;
    }
    // Unlink from current position
    unlinkNode(node);
    // Link at front
    linkAfter(&sentinel_, node);
  }

  /**
   * @brief Insert node at front (most recently used position).
   *
   * O(1) operation. Node must not already be in a list.
   *
   * @param node Node to insert
   */
  void pushFront(Node *node) noexcept {
    if (node == nullptr || node == &sentinel_) {
      return;
    }
    linkAfter(&sentinel_, node);
    ++size_;
  }

  /**
   * @brief Remove and return least recently used node.
   *
   * O(1) operation. Returns nullptr if list is empty.
   *
   * @return Pointer to removed node, or nullptr if empty
   */
  Node *popBack() noexcept {
    if (empty()) {
      return nullptr;
    }
    Node *lru = sentinel_.prev;
    unlinkNode(lru);
    --size_;
    return lru;
  }

  /**
   * @brief Remove specific node from list.
   *
   * O(1) operation. Node must be in this list.
   *
   * @param node Node to remove
   */
  void remove(Node *node) noexcept {
    if (node == nullptr || node == &sentinel_) {
      return;
    }
    unlinkNode(node);
    --size_;
  }

  /**
   * @brief Get least recently used node without removing.
   *
   * @return Pointer to LRU node, or nullptr if empty
   */
  [[nodiscard]] Node *back() const noexcept {
    if (empty()) {
      return nullptr;
    }
    return sentinel_.prev;
  }

  /**
   * @brief Get most recently used node without removing.
   *
   * @return Pointer to MRU node, or nullptr if empty
   */
  [[nodiscard]] Node *front() const noexcept {
    if (empty()) {
      return nullptr;
    }
    return sentinel_.next;
  }

  /**
   * @brief Clear all nodes from list.
   *
   * Note: Does not delete nodes, just unlinks them.
   */
  void clear() noexcept {
    sentinel_.prev = &sentinel_;
    sentinel_.next = &sentinel_;
    size_ = 0;
  }

private:
  /**
   * @brief Link node after the given position.
   */
  void linkAfter(Node *pos, Node *node) noexcept {
    node->prev = pos;
    node->next = pos->next;
    pos->next->prev = node;
    pos->next = node;
  }

  /**
   * @brief Unlink node from its current position.
   */
  void unlinkNode(Node *node) noexcept {
    node->prev->next = node->next;
    node->next->prev = node->prev;
    node->prev = nullptr;
    node->next = nullptr;
  }

  mutable Node sentinel_{};
  std::size_t size_{0};
};

} // namespace orteaf::internal::base
