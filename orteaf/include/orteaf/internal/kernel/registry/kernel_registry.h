#pragma once

#include <memory>
#include <unordered_map>

#include "orteaf/internal/base/lru_list.h"
#include "orteaf/internal/kernel/kernel_key.h"
#include "orteaf/internal/kernel/registry/kernel_entry_traits.h"
#include "orteaf/internal/kernel/registry/kernel_registry_config.h"

namespace orteaf::internal::kernel::registry {

/**
 * @brief Template-based 3-tier cache KernelRegistry with LRU eviction.
 *
 * Manages kernel entries in a hierarchical cache:
 * - Cache (L1): Fixed-size, fastest access, holds pointers to Main Memory
 * - Main Memory: Configurable capacity, holds full Entry
 * - Secondary Storage: Unbounded, holds lightweight Metadata for reconstruction
 *
 * Uses LRU (Least Recently Used) algorithm for eviction at each tier.
 * KernelKey serves as the virtual address for kernel lookup.
 *
 * @tparam Traits Must satisfy KernelEntryTraitsConcept
 */
template <KernelEntryTraitsConcept Traits> class KernelRegistry {
public:
  using Entry = typename Traits::Entry;
  using Metadata = typename Traits::Metadata;
  using Key = ::orteaf::internal::kernel::KernelKey;

  /**
   * @brief Statistics for cache performance monitoring.
   */
  struct Stats {
    std::size_t cache_hits{0};
    std::size_t main_memory_hits{0};
    std::size_t secondary_hits{0};
    std::size_t misses{0};
  };

  /**
   * @brief Construct registry with default configuration.
   */
  KernelRegistry() = default;

  /**
   * @brief Construct registry with custom configuration.
   */
  explicit KernelRegistry(KernelRegistryConfig config) : config_(config) {}

  // Non-copyable
  KernelRegistry(const KernelRegistry &) = delete;
  KernelRegistry &operator=(const KernelRegistry &) = delete;
  KernelRegistry(KernelRegistry &&) = default;
  KernelRegistry &operator=(KernelRegistry &&) = default;

  /**
   * @brief Look up a kernel by key.
   *
   * Searches through all tiers (Cache → Main Memory → Secondary Storage).
   * If found in a lower tier, promotes to higher tier.
   * Updates LRU position on access.
   *
   * @param key Kernel key to look up
   * @return Pointer to entry, or nullptr if not found
   */
  Entry *lookup(Key key) {
    // Try Cache first
    if (auto *entry = lookupCache(key)) {
      ++stats_.cache_hits;
      return entry;
    }

    // Try Main Memory
    if (auto *entry = lookupMainMemory(key)) {
      ++stats_.main_memory_hits;
      promoteToCache(key, entry);
      return entry;
    }

    // Try Secondary Storage (rebuilds entry)
    if (auto *entry = lookupSecondaryStorage(key)) {
      ++stats_.secondary_hits;
      promoteToCache(key, entry);
      return entry;
    }

    ++stats_.misses;
    return nullptr;
  }

  /**
   * @brief Register a kernel with metadata (Demand Paging model).
   *
   * Kernel is registered in Secondary Storage. On first lookup(),
   * the entry will be rebuilt and promoted to Main Memory and Cache.
   *
   * @param key Kernel key for lookup
   * @param metadata Metadata for kernel reconstruction
   */
  void registerKernel(Key key, Metadata metadata) {
    // If already in any tier, skip
    if (cache_.count(key) > 0 || main_memory_.count(key) > 0 ||
        secondary_storage_.count(key) > 0) {
      return;
    }
    secondary_storage_[key] = std::move(metadata);
  }

  /**
   * @brief Check if a kernel is registered (in any tier).
   *
   * @param key Kernel key to check
   * @return true if kernel is registered
   */
  [[nodiscard]] bool contains(Key key) const {
    return cache_.count(key) > 0 || main_memory_.count(key) > 0 ||
           secondary_storage_.count(key) > 0;
  }

  /**
   * @brief Prefetch a kernel into Cache from lower tiers.
   *
   * @param key Kernel key to prefetch
   * @return true if kernel was found and prefetched
   */
  bool prefetch(Key key) { return lookup(key) != nullptr; }

  /**
   * @brief Flush Cache tier to Main Memory.
   *
   * Clears Cache but keeps entries in Main Memory.
   */
  void flush() {
    cache_.clear();
    cache_lru_.clear();
    cache_nodes_.clear();
  }

  /**
   * @brief Clear all tiers.
   */
  void clear() {
    cache_.clear();
    cache_lru_.clear();
    cache_nodes_.clear();

    main_memory_.clear();
    main_memory_lru_.clear();
    main_memory_nodes_.clear();

    secondary_storage_.clear();

    stats_ = {};
  }

  /**
   * @brief Get performance statistics.
   */
  [[nodiscard]] const Stats &stats() const noexcept { return stats_; }

  /**
   * @brief Get current Cache tier size.
   */
  [[nodiscard]] std::size_t cacheSize() const noexcept { return cache_.size(); }

  /**
   * @brief Get current Main Memory tier size.
   */
  [[nodiscard]] std::size_t mainMemorySize() const noexcept {
    return main_memory_.size();
  }

  /**
   * @brief Get current Secondary Storage tier size.
   */
  [[nodiscard]] std::size_t secondaryStorageSize() const noexcept {
    return secondary_storage_.size();
  }

  /**
   * @brief Get configuration.
   */
  [[nodiscard]] const KernelRegistryConfig &config() const noexcept {
    return config_;
  }

private:
  using LruNode = base::LruNode<Key>;
  using LruList = base::LruList<Key>;

  // ----- Cache Tier -----

  Entry *lookupCache(Key key) {
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      return nullptr;
    }
    // Touch in Cache LRU
    auto node_it = cache_nodes_.find(key);
    if (node_it != cache_nodes_.end()) {
      cache_lru_.touch(&node_it->second);
    }
    return it->second;
  }

  void promoteToCache(Key key, Entry *entry) {
    // Already in Cache?
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      auto node_it = cache_nodes_.find(key);
      if (node_it != cache_nodes_.end()) {
        cache_lru_.touch(&node_it->second);
      }
      return;
    }

    // Evict from Cache if at capacity
    if (cache_.size() >= config_.cache_capacity) {
      evictFromCache();
    }

    // Insert into Cache
    cache_[key] = entry;
    cache_nodes_[key] = LruNode(key);
    cache_lru_.pushFront(&cache_nodes_[key]);
  }

  void evictFromCache() {
    auto *node = cache_lru_.popBack();
    if (node == nullptr) {
      return;
    }
    Key key = node->key;
    cache_.erase(key);
    cache_nodes_.erase(key);
    // Entry stays in Main Memory (not demoted further)
  }

  // ----- Main Memory Tier -----

  Entry *lookupMainMemory(Key key) {
    auto it = main_memory_.find(key);
    if (it == main_memory_.end()) {
      return nullptr;
    }
    touchMainMemory(key);
    return it->second.get();
  }

  void touchMainMemory(Key key) {
    auto node_it = main_memory_nodes_.find(key);
    if (node_it != main_memory_nodes_.end()) {
      main_memory_lru_.touch(&node_it->second);
    }
  }

  void evictFromMainMemory() {
    auto *node = main_memory_lru_.popBack();
    if (node == nullptr) {
      return;
    }
    Key key = node->key;

    // Demote to Secondary Storage
    auto it = main_memory_.find(key);
    if (it != main_memory_.end()) {
      secondary_storage_[key] = Traits::toMetadata(*it->second);
      main_memory_.erase(it);
    }
    main_memory_nodes_.erase(key);

    // Also remove from Cache if present
    cache_.erase(key);
    auto cache_node_it = cache_nodes_.find(key);
    if (cache_node_it != cache_nodes_.end()) {
      cache_lru_.remove(&cache_node_it->second);
      cache_nodes_.erase(cache_node_it);
    }
  }

  // ----- Secondary Storage Tier -----

  Entry *lookupSecondaryStorage(Key key) {
    auto it = secondary_storage_.find(key);
    if (it == secondary_storage_.end()) {
      return nullptr;
    }

    // Rebuild entry and promote to Main Memory
    Entry rebuilt = it->second.rebuild();
    secondary_storage_.erase(it);

    // Evict if needed
    if (main_memory_.size() >= config_.main_memory_capacity) {
      evictFromMainMemory();
    }

    // Insert into Main Memory
    main_memory_[key] = std::make_unique<Entry>(std::move(rebuilt));
    main_memory_nodes_[key] = LruNode(key);
    main_memory_lru_.pushFront(&main_memory_nodes_[key]);

    return main_memory_[key].get();
  }

  // ----- Storage -----

  // Cache tier: pointers to Main Memory entries
  std::unordered_map<Key, Entry *> cache_;
  std::unordered_map<Key, LruNode> cache_nodes_;
  LruList cache_lru_;

  // Main Memory tier: owned entries
  std::unordered_map<Key, std::unique_ptr<Entry>> main_memory_;
  std::unordered_map<Key, LruNode> main_memory_nodes_;
  LruList main_memory_lru_;

  // Secondary Storage tier: lightweight metadata
  std::unordered_map<Key, Metadata> secondary_storage_;

  KernelRegistryConfig config_;
  Stats stats_;
};

} // namespace orteaf::internal::kernel::registry

// Convenience type aliases for specific backends
#if ORTEAF_ENABLE_MPS
#include "orteaf/internal/kernel/mps/mps_kernel_metadata.h"
namespace orteaf::internal::kernel::registry {
using MpsKernelRegistry = KernelRegistry<MpsKernelEntryTraits>;
} // namespace orteaf::internal::kernel::registry
#endif
