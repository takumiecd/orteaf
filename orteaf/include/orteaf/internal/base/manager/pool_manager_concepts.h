#pragma once

#include <concepts>
#include <cstddef>

namespace orteaf::internal::base {

// =============================================================================
// PoolManager 型定義 Concept
// =============================================================================

/// @brief PoolManagerが必要な基本型を定義しているか
template <typename Manager>
concept PoolManagerTypeConcept = requires {
  typename Manager::PayloadPool;
  typename Manager::ControlBlock;
  typename Manager::ControlBlockHandle;
  typename Manager::PayloadHandle;
  typename Manager::WeakLeaseType;
  typename Manager::StrongLeaseType;
  typename Manager::Config;
};

// =============================================================================
// 状態確認 Concept
// =============================================================================

/// @brief 設定状態の確認をサポート
template <typename Manager>
concept ManagerStateQueryConcept =
    PoolManagerTypeConcept<Manager> && requires(const Manager &m) {
      { m.isConfigured() } -> std::convertible_to<bool>;
      { m.ensureConfigured() };
    };

// =============================================================================
// Payload 状態確認 Concept
// =============================================================================

/// @brief Payload の生存状態確認をサポート
template <typename Manager>
concept ManagerPayloadAliveCheckConcept =
    PoolManagerTypeConcept<Manager> &&
    requires(const Manager &m, typename Manager::PayloadHandle h) {
      { m.isAlive(h) } -> std::convertible_to<bool>;
    };

// =============================================================================
// Configuration Concept
// =============================================================================

/// @brief configure(config, request, context) をサポート
template <typename Manager, typename Request, typename Context>
concept ManagerConfigurableConcept =
    PoolManagerTypeConcept<Manager> &&
    requires(Manager &m, const typename Manager::Config &cfg,
             const Request &req, const Context &ctx) {
      { m.configure(cfg, req, ctx) };
    };

// =============================================================================
// Shutdown Concept
// =============================================================================

/// @brief shutdown(request, context)をサポート
/// PoolManager の shutdown は checkCanTeardown → payload clear →
/// checkCanShutdown → control block clear の流れで行う
template <typename Manager, typename Request, typename Context>
concept ManagerShutdownableConcept =
    PoolManagerTypeConcept<Manager> &&
    requires(Manager &m, const Request &req, const Context &ctx) {
      { m.shutdown(req, ctx) };
    };

// =============================================================================
// Block Size 設定 Concept
// =============================================================================

/// @brief block size 設定をサポート
template <typename Manager>
concept ManagerBlockSizeSettableConcept =
    PoolManagerTypeConcept<Manager> && requires(Manager &m, std::size_t n) {
      { m.setControlBlockBlockSize(n) };
      { m.setPayloadBlockSize(n) };
    };

// =============================================================================
// Growth Chunk Size Concept
// =============================================================================

/// @brief growth chunk size 設定をサポート
template <typename Manager>
concept ManagerGrowthChunkSettableConcept =
    PoolManagerTypeConcept<Manager> &&
    requires(Manager &m, const Manager &cm, std::size_t n) {
      { cm.controlBlockGrowthChunkSize() } -> std::convertible_to<std::size_t>;
      { m.setControlBlockGrowthChunkSize(n) };
      { cm.payloadGrowthChunkSize() } -> std::convertible_to<std::size_t>;
      { m.setPayloadGrowthChunkSize(n) };
    };

// =============================================================================
// Payload 操作 Concept
// =============================================================================

/// @brief Payload 全作成をサポート
template <typename Manager, typename Request, typename Context>
concept ManagerPayloadCreateAllConcept =
    PoolManagerTypeConcept<Manager> &&
    requires(Manager &m, const Request &req, const Context &ctx) {
      { m.createAllPayloads(req, ctx) } -> std::convertible_to<bool>;
    };

/// @brief Payload の emplace をサポート
template <typename Manager, typename Request, typename Context>
concept ManagerPayloadEmplaceConcept =
    PoolManagerTypeConcept<Manager> &&
    requires(Manager &m, typename Manager::PayloadHandle h,
             const Request &req, const Context &ctx) {
      { m.emplacePayload(h, req, ctx) } -> std::convertible_to<bool>;
    };

/// @brief Payload の acquire をサポート
template <typename Manager, typename Request, typename Context>
concept ManagerPayloadAcquirableConcept =
    PoolManagerTypeConcept<Manager> &&
    requires(Manager &m, const Request &req, const Context &ctx) {
      {
        m.acquirePayloadOrGrowAndCreate(req, ctx)
      } -> std::same_as<typename Manager::PayloadHandle>;
    };

/// @brief Payload の reserve をサポート
template <typename Manager>
concept ManagerPayloadReservableConcept =
    PoolManagerTypeConcept<Manager> && requires(Manager &m) {
      { m.reserveUncreatedPayloadOrGrow() } -> std::same_as<
          typename Manager::PayloadHandle>;
    };

// =============================================================================
// Lease 取得 Concept
// =============================================================================

/// @brief WeakLease 取得をサポート
template <typename Manager>
concept ManagerWeakLeaseAcquirableConcept =
    PoolManagerTypeConcept<Manager> &&
    requires(Manager &m, typename Manager::PayloadHandle h) {
      {
        m.acquireWeakLease(h)
      } -> std::same_as<typename Manager::WeakLeaseType>;
    };

/// @brief StrongLease 取得をサポート
template <typename Manager>
concept ManagerStrongLeaseAcquirableConcept =
    PoolManagerTypeConcept<Manager> &&
    requires(Manager &m, typename Manager::PayloadHandle h) {
      {
        m.acquireStrongLease(h)
      } -> std::same_as<typename Manager::StrongLeaseType>;
    };

// =============================================================================
// 組み合わせ Concept
// =============================================================================

/// @brief PoolManager の全機能をまとめた Concept
template <typename Manager, typename Request, typename Context>
concept FullPoolManagerConcept =
    PoolManagerTypeConcept<Manager> &&
    ManagerStateQueryConcept<Manager> &&
    ManagerShutdownableConcept<Manager, Request, Context> &&
    ManagerConfigurableConcept<Manager, Request, Context> &&
    ManagerBlockSizeSettableConcept<Manager> &&
    ManagerGrowthChunkSettableConcept<Manager> &&
    ManagerPayloadAliveCheckConcept<Manager> &&
    ManagerPayloadCreateAllConcept<Manager, Request, Context> &&
    ManagerPayloadEmplaceConcept<Manager, Request, Context> &&
    ManagerPayloadAcquirableConcept<Manager, Request, Context> &&
    ManagerPayloadReservableConcept<Manager> &&
    ManagerWeakLeaseAcquirableConcept<Manager> &&
    ManagerStrongLeaseAcquirableConcept<Manager>;

} // namespace orteaf::internal::base
