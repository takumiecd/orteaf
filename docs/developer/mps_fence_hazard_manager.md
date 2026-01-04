# MPS Fence 寿命管理 (設計メモ)

## 目的
フェンスの参照がユーザー側から消えても、実際に処理が終わるまで
`MTLFence` を再利用してはいけない。完了判定は
`MpsFenceHazard::isCompleted()`（command buffer 起点）で行い、フェンスの
寿命を安全に管理する。

## 方針
`MpsFenceManager::StrongFenceLease` を **コマンドキュー単位の FIFO** で管理する。
同一キュー上の command buffer は順序通りに完了するため、**新しいものが
完了していれば、より古いものも完了している**とみなしてまとめて解放する。

## データ構造
- `MpsFenceHazard`（`MpsFenceManager` の payload）
  - `MpsFence_t fence`
  - `MpsCommandBuffer_t command_buffer`
  - `CommandQueueHandle command_queue_handle`
  - `isReady<FastOps>()` で `command_buffer` を `nullptr` に更新
- `MpsFenceLifetimeManager`（名称案）
  - `MpsFenceManager*` を保持
  - `StrongFenceLease` の FIFO を保持
  - `acquire` と `releaseReady` を提供（寿命管理の単一入口）
- `MpsCommandQueueResource`
  - `MpsCommandQueue_t queue`
  - `MpsFenceLifetimeManager lifetime`
  - `MpsFenceManager*` を注入して利用
- `MpsCommandQueueManager`
  - `MpsFenceManager*` を保持し、`MpsCommandQueueResource` に渡す

## FIFO 解放アルゴリズム（queueごと）
1. 生成順に FIFO へ追加する。
2. メンテナンス時、**後ろ（最新）から isReady を確認**する。
   - `false` なら何も解放しない。
   - `true` なら **先頭からその要素まで** をまとめて release。
3. release された `StrongFenceLease` は `MpsFenceManager` に返却される。

## API 案
### MpsFenceLifetimeManager（queueごと）
- `StrongFenceLease acquire();`
  - `MpsFenceManager` から lease を取得する唯一の入口
- `void track(StrongFenceLease &&lease);`
  - `command_buffer` 設定済みの lease を FIFO に追加
- `std::size_t releaseReady();`
  - 解放した件数を返す
- `void clear();`
  - すべて release（queue破棄時）

### MpsCommandQueueResource
- `MpsCommandQueue_t queue;`
- `MpsFenceLifetimeManager lifetime;`

## 統合ポイント
- フェンスを更新した直後に `track` する。
- メンテナンスは以下のどこかで呼ぶ想定：
  - reuse policy の処理
  - resource の `isCompleted()` チェック
  - runtime の明示的メンテナンス

## 仮定
- 同一 command queue 上では完了順序が保持される。
- フェンス再利用の唯一の条件は `MpsFenceHazard::isReady()`。
