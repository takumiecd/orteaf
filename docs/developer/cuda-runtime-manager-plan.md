# CUDA Runtime Manager Plan

## References
- MPS manager patterns: `orteaf/include/orteaf/internal/execution/mps/manager/`
- MPS manager implementations: `orteaf/src/internal/execution/mps/manager/`
- CPU manager patterns: `orteaf/include/orteaf/internal/execution/cpu/manager/`
- CUDA wrappers: `orteaf/include/orteaf/internal/execution/cuda/platform/wrapper/`

Note: This repo does not currently have `runtime/mps/manager`; the plan below mirrors
`execution/mps/manager` and maps it into `runtime/cuda/manager`.

## Goal
Introduce a CUDA runtime manager layer under `internal/runtime/cuda/manager` that owns
CUDA resources (device/context/stream/event) and exposes a clean lifecycle API similar
to the existing MPS/CPU managers.

## Target File Layout
- `orteaf/include/orteaf/internal/runtime/cuda/manager/`
  - `cuda_execution_manager.h`
  - `cuda_device_manager.h`
  - `cuda_buffer_manager.h`
  - `cuda_context_manager.h`
  - `cuda_stream_manager.h`
  - `cuda_event_manager.h`
- `orteaf/src/internal/runtime/cuda/manager/`
  - `cuda_execution_manager.cpp`
  - `cuda_device_manager.cpp`
  - `cuda_buffer_manager.cpp`
  - `cuda_context_manager.cpp`
  - `cuda_stream_manager.cpp`
  - `cuda_event_manager.cpp`
- `orteaf/include/orteaf/internal/execution/cuda/platform/`
  - Extend `cuda_slow_ops.h`
- `orteaf/src/internal/execution/cuda/platform/`
  - `cuda_slow_ops.cpp` (default SlowOps implementation)

## Responsibilities
- `CudaExecutionManager`
  - Owns a `CudaSlowOps` instance.
  - Configures and shuts down `CudaDeviceManager`.
- `CudaDeviceManager`
  - Enumerates CUDA devices via SlowOps.
  - Creates per-device payloads and owns `CudaContextManager`.
  - Tracks device architecture metadata.
- `CudaContextManager`
  - Creates and releases CUDA contexts for a device (initially primary context).
- `CudaBufferManager`
  - Allocates and frees CUDA device buffers using the CUDA allocator resource.
- `CudaStreamManager`
  - Creates/destroys streams within a device context.
- `CudaEventManager`
  - Creates/destroys events within a device context.

## Manager Relationships
- `CudaExecutionManager` -> `CudaDeviceManager`.
- `CudaDeviceManager` -> per-device resource that holds `CudaContextManager`.
- `CudaContextManager` -> per-context resource that holds buffer/stream/event managers.
- `CudaStreamManager` / `CudaEventManager` require an active CUDA context; manager
  configure step sets the context before resource creation.

## Implementation Notes
- Use `PoolManager` with `FixedSlotStore` for devices (device count is fixed).
- Use `PoolManager` with `SlotPool` for contexts/streams/events (growable pools).
- Follow error handling patterns from MPS managers (`throwError` on invalid config).
- Keep API test-friendly with `configureForTest` and `isConfiguredForTest` hooks
  guarded by `ORTEAF_ENABLE_TEST`.

## Milestones
1. Extend `CudaSlowOps` interface and add a default implementation that forwards to
   the existing CUDA wrapper functions.
2. Add headers for CUDA runtime managers with configs and public APIs.
3. Implement manager sources and wire the device manager to configure sub-managers.
