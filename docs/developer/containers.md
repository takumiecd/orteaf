# Base Containers (HeapVector, BlockVector, SmallVector)

This document describes the custom container types in
`orteaf/include/orteaf/internal/base/` and how to use them safely.

## Overview

The project uses a small set of lightweight containers instead of `std::vector`
in core code paths. The goals are:

- predictable memory behavior (avoid surprising reallocations),
- minimal dependencies and clear invariants,
- explicit control over pointer stability.

The three primary containers are:

- `HeapVector<T>`: contiguous, heap-backed, minimal `std::vector`-like API.
- `BlockVector<T, BlockSize>`: segmented storage with stable element addresses.
- `RuntimeBlockVector<T>`: segmented storage with runtime-configurable block size.
- `SmallVector<T, N>`: contiguous with inline storage before spilling to heap.

## HeapVector

**Path:** `orteaf/include/orteaf/internal/base/heap_vector.h`

### Intent

Use `HeapVector` when you need contiguous storage and a compact API surface.
It behaves similarly to `std::vector` but is intentionally smaller in scope.

### Key properties

- contiguous storage,
- `resize()` can reallocate (pointers may be invalidated),
- `pushBack()`/`emplaceBack()` are available,
- `at()` provides bounds-checked access.

### Good fits

- small internal buffers,
- scratch arrays where pointer stability is not required,
- metadata vectors that can move freely.

## BlockVector

**Path:** `orteaf/include/orteaf/internal/base/runtime_block_vector.h`

### Intent

Use `BlockVector` when **pointer stability across growth** is required. It
stores elements in fixed-size blocks and never relocates existing elements when
growing the container.

### Key properties

- element addresses are stable across `resize()`/`reserve()`,
- storage is contiguous **within** each block, not globally contiguous,
- iteration walks elements by index and provides a random-access iterator,
- `at()` provides bounds-checked access.

### Good fits

- pools that hand out raw pointers or references,
- control block storage (e.g., lease managers),
- long-lived objects that must not move.

### Trade-offs

- slightly worse cache locality than a single contiguous array,
- higher memory overhead due to block granularity,
- iteration is still O(1) per step, but not a simple pointer increment.

## RuntimeBlockVector

**Path:** `orteaf/include/orteaf/internal/base/block_vector.h`

### Intent

Use `RuntimeBlockVector` when you want stable addresses but need to choose the
block size at runtime (e.g., from a configuration system).

### Key properties

- same pointer stability as `BlockVector`,
- block size is provided at construction time,
- slightly less compiler optimization potential due to runtime division/mod.

### Good fits

- manager or pool configuration derived at runtime,
- avoiding template parameter propagation across layers.

## SmallVector

**Path:** `orteaf/include/orteaf/internal/base/small_vector.h`

### Intent

Use `SmallVector` when you expect **small sizes most of the time** but still
need a vector-like API. Elements are stored inline up to `N`, then spill to
heap storage.

### Key properties

- contiguous storage,
- inline storage for up to `N` elements,
- `reserve()`/`shrinkToFit()` available,
- `insert()`/`emplace()`/`erase()` provide `std::vector`-like behavior.

### Exception safety notes

`insert()`/`emplace()` overloads that shift elements are constrained to
nothrow move/copy construction to avoid leaving storage gaps. If you need
insertion for types with throwing moves, prefer `HeapVector` or copy into a
temporary buffer and rebuild.

### Good fits

- small, short-lived arrays,
- inline configuration lists,
- performance-sensitive paths with small cardinality.

## Choosing the right container

Use this quick guide:

- **Need stable addresses across growth** → `BlockVector`.
- **Expect small size with occasional growth** → `SmallVector`.
- **Need simple contiguous storage** → `HeapVector`.

## Pitfalls to avoid

- Do not keep raw pointers into `HeapVector` or `SmallVector` across growth.
- Do not assume `BlockVector` is globally contiguous.
- Do not use `SmallVector::insert()` with types that can throw on move/copy.

## Related documentation

- `docs/developer/slot-pool.md` for pool behavior and freelist semantics.
- `docs/developer/runtime_manager_safety.md` for pointer stability constraints.
