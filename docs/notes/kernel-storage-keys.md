# Kernel Storage/Param Keys (Draft)

## Goal
Support tensors that may own multiple storages by distinguishing the storage
role (Data/Index/etc.) from the kernel-side semantic (Input0/Output/etc.).

## Decisions
- **StorageKey = StorageId + StorageRole**
  - StorageId keeps kernel semantics.
  - StorageRole distinguishes sub-storages within a tensor.
  - Default role is `Data` for backward compatibility.
- **ParamKey = ParamId + optional StorageKey**
  - Global params remain unscoped.
  - Per-tensor params (shape/strides) can be scoped to a specific storage key.

## API Sketch
```cpp
StorageKey key = makeStorageKey(StorageId::Input0); // default role = Data
StorageKey idx = makeStorageKey(StorageId::Input0, StorageRole::Index);

ParamKey global = ParamKey::global(ParamId::Alpha);
ParamKey scoped = ParamKey::scoped(ParamId::Shape, key);
```

## Current Integration
- Storage bindings and storage lists are keyed by `StorageKey`.
- `StorageField` / `OptionalStorageField` accept `(StorageId, StorageRole)`.
- `ParamKey` is supported by `Param` and `ParamList` with global/scoped lookup.

## Migration Notes
- Existing code using `StorageId` works via default role `Data`.
- Multi-storage impls should specify roles explicitly.
