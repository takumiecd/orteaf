# Kernel Operand/Param Keys (Draft)

## Goal
Support tensors that may own multiple storages by distinguishing the operand
role (Data/Index/etc.) from the kernel-side semantic (Input0/Output/etc.).

## Decisions
- **OperandKey = OperandId + Role**
  - OperandId keeps kernel semantics.
  - Role distinguishes sub-storages within a tensor.
  - Default role is `Data` for backward compatibility.
- **ParamKey = ParamId + optional OperandKey**
  - Global params remain unscoped.
  - Per-tensor params (shape/strides) can be scoped to a specific operand key.

## API Sketch
```cpp
OperandKey key = makeOperandKey(OperandId::Input0); // default role = Data
OperandKey idx = makeOperandKey(OperandId::Input0, Role::Index);

ParamKey global = ParamKey::global(ParamId::Alpha);
ParamKey scoped = ParamKey::scoped(ParamId::Shape, key);
```

## Current Integration
- Storage bindings and storage lists are keyed by `OperandKey`.
- `StorageField` / `OptionalStorageField` accept `(OperandId, Role)`.
- `ParamKey` is supported by `Param` and `ParamList` with global/scoped lookup.

## Migration Notes
- Existing code using `OperandId` works via default role `Data`.
- Multi-storage impls should specify roles explicitly.
