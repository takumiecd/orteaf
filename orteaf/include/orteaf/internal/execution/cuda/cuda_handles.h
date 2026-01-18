#pragma once

#include <orteaf/internal/base/handle.h>

namespace orteaf::internal::execution::cuda {

struct CudaDeviceTag {};
struct CudaStreamTag {};
struct CudaContextTag {};
struct CudaBufferTag {};
struct CudaGraphTag {};
struct CudaEventTag {};
struct CudaFenceTag {};
struct CudaModuleTag {};

using CudaDeviceHandle = ::orteaf::internal::base::Handle<CudaDeviceTag, uint32_t, void>;
using CudaStreamHandle = ::orteaf::internal::base::Handle<CudaStreamTag, uint32_t, uint8_t>;
using CudaContextHandle = ::orteaf::internal::base::Handle<CudaContextTag, uint32_t, uint8_t>;
using CudaBufferHandle = ::orteaf::internal::base::Handle<CudaBufferTag, uint32_t, uint32_t>;
using CudaBufferViewHandle = ::orteaf::internal::base::Handle<CudaBufferTag, uint32_t, void>;
using CudaGraphHandle = ::orteaf::internal::base::Handle<CudaGraphTag, uint32_t, uint8_t>;
using CudaEventHandle = ::orteaf::internal::base::Handle<CudaEventTag, uint32_t, uint8_t>;
using CudaFenceHandle = ::orteaf::internal::base::Handle<CudaFenceTag, uint32_t, uint8_t>;
using CudaModuleHandle = ::orteaf::internal::base::Handle<CudaModuleTag, uint32_t, uint8_t>;

static_assert(std::is_trivially_copyable_v<CudaDeviceHandle>);
static_assert(std::is_trivially_copyable_v<CudaBufferHandle>);
static_assert(std::is_trivially_copyable_v<CudaBufferViewHandle>);
static_assert(std::is_trivially_copyable_v<CudaEventHandle>);
static_assert(std::is_trivially_copyable_v<CudaFenceHandle>);
static_assert(std::is_trivially_copyable_v<CudaModuleHandle>);

} // namespace orteaf::internal::execution::cuda
