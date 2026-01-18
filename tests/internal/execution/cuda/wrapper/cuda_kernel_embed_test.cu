#include <gtest/gtest.h>

#include <string>

#include "orteaf/internal/execution/cuda/platform/wrapper/cuda_context.h"
#include "orteaf/internal/execution/cuda/platform/wrapper/cuda_device.h"
#include "orteaf/internal/execution/cuda/platform/wrapper/cuda_init.h"
#include "orteaf/internal/execution/cuda/platform/wrapper/cuda_kernel_embed_api.h"
#include "orteaf/internal/execution/cuda/platform/wrapper/cuda_module.h"
#include "tests/internal/testing/error_assert.h"

using namespace orteaf::internal::execution::cuda::platform::wrapper::kernel_embed;
namespace cuda = orteaf::internal::execution::cuda::platform::wrapper;

namespace {

struct ScopedCudaContext {
    cuda::CudaDevice_t device{};
    cuda::CudaContext_t context{nullptr};
    bool ready{false};

    ScopedCudaContext() {
        cuda::cudaInit();
        int count = cuda::getDeviceCount();
        if (count == 0) {
            return;
        }
        device = cuda::getDevice(0);
        context = cuda::getPrimaryContext(device);
        cuda::setContext(context);
        ready = (context != nullptr);
    }

    ~ScopedCudaContext() {
        if (context != nullptr) {
            cuda::releasePrimaryContext(device);
        }
    }
};

bool has_format(std::string_view name, CudaKernelFmt fmt) {
    for (const auto &entry : kernels()) {
        if (std::string_view(entry.name) == name && entry.fmt == fmt) {
            return true;
        }
    }
    return false;
}

}  // namespace

TEST(CudaKernelEmbedTest, CanRetrieveEmbeddedBlob) {
    constexpr std::string_view kKernelName = "embed_test_kernels";

    // Request a format that might not exist to validate fallback logic.
    Blob blob = findKernelData(kKernelName, CudaKernelFmt::Ptx);
#if ORTEAF_EMBED_HAS_FATBIN || ORTEAF_EMBED_HAS_CUBIN || ORTEAF_EMBED_HAS_PTX
    EXPECT_NE(blob.data, nullptr);
    EXPECT_GT(blob.size, 0);
#else
    EXPECT_EQ(blob.data, nullptr);
    EXPECT_EQ(blob.size, 0);
#endif

    // Each advertised format should agree with available().
#if ORTEAF_EMBED_HAS_FATBIN
    EXPECT_TRUE(has_format(kKernelName, CudaKernelFmt::Fatbin));
#else
    EXPECT_FALSE(has_format(kKernelName, CudaKernelFmt::Fatbin));
#endif
#if ORTEAF_EMBED_HAS_CUBIN
    EXPECT_TRUE(has_format(kKernelName, CudaKernelFmt::Cubin));
#else
    EXPECT_FALSE(has_format(kKernelName, CudaKernelFmt::Cubin));
#endif
#if ORTEAF_EMBED_HAS_PTX
    EXPECT_TRUE(has_format(kKernelName, CudaKernelFmt::Ptx));
#else
    EXPECT_FALSE(has_format(kKernelName, CudaKernelFmt::Ptx));
#endif

    // Looking up a missing kernel should return an empty blob.
    Blob missing = findKernelData("nonexistent_kernel_for_embed_test");
    EXPECT_EQ(missing.data, nullptr);
    EXPECT_EQ(missing.size, 0);
    EXPECT_FALSE(available("nonexistent_kernel_for_embed_test"));
}

TEST(CudaKernelEmbedTest, FindAnyAvailableReturnsBlobWhenFormatsPresent) {
    constexpr std::string_view kKernelName = "embed_test_kernels";
    Blob blob = findKernelData(kKernelName);

#if ORTEAF_EMBED_HAS_FATBIN || ORTEAF_EMBED_HAS_CUBIN || ORTEAF_EMBED_HAS_PTX
    ASSERT_NE(blob.data, nullptr);
    EXPECT_GT(blob.size, 0);
#else
    GTEST_SKIP() << "No CUDA kernel formats were generated.";
#endif
    EXPECT_TRUE(available(kKernelName));
}

TEST(CudaKernelEmbedTest, EmptyKernelNameThrows) {
    ::orteaf::tests::ExpectError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
        [] { findKernelData(std::string_view{}); });
}

TEST(CudaKernelEmbedTest, EntriesExposeEmbeddedKernelData) {
    constexpr std::string_view kKernelName = "embed_test_kernels";
    if (!available(kKernelName)) {
        GTEST_SKIP() << "Embedded CUDA kernel not available in this build.";
    }

    auto entries = kernels();
    ASSERT_FALSE(entries.empty());

    bool found = false;
    for (const auto &entry : entries) {
        if (std::string_view(entry.name) == kKernelName) {
            found = true;
            EXPECT_NE(entry.begin, nullptr);
            EXPECT_NE(entry.end, nullptr);
            EXPECT_GT(entry.end, entry.begin);
            break;
        }
    }
    EXPECT_TRUE(found);
}

TEST(CudaKernelEmbedTest, CreateEmbeddedModuleFailsWithMissingKernel) {
    auto module = createEmbeddedModule("nonexistent_kernel_for_embed_test");
    EXPECT_EQ(module, nullptr);
}

TEST(CudaKernelEmbedTest, CreateEmbeddedModuleEmptyNameThrows) {
    ::orteaf::tests::ExpectError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
        [] { createEmbeddedModule(std::string_view{}); });
}

TEST(CudaKernelEmbedTest, CreateEmbeddedModuleLoadsAndFindsFunction) {
    constexpr std::string_view kKernelName = "embed_test_kernels";
    constexpr std::string_view kKernelSymbol = "orteaf_embed_test_fill_kernel";

    if (!available(kKernelName)) {
        GTEST_SKIP() << "Embedded CUDA kernel not available in this build.";
    }

    ScopedCudaContext context;
    if (!context.ready) {
        GTEST_SKIP() << "No CUDA device available.";
    }

    auto module = createEmbeddedModule(kKernelName);
    ASSERT_NE(module, nullptr);

    std::string kernel_name{kKernelSymbol};
    auto fn = cuda::getFunction(module, kernel_name.c_str());
    EXPECT_NE(fn, nullptr);

    cuda::unloadModule(module);
}
