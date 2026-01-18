#include <gtest/gtest.h>

#include "orteaf/internal/execution/cuda/platform/wrapper/cuda_kernel_embed_api.h"
#include "tests/internal/testing/error_assert.h"

using namespace orteaf::internal::execution::cuda::platform::wrapper::kernel_embed;

namespace {

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
