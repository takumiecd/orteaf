#include <gtest/gtest.h>

#include "orteaf/internal/runtime/kernel/mps/mps_kernel_launcher_impl.h"
#include "orteaf/internal/runtime/ops/mps/private/mps_private_ops.h"

namespace base = orteaf::internal::base;

namespace mps_rt = orteaf::internal::runtime::mps;

TEST(MpsKernelLauncherImplTest, StoresUniqueKeysInOrder) {
    mps_rt::MpsKernelLauncherImpl<3> impl({
        {"libA", "funcX"},
        {"libB", "funcY"},
        {"libA", "funcX"},  // duplicate should be ignored
    });

    const auto& keys = impl.keysForTest();
    ASSERT_EQ(impl.sizeForTest(), 2u);

    EXPECT_EQ(keys[0].first.identifier, "libA");
    EXPECT_EQ(keys[0].second.identifier, "funcX");

    EXPECT_EQ(keys[1].first.identifier, "libB");
    EXPECT_EQ(keys[1].second.identifier, "funcY");
}

namespace {

class DummyPrivateOps {
public:
    using PipelineLease = mps_rt::MpsComputePipelineStateManager::PipelineLease;
    static void reset() {
        last_device = {};
        last_library.clear();
        last_function.clear();
    }

    // Record requests and return dummy leases.
    static PipelineLease acquirePipeline(base::DeviceHandle device,
                                         const mps_rt::LibraryKey& library_key,
                                         const mps_rt::FunctionKey& function_key) {
        last_device = device;
        last_library = library_key.identifier;
        last_function = function_key.identifier;
        // Return an empty lease; we only validate call ordering and size.
        return PipelineLease{};
    }

    static inline base::DeviceHandle last_device{};
    static inline std::string last_library{};
    static inline std::string last_function{};
};

}  // namespace

TEST(MpsKernelLauncherImplTest, InitializeAcquiresPipelinesInOrder) {
    mps_rt::MpsKernelLauncherImpl<2> impl({
        {"libA", "funcX"},
        {"libB", "funcY"},
    });

    DummyPrivateOps::reset();
    const base::DeviceHandle device{0};

    impl.initialize<DummyPrivateOps>(device);

    // validate size and that initialized flag is set
    EXPECT_TRUE(impl.initialized());
    EXPECT_EQ(impl.sizeForTest(), 2u);

    EXPECT_EQ(DummyPrivateOps::last_device, device);
    EXPECT_EQ(DummyPrivateOps::last_library, "libB");    // last call should be second key
    EXPECT_EQ(DummyPrivateOps::last_function, "funcY");  // last call should be second key
}
