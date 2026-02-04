#include <gtest/gtest.h>

#include "orteaf/internal/init/library_init.h"
#include "orteaf/internal/tensor/api/tensor_api.h"

namespace init = ::orteaf::internal::init;

namespace {

class LibraryInitTest : public ::testing::Test {
protected:
  void TearDown() override { init::shutdown(); }
};

TEST_F(LibraryInitTest, InitializeThenShutdown) {
  init::initialize();
  EXPECT_TRUE(init::isInitialized());
  EXPECT_TRUE(::orteaf::internal::tensor::api::TensorApi::isConfigured());

  init::shutdown();
  EXPECT_FALSE(init::isInitialized());
  EXPECT_FALSE(::orteaf::internal::tensor::api::TensorApi::isConfigured());
}

TEST_F(LibraryInitTest, InitializeTwiceThrows) {
  init::initialize();
  EXPECT_ANY_THROW(init::initialize());
}

}  // namespace
