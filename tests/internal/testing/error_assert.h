#pragma once

#include "orteaf/internal/diagnostics/error/error.h"

#include <gtest/gtest.h>

#include <system_error>

namespace orteaf::tests {

/**
 * @brief Assert helper that verifies a callable throws std::system_error with the expected OrteafErrc.
 *
 * Usage:
 *   ExpectError(OrteafErrc::NullPointer, [] { someFunction(nullptr); });
 */
template <typename Fn>
void ExpectError(::orteaf::internal::diagnostics::error::OrteafErrc errc, Fn&& fn) {
    namespace diag = ::orteaf::internal::diagnostics::error;
    const auto expected = diag::makeErrorCode(errc);
    EXPECT_THROW(
        try {
            fn();
        } catch (const std::system_error& ex) {
            EXPECT_EQ(ex.code(), expected);
            throw;
        },
        std::system_error);
}

}  // namespace orteaf::tests
