/**
 * @file mps_error.h
 * @brief Helpers to construct and destroy NSError-based opaque errors.
 */
#pragma once

#include <string>
#include <string_view>

namespace orteaf::internal::backend::mps {

struct MPSError_st; using MPSError_t = MPSError_st*;

static_assert(sizeof(MPSError_t) == sizeof(void*), "MPSError must be pointer-sized.");

/** Create an error in NSCocoaErrorDomain with a description. */
MPSError_t create_error(const std::string& message);
/** Create an error with explicit domain and description. */
MPSError_t create_error(std::string_view domain, std::string_view description);
/** Create an error with domain, description, and additional userInfo (NSDictionary*). */
MPSError_t create_error(std::string_view domain,
                        std::string_view description,
                        void* additional_user_info);
/** Release an error object; ignores nullptr. */
void destroy_error(MPSError_t error);

} // namespace orteaf::internal::backend::mps