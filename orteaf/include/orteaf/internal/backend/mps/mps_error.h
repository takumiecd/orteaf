#pragma once

#include <string>
#include <string_view>

struct MPSError_st; using MPSError_t = MPSError_st*;

static_assert(sizeof(MPSError_t) == sizeof(void*), "MPSError must be pointer-sized.");

namespace orteaf::internal::backend::mps {

MPSError_t create_error(const std::string& message);
MPSError_t create_error(std::string_view domain, std::string_view description);
MPSError_t create_error(std::string_view domain,
                        std::string_view description,
                        void* additional_user_info);
void destroy_error(MPSError_t error);

} // namespace orteaf::internal::backend::mps