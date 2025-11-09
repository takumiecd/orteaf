find_package(yaml-cpp REQUIRED)

add_executable(gen_dtypes gen_dtypes.cpp)

target_link_libraries(gen_dtypes
    PRIVATE
        yaml-cpp::yaml-cpp
)

target_compile_features(gen_dtypes PRIVATE cxx_std_20)

add_executable(gen_ops gen_ops.cpp)

target_link_libraries(gen_ops
    PRIVATE
        yaml-cpp::yaml-cpp
)

target_compile_features(gen_ops PRIVATE cxx_std_20)
