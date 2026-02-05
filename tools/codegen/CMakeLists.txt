find_package(yaml-cpp 0.8 REQUIRED CONFIG)

if(TARGET yaml-cpp::yaml-cpp)
    set(YAML_CPP_TARGET yaml-cpp::yaml-cpp)
elseif(TARGET yaml-cpp)
    set(YAML_CPP_TARGET yaml-cpp)
else()
    message(FATAL_ERROR "yaml-cpp target not found after find_package")
endif()

add_executable(gen_dtypes gen_dtypes.cpp)

target_link_libraries(gen_dtypes
    PRIVATE
        "${YAML_CPP_TARGET}"
)

target_compile_features(gen_dtypes PRIVATE cxx_std_20)

add_executable(gen_ops gen_ops.cpp)

target_link_libraries(gen_ops
    PRIVATE
        "${YAML_CPP_TARGET}"
)

target_compile_features(gen_ops PRIVATE cxx_std_20)

add_executable(gen_executions gen_executions.cpp)

target_link_libraries(gen_executions
    PRIVATE
        "${YAML_CPP_TARGET}"
)

target_compile_features(gen_executions PRIVATE cxx_std_20)

add_executable(gen_architectures gen_architectures.cpp)

target_link_libraries(gen_architectures
    PRIVATE
        "${YAML_CPP_TARGET}"
)

target_compile_features(gen_architectures PRIVATE cxx_std_20)

add_executable(gen_devices gen_devices.cpp)

target_link_libraries(gen_devices
    PRIVATE
        "${YAML_CPP_TARGET}"
)

target_compile_features(gen_devices PRIVATE cxx_std_20)

add_executable(gen_param_ids gen_param_ids.cpp)

target_link_libraries(gen_param_ids
    PRIVATE
        "${YAML_CPP_TARGET}"
)

target_compile_features(gen_param_ids PRIVATE cxx_std_20)

add_executable(gen_operand_ids gen_operand_ids.cpp)

target_link_libraries(gen_operand_ids
    PRIVATE
        "${YAML_CPP_TARGET}"
)

target_compile_features(gen_operand_ids PRIVATE cxx_std_20)

add_executable(gen_roles gen_roles.cpp)

target_link_libraries(gen_roles
    PRIVATE
        "${YAML_CPP_TARGET}"
)

target_compile_features(gen_roles PRIVATE cxx_std_20)

add_executable(gen_kernel_registry gen_kernel_registry.cpp)
target_compile_features(gen_kernel_registry PRIVATE cxx_std_20)
