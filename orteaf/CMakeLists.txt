set(ORTEAF_SOURCE_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(ORTEAF_INCLUDE_ROOT ${ORTEAF_SOURCE_ROOT}/include/orteaf)

set(ORTEAF_USER_INCLUDE_DIR ${ORTEAF_INCLUDE_ROOT}/user)
set(ORTEAF_EXTENSION_INCLUDE_DIR ${ORTEAF_INCLUDE_ROOT}/extension)
set(ORTEAF_INTERNAL_INCLUDE_DIR ${ORTEAF_INCLUDE_ROOT}/internal)

set(ORTEAF_RUNTIME_STATS_LEVEL 0 CACHE STRING
    "Default statistics level for runtimes and allocators (0=off, 1=basic, 2=extended)")
set_property(CACHE ORTEAF_RUNTIME_STATS_LEVEL PROPERTY STRINGS 0 1 2)

foreach(_component CPU CUDA MPS ALLOCATOR)
    set(ORTEAF_${_component}_STATS_LEVEL "AUTO" CACHE STRING
        "Statistics level override for ${_component} (AUTO inherits ORTEAF_RUNTIME_STATS_LEVEL)")
    set_property(CACHE ORTEAF_${_component}_STATS_LEVEL PROPERTY STRINGS AUTO 0 1 2)
endforeach()

function(orteaf_resolve_stats_level out_var override_value)
    if("${override_value}" STREQUAL "AUTO")
        set(${out_var} ${ORTEAF_RUNTIME_STATS_LEVEL} PARENT_SCOPE)
    else()
        set(${out_var} ${override_value} PARENT_SCOPE)
    endif()
endfunction()

orteaf_resolve_stats_level(ORTEAF_CPU_STATS_LEVEL_EFFECTIVE "${ORTEAF_CPU_STATS_LEVEL}")
orteaf_resolve_stats_level(ORTEAF_CUDA_STATS_LEVEL_EFFECTIVE "${ORTEAF_CUDA_STATS_LEVEL}")
orteaf_resolve_stats_level(ORTEAF_MPS_STATS_LEVEL_EFFECTIVE "${ORTEAF_MPS_STATS_LEVEL}")
orteaf_resolve_stats_level(ORTEAF_ALLOCATOR_STATS_LEVEL_EFFECTIVE "${ORTEAF_ALLOCATOR_STATS_LEVEL}")

add_library(orteaf_user INTERFACE)
target_include_directories(orteaf_user INTERFACE ${ORTEAF_USER_INCLUDE_DIR})

add_library(orteaf_extension INTERFACE)
target_include_directories(orteaf_extension INTERFACE
    ${ORTEAF_EXTENSION_INCLUDE_DIR}
    ${ORTEAF_USER_INCLUDE_DIR}
)
target_link_libraries(orteaf_extension INTERFACE orteaf_user)
target_compile_definitions(orteaf_extension INTERFACE
    ORTEAF_RUNTIME_STATS_LEVEL=${ORTEAF_RUNTIME_STATS_LEVEL}
    ORTEAF_CPU_STATS_LEVEL=${ORTEAF_CPU_STATS_LEVEL_EFFECTIVE}
    ORTEAF_CUDA_STATS_LEVEL=${ORTEAF_CUDA_STATS_LEVEL_EFFECTIVE}
    ORTEAF_MPS_STATS_LEVEL=${ORTEAF_MPS_STATS_LEVEL_EFFECTIVE}
    ORTEAF_ALLOCATOR_STATS_LEVEL=${ORTEAF_ALLOCATOR_STATS_LEVEL_EFFECTIVE}
)

file(GLOB_RECURSE ORTEAF_SOURCES CONFIGURE_DEPENDS
    ${ORTEAF_SOURCE_ROOT}/src/*.cpp
)

if(ORTEAF_SOURCES)
    add_library(orteaf STATIC ${ORTEAF_SOURCES})
else()
    add_library(orteaf STATIC)
endif()

target_link_libraries(orteaf
    PUBLIC
        orteaf_extension
)

target_include_directories(orteaf
    PRIVATE
        ${ORTEAF_INTERNAL_INCLUDE_DIR}
)
target_compile_definitions(orteaf
    PUBLIC
        ORTEAF_RUNTIME_STATS_LEVEL=${ORTEAF_RUNTIME_STATS_LEVEL}
        ORTEAF_CPU_STATS_LEVEL=${ORTEAF_CPU_STATS_LEVEL_EFFECTIVE}
        ORTEAF_CUDA_STATS_LEVEL=${ORTEAF_CUDA_STATS_LEVEL_EFFECTIVE}
        ORTEAF_MPS_STATS_LEVEL=${ORTEAF_MPS_STATS_LEVEL_EFFECTIVE}
        ORTEAF_ALLOCATOR_STATS_LEVEL=${ORTEAF_ALLOCATOR_STATS_LEVEL_EFFECTIVE}
)

target_compile_features(orteaf PUBLIC cxx_std_20)
