file(GLOB_RECURSE ORTEAF_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_library(orteaf STATIC ${ORTEAF_SOURCES})

target_include_directories(orteaf
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(orteaf PUBLIC cxx_std_17)
