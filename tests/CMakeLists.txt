include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)
# For Apple Silicon and other environments gtest may try to use system install.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

enable_testing()

file(GLOB_RECURSE ORTEAF_TEST_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

if(ORTEAF_TEST_SOURCES)
    add_executable(orteaf_tests ${ORTEAF_TEST_SOURCES})
    target_link_libraries(orteaf_tests
        PRIVATE
            orteaf
            GTest::gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(orteaf_tests)
endif()
