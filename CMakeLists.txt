# convGemmNHWC CMakeLists.txt

cmake_minimum_required(VERSION 3.10)

project(ConvGemmNHWC VERSION 1.0)

find_package(OpenMP REQUIRED)

# ========================================================================
# Find the BLIS library
# ------------------------------------------------------------------------
# Convert $ENV{LD_LIBRARY_PATH} to a list of paths (using ; as separator)
if (DEFINED ENV{LD_LIBRARY_PATH})
    string(REPLACE ":" ";" LIBRARY_DIRS $ENV{LD_LIBRARY_PATH})
endif ()
# Do find the library
find_library(
        BLIS_LIBRARY
        NAMES libblis.so.4 libblis.so.3
        HINTS
        ${CMAKE_PREFIX_PATH}/lib
        ${CMAKE_INSTALL_PREFIX}/lib
        ${LIBRARY_DIRS})
if (BLIS_LIBRARY)
    message(STATUS "Linking against the BLIS library in '${BLIS_LIBRARY}'")
    # Set BLIS_LIBRARY_NAME, BLIS_LIBRARY_PATH and BLIS_ABI_VERSION
    get_filename_component(BLIS_LIBRARY_NAME ${BLIS_LIBRARY} NAME)
    get_filename_component(BLIS_LIBRARY_PATH ${BLIS_LIBRARY} DIRECTORY)
    string(SUBSTRING ${BLIS_LIBRARY_NAME} 11 1 BLIS_ABI_VERSION)
else ()
    message(FATAL_ERROR
            "Could not find the BLIS library, please read the README.md file.\n"
            "The BLIS library was searched in:\n"
            "  - CMAKE_PREFIX_PATH/lib/: ${CMAKE_PREFIX_PATH}/lib/\n"
            "  - CMAKE_INSTALL_PREFIX/lib/: ${CMAKE_INSTALL_PREFIX}/lib/\n"
            "  - \${LD_LIBRARY_PATH}: $ENV{LD_LIBRARY_PATH}"
            )
endif ()
# ========================================================================


# ========================================================================
# common sources to the convGemm library and the test executables
# ------------------------------------------------------------------------
set(COMMON_SOURCES gemm_blis.c;gemm_blis_B3A2C0.c;gemm_blis_A3B2C0.c;gemm_blis_B3A2C0_orig.c;im2row_nhwc.c;im2col_nchw.c)
string(REGEX REPLACE "([^;]+)" "${PROJECT_SOURCE_DIR}/src/\\1" COMMON_SOURCES "${COMMON_SOURCES}")
# ========================================================================

add_subdirectory(src)

option(COMPILE_TESTS "Compile tests" OFF)

#if (COMPILE_TESTS)
    add_subdirectory(tests)
#endif ()
