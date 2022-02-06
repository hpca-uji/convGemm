# Source CmakeList.txt

# ========================================================================
# convGemm library compilation, linking and installation
# ========================================================================

# Create a SHARED version of convGemm library from the given sources
add_library(convGemm SHARED convGemm.c)
target_sources(convGemm PRIVATE ${COMMON_SOURCES})

# Add link to the Blis library and propagate this link requirement to dependant targets (PUBLIC option)
target_link_libraries(convGemm PUBLIC ${BLIS_LIBRARY} OpenMP::OpenMP_CXX)

# Add compile options for convGemm library and propagate them to dependant targets (PUBLIC option)
target_compile_options(convGemm PUBLIC
        -Ofast -mtune=native -march=native
        -ftree-vectorize -fopt-info-vec-optimized
        -fopenmp
        -D BLIS_ABI_VERSION=${BLIS_ABI_VERSION}
        -Wall
        )

# Project directories that will be propagated to dependant targets (PUBLIC)
target_include_directories(
        convGemm PUBLIC
        "${PROJECT_SOURCE_DIRECTORY}/src"
        "${PROJECT_SOURCE_DIRECTORY}/test"
)

# System directories that will be propagated to dependant targets (PUBLIC)
# There will be no compilation warnings for the headers in these directories (SYSTEM)
target_include_directories(
        convGemm SYSTEM PUBLIC
        "${CMAKE_PREFIX_PATH}/include"
        "${BLIS_LIBRARY_PATH}/../include"
        "${CMAKE_INSTALL_PREFIX}/include"
)

install(TARGETS convGemm DESTINATION lib)
install(FILES "${PROJECT_SOURCE_DIR}/src/convGemm.h" DESTINATION include)
