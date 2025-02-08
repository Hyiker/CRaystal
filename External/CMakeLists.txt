# CMake package manager
include(CPM.cmake)

# fmt
message(STATUS "Configure libfmt")
CPMAddPackage(
    NAME fmt
    GIT_TAG 11.1.1-cuda-patched
    GITHUB_REPOSITORY Hyiker/fmt
    OPTIONS
    "FMT_DOC OFF"
    "FMT_TEST OFF"
    "FMT_UNICODE ON"
    "BUILD_SHARED_LIBS OFF"
)

# glm
message(STATUS "Configure glm")
CPMAddPackage(
    NAME glm
    GIT_TAG 1.0.1
    GITHUB_REPOSITORY g-truc/glm
    OPTIONS
    "GLM_ENABLE_CXX_20 ON"
    "GLM_BUILD_LIBRARY ON"
    "BUILD_STATIC_LIBS ON"
    "BUILD_SHARED_LIBS OFF"
)

# catch2
message(STATUS "Configure Catch2")
CPMAddPackage(
    NAME Catch2
    VERSION 3.8.0
    GITHUB_REPOSITORY catchorg/Catch2
    OPTIONS
    "CATCH_INSTALL_DOCS OFF"
    "CATCH_INSTALL_EXTRAS OFF"
)
set_target_properties(Catch2 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CRAYSTAL_RUNTIME_OUTPUT_DIRECTORY})
set_target_properties(Catch2WithMain PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CRAYSTAL_RUNTIME_OUTPUT_DIRECTORY})

# tinyexr
message(STATUS "Configure Tinyexr")
CPMAddPackage(
    NAME tinyexr
    GIT_TAG cmake-fix
    GITHUB_REPOSITORY Hyiker/tinyexr
    OPTIONS
    "TINYEXR_BUILD_SAMPLE OFF"
    "TINYEXR_USE_MINIZ ON"
)

# assimp
message(STATUS "Configure Assimp")
CPMAddPackage(
    NAME assimp
    VERSION 5.4.3
    GITHUB_REPOSITORY assimp/assimp
    OPTIONS
    "ASSIMP_INJECT_DEBUG_POSTFIX OFF"
    "BUILD_SHARED_LIBS ON"
)
set_target_properties(assimp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CRAYSTAL_RUNTIME_OUTPUT_DIRECTORY})

# pugixml
message(STATUS "Configure pugixml")
CPMAddPackage(
    NAME pugixml
    VERSION 1.15
    GITHUB_REPOSITORY zeux/pugixml
    OPTIONS
    "PUGIXML_INSTALL OFF"
    "BUILD_SHARED_LIBS OFF"
)

# Indicators
message(STATUS "Configure indicators")
CPMAddPackage(
    NAME indicators
    VERSION 2.3
    GITHUB_REPOSITORY p-ranav/indicators
    OPTIONS
    "INDICATORS_BUILD_TESTS OFF"
    "INDICATORS_SAMPLES OFF"
)
