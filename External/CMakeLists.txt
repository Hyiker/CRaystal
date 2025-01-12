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
)
set_target_properties(fmt PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    FOLDER "Libraries"
)

# glm
message(STATUS "Configure glm")
CPMAddPackage(
    NAME glm
    GIT_TAG 1.0.1
    GITHUB_REPOSITORY g-truc/glm
    OPTIONS
    "GLM_ENABLE_CXX_20 ON"
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
