# CMake package manager
include(CPM.cmake)

# fmt
message(STATUS "Configure libfmt")
CPMAddPackage(
    NAME fmt
    GIT_TAG 11.1.1
    GITHUB_REPOSITORY fmtlib/fmt
    PATCHES fmt.patch
    OPTIONS
    "FMT_DOC OFF"
    "FMT_TEST OFF"
    "FMT_UNICODE ON"
)
set_target_properties(fmt PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    FOLDER "Libraries"
)
