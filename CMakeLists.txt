cmake_minimum_required(VERSION 3.15...3.30)
project(
    CRaystal
    VERSION 0.0.1
    LANGUAGES C CXX CUDA
)

# -----------------------------------------------------------------------------
# Banner
# -----------------------------------------------------------------------------
message("-----------------------------------------------------------------------------")
message("-                                 CRAYSTAL                                  -")
message("-----------------------------------------------------------------------------")

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
set(CRAYSTAL_ENABLE_F64 OFF CACHE BOOL "Use 64-bit double for arithmetic instead of 32-bit float")
set(CRAYSTAL_ENABLE_SPECTRUM ON CACHE BOOL "Enable spectrum rendering")

# -----------------------------------------------------------------------------
# Check platform
# -----------------------------------------------------------------------------
if(${CMAKE_SYSTEM_NAME} MATCHES "Window")
    set(CRAYSTAL_PLATFORM "Windows")
    set(CRAYSTAL_WINDOWS TRUE)
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(CRAYSTAL_PLATFORM "Linux")
    set(CRAYSTAL_LINUX TRUE)
else()
    message(FATAL_ERROR "Unsupported platform!")
endif()

message(STATUS "Platform: ${CRAYSTAL_PLATFORM}")

# -----------------------------------------------------------------------------
# CUDA
# -----------------------------------------------------------------------------
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    message(STATUS "Enabling CUDA support.")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CRAYSTAL_HAS_CUDA ON)
else()
    set(CRAYSTAL_HAS_CUDA OFF)
endif()

set(CRAYSTAL_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CRAYSTAL_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# -----------------------------------------------------------------------------
# External dependencies
# -----------------------------------------------------------------------------
add_subdirectory(External)

message(STATUS "CRAYSTAL_HAS_CUDA: ${CRAYSTAL_HAS_CUDA}")
message(STATUS "CRAYSTAL_ENABLE_F64: ${CRAYSTAL_ENABLE_F64}")
message(STATUS "CRAYSTAL_ENABLE_SPECTRUM: ${CRAYSTAL_ENABLE_SPECTRUM}")

function(add_craystal_executable target)
    add_executable(${target})

    target_link_libraries(${target} PRIVATE CRaystal)

    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CRAYSTAL_RUNTIME_OUTPUT_DIRECTORY})

    target_compile_definitions(${target} PRIVATE
        "CRAYSTAL_EXECUTABLE=1"
    )
endfunction()

# -----------------------------------------------------------------------------
# Sources
# -----------------------------------------------------------------------------
add_subdirectory(Source/CRaystal)
add_subdirectory(Source/CRayli)
