#pragma once

#if CRAYSTAL_ENABLE_F64
using Float = double;
#else
using Float = float;
#endif

// Compilers
#define CRAYSTAL_COMPILER_NVCC 1
#define CRAYSTAL_COMPILER_CLANG 2
#define CRAYSTAL_COMPILER_GCC 3
#define CRAYSTAL_COMPILER_MSVC 4

#ifndef CRAYSTAL_COMPILER
#ifdef __CUDACC__
#define CRAYSTAL_COMPILER CRAYSTAL_COMPILER_NVCC
#elif defined(__clang__)
#define CRAYSTAL_COMPILER CRAYSTAL_COMPILER_CLANG
#elif defined(__GNUC__) || defined(__GNUG__)
#define CRAYSTAL_COMPILER CRAYSTAL_COMPILER_GCC
#elif defined(_MSC_VER)
#define CRAYSTAL_COMPILER CRAYSTAL_COMPILER_MSVC
#else
#error "Unsupported compiler"
#endif
#endif

#define CRAYSTAL_NVCC (CRAYSTAL_COMPILER == CRAYSTAL_COMPILER_NVCC)
#define CRAYSTAL_CLANG (CRAYSTAL_COMPILER == CRAYSTAL_COMPILER_CLANG)
#define CRAYSTAL_GCC (CRAYSTAL_COMPILER == CRAYSTAL_COMPILER_GCC)
#define CRAYSTAL_MSVC (CRAYSTAL_COMPILER == CRAYSTAL_COMPILER_MSVC)

// Platform
#define CRAYSTAL_PLATFORM_WINDOWS 1
#define CRAYSTAL_PLATFORM_LINUX 2

// dynamic library export/import
#ifndef CRAYSTAL_PLATFORM
#if defined(_WIN64)
#define CRAYSTAL_PLATFORM CRAYSTAL_PLATFORM_WINDOWS
#elif defined(__linux__)
#define CRAYSTAL_PLATFORM CRAYSTAL_PLATFORM_LINUX
#else
#error "Unsupported target platform"
#endif
#endif

#define CRAYSTAL_WINDOWS (CRAYSTAL_PLATFORM == CRAYSTAL_PLATFORM_WINDOWS)
#define CRAYSTAL_LINUX (CRAYSTAL_PLATFORM == CRAYSTAL_PLATFORM_LINUX)

// Spectrum
#ifdef CRAYSTAL_ENABLE_SPECTRUM
#define CRAYSTAL_SPECTRUM 1
#else
#define CRAYSTAL_SPECTRUM 0
#endif

// Dylib
#if CRAYSTAL_WINDOWS
#define CRAYSTAL_API_EXPORT __declspec(dllexport)
#define CRAYSTAL_API_IMPORT __declspec(dllimport)
#elif CRAYSTAL_LINUX
#define CRAYSTAL_API_EXPORT __attribute__((visibility("default")))
#define CRAYSTAL_API_IMPORT
#endif

#ifdef CRAYSTAL_MODULE
#define CRAYSTAL_API CRAYSTAL_API_EXPORT
#else
#define CRAYSTAL_API CRAYSTAL_API_IMPORT
#endif

// Force inline
#if CRAYSTAL_MSVC  // MSVC
#define FORCE_INLINE __forceinline
#elif CRAYSTAL_NVCC
#define FORCE_INLINE __forceinline__
#elif CRAYSTAL_GCC || CRAYSTAL_CLANG
#define FORCE_INLINE inline __attribute__((always_inline))
#endif

// Cuda device host
#if CRAYSTAL_NVCC
#define CRAYSTAL_DEVICE __device__
#define CRAYSTAL_HOST __host__
#else
#define CRAYSTAL_DEVICE
#define CRAYSTAL_HOST
#endif

#define CRAYSTAL_DEVICE_HOST CRAYSTAL_DEVICE CRAYSTAL_HOST
