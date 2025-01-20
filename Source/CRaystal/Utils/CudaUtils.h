#pragma once
#include "Core/Error.h"
#define CRAYSTAL_CUDA_CHECK(call)                                            \
    {                                                                        \
        cudaError_t result = call;                                           \
        if (result != cudaSuccess) {                                         \
            const char* errorName = cudaGetErrorName(result);                \
            const char* errorString = cudaGetErrorString(result);            \
            CRAYSTAL_THROW("CUDA call {} failed with error {} ({}).", #call, \
                           errorName, errorString);                          \
        }                                                                    \
    }
