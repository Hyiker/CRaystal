#include "Helloworld.h"
#include <fmt/format.h>
__global__ void helloWorldKernel(){
    uint32_t instanceIndex = blockDim.x * blockIdx.x + threadIdx.x;

}

void cuHelloWorld(){
    helloWorldKernel<<<4, 1, 1>>>();
    fmt::println("Cuda kernel hello world: {}", 1);
}

