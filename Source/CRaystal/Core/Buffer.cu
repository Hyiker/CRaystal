#include <cuda_runtime.h>

#include "Buffer.h"
#include "Utils/CudaUtils.h"

namespace CRay {
DeviceBuffer::DeviceBuffer(int sizeInBytes)
    : mpDeviceData(nullptr), mSize(sizeInBytes) {
    if (sizeInBytes != 0)
        CRAYSTAL_CUDA_CHECK(cudaMalloc(&mpDeviceData, sizeInBytes));
}

DeviceBuffer::~DeviceBuffer() {
    free();
    mpDeviceData = nullptr;
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : mpDeviceData(other.mpDeviceData), mSize(other.mSize) {
    other.mpDeviceData = nullptr;
    other.mSize = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
        if (mpDeviceData) {
            free();
        }
        mpDeviceData = other.mpDeviceData;
        mSize = other.mSize;
        other.mpDeviceData = nullptr;
        other.mSize = 0;
    }
    return *this;
}

void* DeviceBuffer::data() { return mpDeviceData; }

const void* DeviceBuffer::data() const { return mpDeviceData; }

int DeviceBuffer::size() const { return mSize; }

void DeviceBuffer::copyFromHost(const void* pHostData) {
    if (mpDeviceData)
        CRAYSTAL_CUDA_CHECK(
            cudaMemcpy(mpDeviceData, pHostData, mSize, cudaMemcpyHostToDevice));
}

void DeviceBuffer::copyToHost(void* pHostData) const {
    if (mpDeviceData)
        CRAYSTAL_CUDA_CHECK(
            cudaMemcpy(pHostData, mpDeviceData, mSize, cudaMemcpyDeviceToHost));
}

void DeviceBuffer::memset(unsigned char value) {
    if (mpDeviceData)
        CRAYSTAL_CUDA_CHECK(cudaMemset(mpDeviceData, value, mSize));
}

void DeviceBuffer::free() {
    if (mpDeviceData) CRAYSTAL_CUDA_CHECK(cudaFree(mpDeviceData));
    mpDeviceData = nullptr;
}

}  // namespace CRay
