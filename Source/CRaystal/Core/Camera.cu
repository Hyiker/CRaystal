#include <cuda.h>

#include "Camera.h"
#include "Error.h"
namespace CRay {

Camera::Camera() : mpDeviceData(nullptr) {
    cudaMalloc(&mpDeviceData, sizeof(CameraProxy));

    updateDeviceData();
}

Camera::Camera(Camera&& other) noexcept
    : mData(std::move(other.mData)), mpDeviceData(other.mpDeviceData) {
    other.mpDeviceData = nullptr;
}

Camera& Camera::operator=(Camera&& other) noexcept {
    if (this != &other) {
        if (mpDeviceData) {
            cudaFree(mpDeviceData);
            mpDeviceData = nullptr;
        }

        mData = std::move(other.mData);

        mpDeviceData = other.mpDeviceData;
        other.mpDeviceData = nullptr;
    }
    return *this;
}

void Camera::calculateCameraData() const {
    mData.cameraW = normalize(mData.target - mData.posW);
    mData.cameraU = normalize(cross(mData.cameraW, mData.up));
    mData.cameraV = normalize(cross(mData.cameraU, mData.cameraW));
}

void Camera::updateDeviceData() const {
    CRAYSTAL_CHECK(mpDeviceData != nullptr, "Cuda pointer is none");
    cudaMemcpy(mpDeviceData, &mData, sizeof(CameraProxy),
               cudaMemcpyHostToDevice);
}

Camera::~Camera() {
    if (mpDeviceData) {
        cudaFree(mpDeviceData);
        mpDeviceData = nullptr;
    }
}
}  // namespace CRay
