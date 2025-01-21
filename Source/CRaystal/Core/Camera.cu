#include <cuda.h>

#include "Camera.h"
#include "Error.h"
namespace CRay {

Camera::Camera() : mpDeviceData(nullptr) {
    mpDeviceData = std::make_unique<DeviceBuffer>(sizeof(CameraProxy));

    updateDeviceData();
}

Camera::Camera(Camera&& other) noexcept
    : mData(std::move(other.mData)),
      mpDeviceData(std::move(other.mpDeviceData)) {
    other.mpDeviceData = nullptr;
}

Camera& Camera::operator=(Camera&& other) noexcept {
    if (this != &other) {
        mData = std::move(other.mData);
        mpDeviceData = std::move(other.mpDeviceData);
        other.mpDeviceData = nullptr;
    }
    return *this;
}

CameraProxy* Camera::getDeviceView() const {
    return (CameraProxy*)mpDeviceData->data();
}

void Camera::calculateCameraData() const {
    mData.cameraW = normalize(mData.target - mData.posW);
    mData.cameraU = normalize(cross(mData.cameraW, mData.up));
    mData.cameraV = normalize(cross(mData.cameraU, mData.cameraW));
}

void Camera::updateDeviceData() const {
    mpDeviceData->copyFromHost(&mData);
}

Camera::~Camera() = default;
}  // namespace CRay
