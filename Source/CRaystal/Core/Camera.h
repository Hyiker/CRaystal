#pragma once
#include "CameraProxy.h"
#include "Macros.h"
#include "Object.h"
namespace CRay {

/** Host side camera class.
 */
class CRAYSTAL_API Camera : public HostObject {
   public:
    Camera();
    Camera(const Camera& other) = delete;
    Camera(Camera&& other) noexcept;
    Camera& operator=(const Camera& other) = delete;
    Camera& operator=(Camera&& other) noexcept;

    void setSensorSize(const UInt2& size) {
        mData.sensorWidth = size.x;
        mData.sensorHeight = size.y;
    }

    void setFovY(Float value) { mData.fovY = value; }

    void setWorldPosition(const Float3& posW) { mData.posW = posW; }

    void setTarget(const Float3& target) { mData.target = target; }

    void setUp(const Float3& up) { mData.up = up; }

    const CameraProxy& getData() const { return mData; }

    void calculateCameraData() const;

    void updateDeviceData() const override;

    ~Camera();

   private:
    mutable CameraProxy mData;

    CameraProxy* mpDeviceData;
};

}  // namespace CRay
