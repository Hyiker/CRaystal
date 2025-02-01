#pragma once
#include "Buffer.h"
#include "CameraProxy.h"
#include "Macros.h"
#include "Object.h"
#include "Sensor.h"

namespace CRay {

/** Host side camera class.
 */
class CRAYSTAL_API Camera : public HostObject {
   public:
    using Ref = std::shared_ptr<Camera>;

    Camera();
    Camera(const Camera& other) = delete;
    Camera(Camera&& other) noexcept;
    Camera& operator=(const Camera& other) = delete;
    Camera& operator=(Camera&& other) noexcept;

    CameraProxy* getDeviceView() const;

    void setFovY(Float value) { mData.fovY = value; }

    void setWorldPosition(const Float3& posW) { mData.posW = posW; }

    void setTarget(const Float3& target) { mData.target = target; }

    void setUp(const Float3& up) { mData.up = up; }

    const CameraProxy& getData() const { return mData; }

    void setSensor(const Sensor::Ref& pSensor) { mpSensor = pSensor; }

    Sensor::Ref getSensor() { return mpSensor; }

    void calculateCameraData() const;

    void updateDeviceData() const override;

    ~Camera();

   private:
    Sensor::Ref mpSensor;

    mutable CameraProxy mData;

    std::unique_ptr<DeviceBuffer> mpDeviceData;
};

}  // namespace CRay
