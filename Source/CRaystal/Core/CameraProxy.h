#pragma once
#include "Macros.h"
#include "Math/CRayMath.h"
#include "Math/Ray.h"
#include "Vec.h"

namespace CRay {

/** CPU/Cuda camera data proxy.
 */
struct CRAYSTAL_API CameraProxy {
    Float3p32 posW = Float3p32(0, 0, 0);     ///< World space position.
    Float fovY = 60.f;                       ///< Field of view in degrees.
    Float3p32 target = Float3p32(0, 0, -1);  ///< Look at target.
    Float nearZ = 0.01f;                     ///< Near z clip.
    Float3p32 up = Float3p32(0, 1, 0);       ///< Up axis.
    Float farZ = 10000.0f;                   ///< Far z clip.
    Float3p32 cameraU = Float3p32(0, 0, 1);  ///< Image space U(right) vector.
    uint32_t sensorWidth = 512u;             ///< Sensor film width.
    Float3p32 cameraV = Float3p32(0, 1, 0);  ///< Image space V(up) vector.
    uint32_t sensorHeight = 512u;            ///< Sensor film height.
    Float3p32 cameraW = Float3p32(1, 0, 0);  ///< Image space W(forward) vector.
    uint32_t _pad0;

    CRAYSTAL_DEVICE_HOST Ray generateRay(const Float2& sensorPos) const {
        Float u = (2.0f * (sensorPos.x / sensorWidth) - 1.0f);   // [-1, 1]
        Float v = (1.0f - 2.0f * (sensorPos.y / sensorHeight));  // [1, -1]

        Float aspectRatio =
            static_cast<Float>(sensorWidth) / static_cast<Float>(sensorHeight);

        Float halfHeight = std::tan(radians(fovY * 0.5));
        Float halfWidth = aspectRatio * halfHeight;

        Float3 rayDir = normalize(cameraU * (u * halfWidth) +
                                  cameraV * (v * halfHeight) + cameraW);

        return Ray(posW, rayDir);
    }
};

}  // namespace CRay
