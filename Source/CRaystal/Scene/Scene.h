#pragma once

#include <vector>

#include "Core/Buffer.h"
#include "Core/Intersection.h"
#include "Core/Macros.h"
#include "Core/Object.h"
#include "Shape.h"
#include "Sphere.h"

namespace CRay {

struct CRAYSTAL_API SceneView {
    // Geometry stats
    uint32_t sphereCount = 0u;  ///< Sphere primitive count.

    // Geometry data
    // Always use device pointer
    Sphere* sphereData;

    CRAYSTAL_DEVICE bool intersect(RayHit& rayHit) const;

    CRAYSTAL_DEVICE const Sphere& getSphere(PrimitiveID primitiveID) const;

    CRAYSTAL_DEVICE Intersection createIntersection(const RayHit& rayHit) const;
};

class CRAYSTAL_API Scene : public HostObject {
   public:
    Scene();

    void addSphere(Sphere sphere);

    /** Finalize the scene construction, create device buffers.
     */
    void finalize();

    SceneView* getDeviceView();

    void updateDeviceData() const override;

   private:
    // Host data
    SceneView mSceneView;
    std::vector<Sphere> mSphereData;

    // Device data
    std::unique_ptr<DeviceBuffer> mpDeviceSceneView;
    std::unique_ptr<DeviceBuffer> mpDeviceSphereData;
};
}  // namespace CRay
