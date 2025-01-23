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
    // Geometry data
    // Use SOA structs
    SphereSOA sphereSOA;

    CRAYSTAL_DEVICE bool intersect(RayHit& rayHit) const;

    CRAYSTAL_DEVICE Intersection createIntersection(const RayHit& rayHit) const;
};

struct SceneData {
    std::vector<SphereData> spheres;
};

class CRAYSTAL_API Scene {
   public:
    Scene(SceneData&& data);

    SceneView* getDeviceView();

   private:
    // Host data
    SceneView mSceneView;

    // Shape managers
    SphereManager::Ref mpSphereManager;

    // Device data
    std::unique_ptr<DeviceBuffer> mpDeviceSceneView;
};
}  // namespace CRay
