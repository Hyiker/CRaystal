#pragma once

#include <memory>
#include <vector>

#include "Core/Buffer.h"
#include "Core/Camera.h"
#include "Core/Intersection.h"
#include "Core/Macros.h"
#include "Core/Object.h"
#include "Shape.h"
#include "Sphere.h"
#include "Triangle.h"

namespace CRay {

struct CRAYSTAL_API SceneView {
    // Geometry data
    // Use SOA structs
    SphereSOA sphereSOA;
    TriangleMeshSOA meshSOA;

    CRAYSTAL_DEVICE bool intersect(RayHit& rayHit) const;

    CRAYSTAL_DEVICE Intersection createIntersection(const RayHit& rayHit) const;
};

struct SceneData {
    std::vector<SphereData> spheres;
    std::vector<MeshData> meshes;
};

class CRAYSTAL_API Scene {
   public:
    using Ref = std::shared_ptr<Scene>;
    Scene(SceneData&& data);

    SceneView* getDeviceView();

    Camera::Ref getCamera() { return mpCamera; }

    void setCamera(const Camera::Ref& pCamera) { mpCamera = pCamera; }

   private:
    Camera::Ref mpCamera;

    // Host data
    SceneView mSceneView;

    // Shape managers
    SphereManager::Ref mpSphereManager;
    TriangleMeshManager::Ref mpMeshManager;

    // Device data
    std::unique_ptr<DeviceBuffer> mpDeviceSceneView;
};

}  // namespace CRay
