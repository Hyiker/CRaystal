#pragma once

#include <memory>
#include <vector>

#include "Core/BVH.h"
#include "Core/Buffer.h"
#include "Core/Camera.h"
#include "Core/Intersection.h"
#include "Core/Macros.h"
#include "Core/Material.h"
#include "Core/Object.h"
#include "Shape.h"
#include "Sphere.h"
#include "Triangle.h"

namespace CRay {

using AccelerationStructure = BVH;

struct CRAYSTAL_API SceneView {
    // Geometry data
    // Use SOA structs
    SphereSOA sphereSOA;
    TriangleMeshSOA meshSOA;

    AccelerationStructure::DeviceView acceleration;

    // Material data
    MaterialView materialSystem;

    CRAYSTAL_DEVICE bool intersect(RayHit& rayHit) const;

    CRAYSTAL_DEVICE Intersection createIntersection(const RayHit& rayHit) const;
};

struct SceneData {
    std::vector<SphereData> spheres;
    std::vector<MeshData> meshes;

    std::vector<MaterialData> materials;
    std::vector<uint32_t> emissiveIndex;
};

class CRAYSTAL_API Scene {
   public:
    using Ref = std::shared_ptr<Scene>;
    Scene(SceneData&& data);

    SceneView* getDeviceView();

    Camera::Ref getCamera() { return mpCamera; }

    void setCamera(const Camera::Ref& pCamera) { mpCamera = pCamera; }

   private:
    AccelerationStructure::Ref mpAcceleration;

    Camera::Ref mpCamera;

    // Host data
    SceneView mSceneView;

    // Shape managers
    SphereManager::Ref mpSphereManager;
    TriangleMeshManager::Ref mpMeshManager;

    // Material managers
    MaterialManager::Ref mpMaterialManager;

    // Device data
    std::unique_ptr<DeviceBuffer> mpDeviceSceneView;
};

}  // namespace CRay
