#include "Math/MathDefs.h"
#include "Scene.h"
namespace CRay {

CRAYSTAL_DEVICE bool SceneView::intersect(RayHit& rayHit) const {
    bool intersected = false;
    for (uint32_t i = 0; i < sphereSOA.count; i++) {
        intersected |= intersectShape(PrimitiveID(i), sphereSOA, rayHit);
    }

    intersected |= acceleration.intersect(meshSOA, rayHit);

    return intersected;
}

CRAYSTAL_DEVICE Intersection
SceneView::createIntersection(const RayHit& rayHit) const {
    const Ray& ray = rayHit.ray;

    Float3 posW = ray.origin + rayHit.hitT * ray.direction;
    Float3 viewW = -normalize(ray.direction);
    Float3 faceNormal, shadingNormal;

    const HitInfo& hit = rayHit.hitInfo;
    Float3 barycentric = hit.getBarycentricWeights();

    switch (hit.type) {
        case HitType::Sphere: {
            SphereData sphere = sphereSOA.getSphere(hit.primitiveIndex);
            faceNormal = normalize(posW - sphere.center);
            shadingNormal = faceNormal;
        } break;
        case HitType::Triangle: {
            TriangleData triangle = meshSOA.getTriangle(hit.primitiveIndex);
            faceNormal = triangle.getFaceNormal();
            shadingNormal = triangle.vData[0].normal * barycentric.x +
                            triangle.vData[1].normal * barycentric.y +
                            triangle.vData[2].normal * barycentric.z;
        } break;
    }

    return Intersection(posW, faceNormal, shadingNormal, viewW);
}

Scene::Scene(SceneData&& data) {
    mpAcceleration = std::make_shared<AccelerationStructure>();
    mpAcceleration->build(data);

    mpSphereManager = std::make_shared<SphereManager>(data.spheres);
    mpMeshManager = std::make_shared<TriangleMeshManager>(data.meshes);
    mpMaterialManager = std::make_shared<MaterialManager>(data.materials);

    mpDeviceSceneView = std::make_unique<DeviceBuffer>(sizeof(SceneView));

    mSceneView.sphereSOA = mpSphereManager->getDeviceView();
    mSceneView.meshSOA = mpMeshManager->getDeviceView();
    mSceneView.acceleration = mpAcceleration->getDeviceView();
    mSceneView.materialSystem = mpMaterialManager->getDeviceView();

    mpDeviceSceneView->copyFromHost(&mSceneView);
}

SceneView* Scene::getDeviceView() {
    return (SceneView*)mpDeviceSceneView->data();
}

}  // namespace CRay
