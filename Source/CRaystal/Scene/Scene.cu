#include "Math/MathDefs.h"
#include "Scene.h"
namespace CRay {

CRAYSTAL_DEVICE bool SceneView::intersect(RayHit& rayHit) const {
    bool intersected = false;
    for (uint32_t i = 0; i < sphereCount; i++) {
        intersected |= intersectShape(PrimitiveID(i), sphereData[i], rayHit);
    }
    return intersected;
}

CRAYSTAL_DEVICE const Sphere& SceneView::getSphere(
    PrimitiveID primitiveID) const {
    return sphereData[primitiveID];
}

CRAYSTAL_DEVICE Intersection
SceneView::createIntersection(const RayHit& rayHit) const {
    const Ray& ray = rayHit.ray;

    Float3 posW = ray.origin + rayHit.hitT * ray.direction;
    Float3 viewW = -normalize(ray.direction);
    Float3 faceNormal;

    const HitInfo& hit = rayHit.hitInfo;

    switch (hit.type) {
        case HitType::Sphere: {
            const Sphere& sphere = getSphere(hit.primitiveIndex);
            faceNormal = normalize(posW - sphere.center);
        } break;
    }

    return Intersection(posW, faceNormal, faceNormal, viewW);
}

Scene::Scene() {
    mpDeviceSceneView = std::make_unique<DeviceBuffer>(sizeof(SceneView));
}

SceneView* Scene::getDeviceView() {
    return (SceneView*)mpDeviceSceneView->data();
}

void Scene::finalize() {
    if (mSceneView.sphereCount) {
        mpDeviceSphereData = std::make_unique<DeviceBuffer>(
            mSceneView.sphereCount * sizeof(Sphere));
        mSceneView.sphereData = (Sphere*)mpDeviceSphereData->data();
    }
}

void Scene::updateDeviceData() const {
    mpDeviceSceneView->copyFromHost(&mSceneView);
    mpDeviceSphereData->copyFromHost(mSphereData.data());
}

void Scene::addSphere(Sphere sphere) {
    mSceneView.sphereCount++;
    mSphereData.push_back(std::move(sphere));
}

}  // namespace CRay
