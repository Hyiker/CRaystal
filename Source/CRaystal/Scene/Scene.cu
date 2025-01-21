#include "Math/MathDefs.h"
#include "Scene.h"
namespace CRay {

bool SceneView::intersect(RayHit& rayHit) const {
    rayHit.hitT = kFltInf;
    bool intersected = false;
    for (uint32_t i = 0; i < sphereCount; i++) {
        intersected |= intersectShape(PrimitiveID(i), sphereData[i], rayHit);
    }
    return intersected;
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
