
#include "CRaystal.h"
#include "Utils/Progress.h"
#include "Walkthrough.h"

namespace CRay {

__global__ void renderFrame(const SceneView* pScene, const CameraProxy* pCamera,
                            SensorData* pSensor) {
    UInt2 xy(blockIdx.x * blockDim.x + threadIdx.x,
             blockIdx.y * blockDim.y + threadIdx.y);

    UInt2 sensorSize = pSensor->size;
    if (xy.x >= sensorSize.x || xy.y >= sensorSize.y) {
        return;
    }

    Float2 pixel = Float2(xy) + Float2(0.5);

    auto ray = pCamera->generateRay(sensorSize, pixel);
    Spectrum color;

    RayHit rayHit;
    rayHit.ray = ray;
    if (pScene->intersect(rayHit)) {
        const Intersection it = pScene->createIntersection(rayHit);

        color = Spectrum(it.getOrientedFaceNormal());
    }

    pSensor->addSample(color, pixel);
}

void crayRenderSample(const Scene::Ref& pScene) {
    auto pCamera = pScene->getCamera();
    auto pSensor = pCamera->getSensor();

    pSensor->updateDeviceData();
    pCamera->updateDeviceData();

    UInt2 size = pSensor->getSize();

    for (int i : Progress(pSensor->getSPP(), "Render progress ")) {
        renderFrame<<<dim3(size.x, size.y, 1), dim3(16, 16, 1)>>>(
            pScene->getDeviceView(), pCamera->getDeviceView(),
            pSensor->getDeviceView());

        cudaDeviceSynchronize();
    }

    pSensor->readbackDeviceData();
    auto pImage = pSensor->createImage();
    pImage->writeEXR("walkthrough.exr");
}

}  // namespace CRay
