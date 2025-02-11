#include "Core/Sampler.h"
#include "Integrator.h"
#include "Utils/Progress.h"

namespace CRay {
PathTraceIntegrator::PathTraceIntegrator() {
    mpConstDataBuffer = std::make_unique<DeviceBuffer>(sizeof(DeviceView));
}

__global__ void pathTraceKernel(uint32_t frameIdx,
                                const PathTraceIntegratorView* pIntegrator,
                                const SceneView* pScene,
                                const CameraProxy* pCamera,
                                SensorData* pSensor) {
    UInt2 xy(blockIdx.x * blockDim.x + threadIdx.x,
             blockIdx.y * blockDim.y + threadIdx.y);

    UInt2 sensorSize = pSensor->size;
    if (xy.x >= sensorSize.x || xy.y >= sensorSize.y) {
        return;
    }

    Sampler sampler(xy, frameIdx);
    Float2 pixel = Float2(xy) + sampler.nextSample2D();

    auto ray = pCamera->generateRay(sensorSize, pixel);

    Spectrum color;

    RayHit rayHit;
    rayHit.ray = ray;
    if (pScene->intersect(rayHit)) {
        const Intersection it = pScene->createIntersection(rayHit);
        uint32_t materialID =
            pScene->meshSOA.getMeshDesc(rayHit.hitInfo.primitiveIndex)
                .materialID;

        MaterialData materialData =
            pScene->materialSystem.getMaterialData(materialID);
        if (materialData.isEmissive()) {
            color = Spectrum(materialData.emission);
        } else {
            color = Spectrum(materialData.diffuseRefl);
        }
    }

    pSensor->addSample(color, pixel);
}

PathTraceIntegratorView* PathTraceIntegrator::getDeviceView() const {
    return (PathTraceIntegratorView*)mpConstDataBuffer->data();
}

void PathTraceIntegrator::dispatch(Scene& scene, int spp) const {
    mpConstDataBuffer->copyFromHost(&mView);

    auto pCamera = scene.getCamera();
    auto pSensor = pCamera->getSensor();

    pSensor->setSPP(spp);

    pSensor->updateDeviceData();
    pCamera->updateDeviceData();

    UInt2 size = pSensor->getSize();

    for (int i : Progress(pSensor->getSPP(), "Render progress ")) {
        pathTraceKernel<<<dim3(size.x, size.y, 1), dim3(16, 16, 1)>>>(
            i, getDeviceView(), scene.getDeviceView(), pCamera->getDeviceView(),
            pSensor->getDeviceView());

        cudaDeviceSynchronize();
    }

    pSensor->readbackDeviceData();
}

}  // namespace CRay
