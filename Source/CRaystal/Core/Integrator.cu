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

    Spectrum radiance(0.f);  ///< Radiance carried by the ray.
    Spectrum beta(1.f);      ///< The accumulated attenuation factor.

    RayHit primaryRayHit;
    for (uint32_t depth = 0u; depth < pIntegrator->maxDepth; depth++) {
        primaryRayHit = RayHit();
        primaryRayHit.ray = ray;

        bool terminatePath = false;
        if (pScene->intersect(primaryRayHit)) {
            const Intersection it = pScene->createIntersection(primaryRayHit);
            uint32_t materialID =
                pScene->meshSOA
                    .getMeshDesc(primaryRayHit.hitInfo.primitiveIndex)
                    .materialID;

            MaterialData materialData =
                pScene->materialSystem.getMaterialData(materialID);
            if (materialData.isEmissive() && it.isFrontFacing) {
                radiance += materialData.emission * beta;
            }

            BSDF bsdf = getBSDF(materialData, it.frame);

            Float3 wi;
            Float pdf;
            auto f = bsdf.sampleEvaluate(sampler, it.viewW, wi, pdf);

            beta *= f / pdf;

            ray.origin = it.posW + Float(1e-5) * it.getOrientedFaceNormal();
            ray.direction = wi;

            terminatePath |= beta.maxValue() < 1e-6f || pdf == 0.0;

        } else {
            terminatePath = true;
        }

        if (sampler.nextSample1D() < pIntegrator->rrThreshold) {
            terminatePath = true;
        }

        if (terminatePath)
            break;
        else
            beta /= 1.0 - pIntegrator->rrThreshold;
    }

    pSensor->addSample(radiance, pixel);
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
