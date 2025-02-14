#include "Core/Sampler.h"
#include "Integrator.h"
#include "Math/Sampling.h"
#include "Utils/Progress.h"
namespace CRay {
PathTraceIntegrator::PathTraceIntegrator() {
    mpConstDataBuffer = std::make_unique<DeviceBuffer>(sizeof(DeviceView));
}

CRAYSTAL_DEVICE Spectrum evaluateNEE(const SceneView& scene,
                                     const Intersection& intersection,
                                     const BSDF& bsdf, Sampler& sampler) {
    /** Sample light
     */
    const int emissiveCnt = scene.materialSystem.getEmissiveCount();
    if (emissiveCnt == 0) return Spectrum(0);
    PrimitiveID emissiveIndex =
        std::min<int>(emissiveCnt * sampler.nextSample1D(), emissiveCnt - 1);

    auto emissiveTriangle = scene.meshSOA.getTriangle(emissiveIndex);

    Float2 barycentric = sampleBarycentric(sampler.nextSample2D());
    auto vertexData = emissiveTriangle.interpolate(Float3(
        1.f - barycentric.x - barycentric.y, barycentric.x, barycentric.y));

    Float3 targetPos = vertexData.position;
    Float3 toLight = targetPos - intersection.posW;
    Float distSqr = dot(toLight, toLight);
    Float3 lightDir = toLight / std::sqrt(distSqr);
    Float pdf = 1.0 / emissiveTriangle.getArea();

    // Shadow test
    Ray shadowRay(intersection.posW, normalize(toLight));
    shadowRay.tMax = length(toLight) - kEps;
    if (scene.intersectOcclusion(shadowRay)) {
        return Spectrum(0);
    }

    Spectrum emission =
        scene.materialSystem
            .getMaterialData(
                scene.meshSOA.getMeshDesc(emissiveIndex).materialID)
            .emission;

    return bsdf.evaluate(intersection.viewW, lightDir) * emission *
           absDot(lightDir, vertexData.normal) / pdf / distSqr;
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

            radiance += evaluateNEE(*pScene, it, bsdf, sampler);

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
