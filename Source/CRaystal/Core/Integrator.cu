#include "Core/Sampler.h"
#include "Integrator.h"
#include "Material/BSDF.h"
#include "Math/Sampling.h"
#include "Utils/Progress.h"

namespace CRay {
PathTraceIntegrator::PathTraceIntegrator(Configs configs)
    : mConfigs(std::move(configs)) {
    mpConstDataBuffer = std::make_unique<DeviceBuffer>(sizeof(DeviceView));
}

struct LightSample {
    Spectrum weight = Spectrum(0);  ///< Light sample radiance.
    Float3 dirW = Float3(0);        ///< Light direction in world space.
    Float pdf = 0;                  ///< Light sample pdf.
    PrimitiveID emissiveID;         ///< Light primitive ID.
    Float lightSelectPdf = 0.0;
};

static CRAYSTAL_DEVICE LightSample sampleLight(const SceneView& scene,
                                               const Intersection& intersection,
                                               const BSDF& bsdf,
                                               Sampler& sampler) {
    // Sample scene emissive triangle.
    const int emissiveCnt = scene.materialSystem.getEmissiveCount();
    if (emissiveCnt == 0) return {};
    PrimitiveID emissiveIndex =
        scene.materialSystem.emissiveTriangleIndex[std::min<int>(
            emissiveCnt * sampler.nextSample1D(), emissiveCnt - 1)];

    // Create sample point on triangle.
    auto emissiveTriangle = scene.meshSOA.getTriangle(emissiveIndex);
    Float2 barycentric = sampleBarycentric(sampler.nextSample2D());
    auto vertexData = emissiveTriangle.interpolate(Float3(
        1.f - barycentric.x - barycentric.y, barycentric.x, barycentric.y));

    Float3 targetPos = vertexData.position;
    Float3 toLight = targetPos - intersection.posW;
    Float distSqr = dot(toLight, toLight);
    Float dist = std::sqrt(distSqr);
    Float3 lightDir = toLight / dist;

    if (dot(intersection.getOrientedFaceNormal(), lightDir) < 0.0 ||
        dot(vertexData.normal, lightDir) > 0.0 || dist < kEps) {
        return {};
    }

    // Shadow test
    Ray shadowRay(intersection.posW, lightDir);
    shadowRay.offsetOrigin(intersection.getOrientedFaceNormal());
    shadowRay.tMax = length(shadowRay.origin - targetPos) - kEps;

    if (scene.intersectOcclusion(shadowRay)) {
        return {};
    }

    uint32_t materialID = scene.meshSOA.getMeshDesc(emissiveIndex).materialID;
    MaterialData materialData =
        scene.materialSystem.getMaterialData(materialID);

    LightSample sample;
    sample.weight =
        bsdf.evaluate(intersection.viewW, lightDir) * materialData.emission;
    sample.dirW = lightDir;
    sample.pdf = distSqr / (emissiveTriangle.getArea() * emissiveCnt *
                            absDot(lightDir, vertexData.normal));
    sample.emissiveID = emissiveIndex;
    sample.lightSelectPdf = 1.0 / emissiveCnt;
    return sample;
}

static CRAYSTAL_DEVICE Spectrum evalMIS(const SceneView& scene,
                                        const Intersection& intersection,
                                        const BSDF& bsdf, Sampler& sampler) {
    Spectrum value(0.0);
    // Light sample MIS
    LightSample ls = sampleLight(scene, intersection, bsdf, sampler);
    {
        Float lightPdf = ls.pdf;
        if (lightPdf > 0.0) {
            Float bsdfPdf = bsdf.evaluatePdf(intersection.viewW, ls.dirW);

            if (bsdfPdf > 0.0)
                value += powerHeuristic(1, lightPdf, 1, bsdfPdf) * ls.weight /
                         lightPdf;
        }
    }

    {
        // BSDF sample MIS
        Float3 sampledDir;
        Float bsdfPdf = 0.0;
        Spectrum bsdfWeight = bsdf.sampleEvaluate(sampler, intersection.viewW,
                                                  sampledDir, bsdfPdf);
        if (bsdfWeight.maxValue() > 0.0 && bsdfPdf > 0.0) {
            Ray ray(intersection.posW, sampledDir);
            ray.offsetOrigin(intersection.getOrientedFaceNormal());

            RayHit rayHit;
            rayHit.ray = ray;
            if (scene.intersect(rayHit) &&
                rayHit.hitInfo.primitiveIndex == ls.emissiveID) {
                auto lightIt = scene.createIntersection(rayHit);

                uint32_t materialID =
                    scene.meshSOA.getMeshDesc(rayHit.hitInfo.primitiveIndex)
                        .materialID;
                MaterialData materialData =
                    scene.materialSystem.getMaterialData(materialID);
                if (materialData.isEmissive() && lightIt.isFrontFacing) {
                    Spectrum emission = materialData.emission;

                    // Convert from solid angle pdf to area pdf
                    Float3 toLight = lightIt.posW - intersection.posW;
                    Float distSqr = dot(toLight, toLight);
                    Float3 lightVec = toLight / std::sqrt(distSqr);
                    Float cosThetaLight = absDot(lightVec, lightIt.faceNormal);
                    if (cosThetaLight != 0.0) {
                        Float area =
                            scene.meshSOA
                                .getTriangle(rayHit.hitInfo.primitiveIndex)
                                .getArea();
                        Float lightPdf = distSqr / (area * cosThetaLight);

                        value += powerHeuristic(1, bsdfPdf, 1, lightPdf) *
                                 emission * bsdfWeight / bsdfPdf /
                                 ls.lightSelectPdf;
                    }
                }
            }
        }
    }
    return value;
}

template <bool useMIS = true>
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
                break;
            }

            BSDF bsdf = getBSDF(pScene->materialSystem, materialData, it);

            if constexpr (useMIS) {
                radiance += evalMIS(*pScene, it, bsdf, sampler) * beta;
            } else {
                LightSample ls = sampleLight(*pScene, it, bsdf, sampler);
                if (ls.pdf > 0.0) radiance += ls.weight * beta / ls.pdf;
            }

            Float3 wi;
            Float pdf = 0.0;
            auto f = bsdf.sampleEvaluate(sampler, it.viewW, wi, pdf);
            if (pdf <= 0.0f) break;

            beta *= f / pdf;

            ray.origin = it.posW;
            ray.direction = wi;
            ray.offsetOrigin(it.getOrientedFaceNormal());

        } else {
            terminatePath = true;
        }

        if (sampler.nextSample1D() < pIntegrator->rrThreshold) {
            terminatePath = true;
        }

        if (terminatePath) break;

        Float q = saturate(
            std::max<Float>(pIntegrator->rrThreshold, 1.0f - beta.maxValue()));
        if (sampler.nextSample1D() < q) {
            break;
        }

        beta /= (1.0f - q);
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
        if (mConfigs.useMIS) {
            pathTraceKernel<true><<<dim3(size.x, size.y, 1), dim3(16, 16, 1)>>>(
                i, getDeviceView(), scene.getDeviceView(),
                pCamera->getDeviceView(), pSensor->getDeviceView());
        } else {
            pathTraceKernel<false>
                <<<dim3(size.x, size.y, 1), dim3(16, 16, 1)>>>(
                    i, getDeviceView(), scene.getDeviceView(),
                    pCamera->getDeviceView(), pSensor->getDeviceView());
        }

        cudaDeviceSynchronize();
    }

    pSensor->readbackDeviceData();
}

}  // namespace CRay
