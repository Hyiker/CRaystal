#include "BSDF.h"
#include "Fresnel.h"
#include "Math/CRayMath.h"
#include "Math/MathDefs.h"
#include "Math/Sampling.h"

namespace CRay {

CRAYSTAL_DEVICE_HOST PrincipledBRDF::PrincipledBRDF(const Spectrum& baseColor,
                                                    Float roughness,
                                                    Float metallic,
                                                    Float anisotropic)
    : mBaseColor(baseColor),
      mRoughness(std::clamp(roughness, 0.0f, 1.0f)),
      mMetallic(std::clamp(metallic, 0.0f, 1.0f)),
      mAnisotropic(std::clamp(anisotropic, 0.0f, 1.0f)) {
    mRoughness = std::clamp(mRoughness, 0.001f, 1.0f);
    mMetallic = std::clamp(mMetallic, 0.0f, 1.0f);
}

CRAYSTAL_DEVICE_HOST PrincipledBRDF::PrincipledBRDF(const Spectrum& kd,
                                                    const Spectrum& ks,
                                                    Float specularFactor)
    : PrincipledBRDF(kd, std::sqrt(2.0f / (specularFactor + 2.0f)),
                     ks.averageValue(), 0.0f) {}

CRAYSTAL_DEVICE_HOST Spectrum
PrincipledBRDF::evaluateImpl(const Float3& wo, const Float3& wi) const {
    if (wo.z <= 0.0 || wi.z <= 0.0) return Spectrum(0);

    Float3 halfVec = (wo + wi) / Float(2.0);
    Float cosThetaV = wo.z;
    Float cosThetaI = wi.z;
    Float cosThetaH = halfVec.z;

    Float cosThetaD = dot(wo, halfVec);

    Spectrum diffuse =
        evaluateDiffuse(cosThetaV, cosThetaI, cosThetaD) * (1.0f - mMetallic);
    Spectrum specular =
        evaluateSpecular(cosThetaV, cosThetaI, cosThetaH, cosThetaD);
    return diffuse + specular;
}

CRAYSTAL_DEVICE_HOST Float
PrincipledBRDF::evaluatePdfImpl(const Float3& wo, const Float3& wi) const {
    return cosineWeightSampleHemispherePdf(wi);
}

CRAYSTAL_DEVICE_HOST Float3 PrincipledBRDF::sampleImpl(const Float3& wo,
                                                       const Float2& u,
                                                       Float& pdf) const {
    Float3 wi = cosineWeightSampleHemisphere(u);
    pdf = cosineWeightSampleHemispherePdf(wi);
    return wi;
}

CRAYSTAL_DEVICE_HOST Spectrum PrincipledBRDF::evaluateDiffuse(
    Float cosThetaV, Float cosThetaI, Float cosThetaD) const {
    Float fd90 = 0.5 + 2.0 * mRoughness * cosThetaD * cosThetaD;
    Float fresnelL = 1.0 + (fd90 - 1.0) * std::pow(1.0 - cosThetaI, 5.0);
    Float fresnelV = 1.0 + (fd90 - 1.0) * std::pow(1.0 - cosThetaV, 5.0);

    return mBaseColor * kInvPi * fresnelV * fresnelL;
}

static CRAYSTAL_DEVICE_HOST Float evalNormalDistr(Float alpha,
                                                  Float cosThetaH) {
    // Generalized-Trowbridge-Reitz
    float a2 = alpha * alpha;
    float cos2th = cosThetaH * cosThetaH;
    float den = (1.0 + (a2 - 1.0) * cos2th);

    return a2 * kInvPi / (den * den);
}

static CRAYSTAL_DEVICE_HOST Float smithGGX(float alphaG, float cosThetaV) {
    float a = alphaG * alphaG;
    float b = cosThetaV * cosThetaV;
    return 1 / (cosThetaV + sqrt(a + b - a * b));
}

static CRAYSTAL_DEVICE_HOST Float evalGeometryDistr(Float roughness,
                                                    Float cosThetaI,
                                                    Float cosThetaV) {
    // Smith-GGX
    Float alpha = (0.5 + roughness * 0.5);
    alpha *= alpha;
    return smithGGX(alpha, cosThetaI) * smithGGX(alpha, cosThetaV);
}

CRAYSTAL_DEVICE_HOST Spectrum PrincipledBRDF::evaluateSpecular(
    Float cosThetaV, Float cosThetaI, Float cosThetaH, Float cosThetaD) const {
    Float D = evalNormalDistr(mRoughness, cosThetaH);
    Spectrum f0 = lerp(Spectrum(0.04), mBaseColor, mMetallic);
    Spectrum F = fresnelSchlickApprox(f0, cosThetaD);
    Float G = evalGeometryDistr(mRoughness, cosThetaI, cosThetaV);
    return D * F * G / (4.0 * cosThetaI * cosThetaV);
}

}  // namespace CRay
