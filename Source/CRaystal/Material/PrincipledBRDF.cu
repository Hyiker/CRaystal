#include "BSDF.h"
#include "Fresnel.h"
#include "Math/CRayMath.h"
#include "Math/MathDefs.h"
#include "Math/Sampling.h"

namespace CRay {

static CRAYSTAL_DEVICE_HOST Float evalNormalDistr(Float roughness,
                                                  Float cosThetaH) {
    // Generalized-Trowbridge-Reitz
    Float alpha = roughness * roughness;
    Float a2 = alpha * alpha;
    Float cos2th = cosThetaH * cosThetaH;
    Float den = (1.0 + (a2 - 1.0) * cos2th);

    return a2 * kInvPi / (den * den);
}

static CRAYSTAL_DEVICE_HOST Float smithGGX(float alphaG, float cosThetaV) {
    Float a = alphaG * alphaG;
    Float b = cosThetaV * cosThetaV;
    return 1 / (cosThetaV + std::sqrt(a + b - a * b));
}

static CRAYSTAL_DEVICE_HOST Float evalGeometryDistr(Float roughness,
                                                    Float cosThetaI,
                                                    Float cosThetaV) {
    // Smith-GGX
    Float alpha = (0.5 + roughness * 0.5);
    alpha *= alpha;
    return smithGGX(alpha, cosThetaI) * smithGGX(alpha, cosThetaV);
}

CRAYSTAL_DEVICE_HOST PrincipledBRDF::PrincipledBRDF(const Spectrum& baseColor,
                                                    Float roughness,
                                                    Float metallic,
                                                    Float anisotropic)
    : mBaseColor(baseColor),
      mRoughness(std::clamp(roughness, 0.0f, 1.0f)),
      mMetallic(std::clamp(metallic, 0.0f, 1.0f)),
      mAnisotropic(std::clamp(anisotropic, 0.0f, 1.0f)) {
    Float specularProb = lerp(0.1f, 1.0f, mMetallic);
    mDiffuseProb = 1.0 - specularProb;
    mSpecularProb = specularProb;
}

CRAYSTAL_DEVICE_HOST PrincipledBRDF::PrincipledBRDF(const Spectrum& kd,
                                                    const Spectrum& ks,
                                                    Float specularFactor)
    : PrincipledBRDF(kd, std::sqrt(2.0f / (specularFactor + 1.0f)),
                     ks.averageValue(), 0.0f) {}

CRAYSTAL_DEVICE_HOST Spectrum
PrincipledBRDF::evaluateImpl(const Float3& wo, const Float3& wi) const {
    if (wo.z <= 0.0 || wi.z <= 0.0) return Spectrum(0);

    Float3 wh = normalize(wo + wi);
    Float cosThetaV = wo.z;
    Float cosThetaI = wi.z;
    Float cosThetaH = wh.z;

    Float cosThetaD = dot(wo, wh);

    Spectrum diffuse =
        evaluateDiffuse(cosThetaV, cosThetaI, cosThetaD) * (1.0f - mMetallic);
    Spectrum specular =
        evaluateSpecular(cosThetaV, cosThetaI, cosThetaH, cosThetaD);
    return diffuse + specular;
}

CRAYSTAL_DEVICE_HOST Float
PrincipledBRDF::evaluatePdfImpl(const Float3& wo, const Float3& wi) const {
    if (wo.z <= 0.0 || wi.z <= 0.0) return 0.0f;

    Float3 wh = normalize(wo + wi);

    Float cosThetaH = wh.z;
    Float cosThetaD = dot(wo, wh);

    if (cosThetaD <= 0.0f) return 0.0f;

    Float diffusePdf = cosineWeightSampleHemispherePdf(wi);

    Float specularPdf = 0.0f;
    if (cosThetaH > 0.0f) {
        Float D = evalNormalDistr(mRoughness, cosThetaH);
        // Jacobian of half-vector transformation
        specularPdf = D * cosThetaH / (4.0f * cosThetaD);
    }

    Float pdf = diffusePdf * mDiffuseProb + specularPdf * mSpecularProb;

    return pdf;
}

CRAYSTAL_DEVICE_HOST Float3 PrincipledBRDF::sampleImpl(const Float3& wo,
                                                       const Float2& u,
                                                       Float& pdf) const {
    if (wo.z <= 0.0) {
        pdf = 0.0f;
        return Float3(0.0f);
    }

    Float u1 = u[0];
    Float u2 = u[1];

    Float3 wi;
    if (u1 < mSpecularProb) {
        u1 /= mSpecularProb;

        Float alpha = mRoughness * mRoughness;
        Float alphaSqr = alpha * alpha;
        Float phi = u2 * k2Pi;
        Float tanThetaSqr = alphaSqr * u1 / (1 - u1);
        Float cosTheta = 1 / sqrt(1 + tanThetaSqr);
        Float r = std::sqrt(std::max<Float>(1 - cosTheta * cosTheta, 0));

        Float3 wh = Float3(cos(phi) * r, sin(phi) * r, cosTheta);

        wi = reflect(-wo, wh);
    } else {
        u1 = (u1 - mSpecularProb) / (1.0f - mSpecularProb);
        wi = cosineWeightSampleHemisphere(Float2(u1, u2));
    }

    pdf = evaluatePdfImpl(wo, wi);

    return wi;
}

CRAYSTAL_DEVICE_HOST Spectrum PrincipledBRDF::evaluateDiffuse(
    Float cosThetaV, Float cosThetaI, Float cosThetaD) const {
    Float fd90 = 0.5 + 2.0 * mRoughness * cosThetaD * cosThetaD;
    Float fresnelL = 1.0 + (fd90 - 1.0) * std::pow(1.0 - cosThetaI, 5.0);
    Float fresnelV = 1.0 + (fd90 - 1.0) * std::pow(1.0 - cosThetaV, 5.0);
    return mBaseColor * kInvPi * fresnelV * fresnelL;
}

CRAYSTAL_DEVICE_HOST Spectrum PrincipledBRDF::evaluateSpecular(
    Float cosThetaV, Float cosThetaI, Float cosThetaH, Float cosThetaD) const {
    Float D = evalNormalDistr(mRoughness, cosThetaH);
    Spectrum f0 = lerp(mBaseColor, Spectrum(0.04), mMetallic);
    Spectrum F = fresnelSchlickApprox(f0, cosThetaD);
    Float G = evalGeometryDistr(mRoughness, cosThetaI, cosThetaV);
    return D * F * G / (4.0 * cosThetaI * cosThetaV);
}

}  // namespace CRay
