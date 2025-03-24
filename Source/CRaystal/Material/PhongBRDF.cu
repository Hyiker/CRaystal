#include "BSDF.h"
#include "Math/MathDefs.h"
#include "Math/Sampling.h"

namespace CRay {

// From Importance Sampling of the Phong Reflectance Model
// https://www.cs.princeton.edu/courses/archive/fall08/cos526/assign3/lawrence.pdf

CRAYSTAL_DEVICE_HOST PhongBRDF::PhongBRDF(Spectrum kd, Spectrum ks,
                                          Float specularExp)
    : mKd(kd), mKs(ks), mSpecularExp(specularExp) {
    mDiffuseProb = std::max<Float>(kd.averageValue(), 0.0);
    mSpecularProb = std::max<Float>(ks.averageValue(), 0.0);

    Float probSum = glm::max(mDiffuseProb + mSpecularProb, kEps);
    mDiffuseProb /= probSum;
    mSpecularProb /= probSum;
}

CRAYSTAL_DEVICE_HOST Spectrum PhongBRDF::evaluateImpl(const Float3& wo,
                                                      const Float3& wi) const {
    Float3 reflectDir = reflect(-wo, Float3(0, 0, 1));
    return mKd * kInvPi +
           mKs * (mSpecularExp + 2) * kInv2Pi *
               std::pow<Float>(std::max<Float>(dot(reflectDir, wi), 0.0),
                               mSpecularExp);
}

CRAYSTAL_DEVICE_HOST Float PhongBRDF::evaluatePdfImpl(const Float3& wo,
                                                      const Float3& wi) const {
    Float diffusePdf = cosineWeightSampleHemispherePdf(wi);

    Float3 reflectDir = reflect(-wo, Float3(0, 0, 1));
    Float cosAlpha = std::max<Float>(dot(reflectDir, wi), 0.0);
    Float specularPdf =
        (mSpecularExp + 1) * kInv2Pi * std::pow(cosAlpha, mSpecularExp);

    return diffusePdf * mDiffuseProb + specularPdf * mSpecularProb;
}

CRAYSTAL_DEVICE_HOST Float3 PhongBRDF::sampleImpl(const Float3& wo,
                                                  const Float2& u,
                                                  Float& pdf) const {
    Float2 u0 = u;
    Float3 wi{};
    pdf = 0.0;

    bool hasDiffuse = mDiffuseProb > 0.0;
    bool hasSpecular = mSpecularProb > 0.0;
    if (hasDiffuse && u0.x < mDiffuseProb) {
        u0.x /= mDiffuseProb;
        wi = cosineWeightSampleHemisphere(u0);
        pdf = cosineWeightSampleHemispherePdf(wi);
    } else if (hasSpecular) {
        u0.x = (u0.x - mDiffuseProb) / mSpecularProb;
        Float alpha = acosSafe(std::pow(u0.x, 1 / (mSpecularExp + 1)));

        Float theta = glm::clamp<Float>(acosSafe(wo.z) + alpha, 0.0, kPi / 2.0);
        Float phi = k2Pi * u0.y;
        Float sinTheta = std::sin(theta);
        wi = Float3(sinTheta * std::cos(phi), sinTheta * std::sin(phi),
                    std::sqrt(1.0 - sinTheta * sinTheta));
        pdf = (mSpecularExp + 1) * kInv2Pi *
              std::pow(std::cos(alpha), mSpecularExp);
    }
    return wi;
}
}  // namespace CRay
