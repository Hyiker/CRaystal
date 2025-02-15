#include "BSDF.h"
#include "Math/MathDefs.h"
#include "Math/Sampling.h"

namespace CRay {
CRAYSTAL_DEVICE_HOST LambertianBRDF::LambertianBRDF(Spectrum kd)
    : mDiffuse(kd) {}

CRAYSTAL_DEVICE_HOST Spectrum
LambertianBRDF::evaluateImpl(const Float3& wo, const Float3& wi) const {
    return mDiffuse * kInvPi;
}

CRAYSTAL_DEVICE_HOST Float3 LambertianBRDF::sampleImpl(const Float3& wo,
                                                       const Float2& u,
                                                       Float& pdf) const {
    Float3 wi = cosineWeightSampleHemisphere(u);
    pdf = cosineWeightSampleHemispherePdf(wi);
    return wi;
}
}  // namespace CRay
