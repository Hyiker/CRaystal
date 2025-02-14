#include "MathDefs.h"
#include "Sampling.h"
namespace CRay {

CRAYSTAL_DEVICE_HOST Float3 uniformSampleSphere(const Float2& u) {
    Float z = 1.0f - 2.0f * u[0];
    Float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
    Float phi = 2.0f * kPi * u[1];
    Float x = r * std::cos(phi);
    Float y = r * std::sin(phi);
    return Float3(x, y, z);
}

CRAYSTAL_DEVICE_HOST Float uniformSampleSpherePdf() { return 1.0f * kInv4Pi; }

CRAYSTAL_DEVICE_HOST Float3 uniformSampleHemisphere(const Float2& u) {
    Float z = u[0];
    Float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
    Float phi = 2.0f * kPi * u[1];
    Float x = r * std::cos(phi);
    Float y = r * std::sin(phi);
    return Float3(x, y, z);
}

CRAYSTAL_DEVICE_HOST Float uniformSampleHemispherePdf() {
    return 1.0f * kInv2Pi;
}

CRAYSTAL_DEVICE_HOST Float3 cosineWeightSampleHemisphere(const Float2& u) {
    // Using Malley's method
    Float r = std::sqrt(u[0]);
    Float phi = 2.0f * kPi * u[1];
    Float x = r * std::cos(phi);
    Float y = r * std::sin(phi);
    Float z = std::sqrt(std::max(0.0f, 1.0f - x * x - y * y));
    return Float3(x, y, z);
}

CRAYSTAL_DEVICE_HOST Float cosineWeightSampleHemispherePdf(const Float3& d) {
    return std::max(0.0f, d[2]) * kInvPi;
}
}  // namespace CRay
