#include "Fresnel.h"
namespace CRay {
CRAYSTAL_DEVICE_HOST Spectrum fresnelSchlickApprox(Spectrum f0,
                                                   Float cosThetaV) {
    Float v0 = 1.f - cosThetaV;
    Float v1 = v0 * v0;
    return f0 + (Spectrum(1) - f0) * v1 * v1 * v0;
}
}  // namespace CRay
