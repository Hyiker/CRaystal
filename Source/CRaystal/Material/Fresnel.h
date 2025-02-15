#pragma once
#include "Core/Macros.h"
#include "Core/Spectrum.h"
namespace CRay {
CRAYSTAL_API CRAYSTAL_DEVICE_HOST Spectrum
fresnelSchlickApprox(Spectrum f0, Float cosThetaV);

}  // namespace CRay
