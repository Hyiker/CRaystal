#pragma once
#include "Core/Macros.h"
#include "Core/Vec.h"
namespace CRay {

CRAYSTAL_API CRAYSTAL_DEVICE_HOST Float2 sampleBarycentric(const Float2& u);

CRAYSTAL_API CRAYSTAL_DEVICE_HOST Float3 uniformSampleSphere(const Float2& u);

CRAYSTAL_API CRAYSTAL_DEVICE_HOST Float uniformSampleSpherePdf();

CRAYSTAL_API CRAYSTAL_DEVICE_HOST Float3
uniformSampleHemisphere(const Float2& u);

CRAYSTAL_API CRAYSTAL_DEVICE_HOST Float uniformSampleHemispherePdf();

CRAYSTAL_API CRAYSTAL_DEVICE_HOST Float3
cosineWeightSampleHemisphere(const Float2& u);

CRAYSTAL_API CRAYSTAL_DEVICE_HOST Float
cosineWeightSampleHemispherePdf(const Float3& d);

/** Evaluate weight for power heuristic with \beta = 2.
 */
CRAYSTAL_API CRAYSTAL_DEVICE_HOST Float powerHeuristic(int nf, Float fPdf,
                                                       int ng, Float gPdf);

}  // namespace CRay
