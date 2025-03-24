#pragma once
#include "Core/Macros.h"
#include "Core/Vec.h"
#include "MathDefs.h"

namespace CRay {

struct CRAYSTAL_API Ray {
    Float3 origin = Float3(0);
    Float tMin = 0.f;
    Float3 direction = Float3(0);
    Float tMax = kFltInf;

    CRAYSTAL_DEVICE_HOST Ray() = default;
    CRAYSTAL_DEVICE_HOST Ray(const Float3 o, const Float3 d)
        : origin(o), direction(d) {}

    CRAYSTAL_DEVICE_HOST void offsetOrigin(const Float3& normal);
};

CRAYSTAL_API CRAYSTAL_DEVICE_HOST Float3 offsetRayOrigin(const Float3p32 p, const Float3p32 n);

}  // namespace CRay
