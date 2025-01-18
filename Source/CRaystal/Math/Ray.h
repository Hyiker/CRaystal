#pragma once
#include "Core/Macros.h"
#include "Core/Vec.h"

namespace CRay {
struct CRAYSTAL_API Ray {
    Float3 origin = Float3(0);
    Float3 direction = Float3(0);

    CRAYSTAL_DEVICE_HOST Ray(const Float3 o, const Float3 d)
        : origin(o), direction(d) {}
};
}  // namespace CRay
