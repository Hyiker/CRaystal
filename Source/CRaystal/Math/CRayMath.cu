#include <cmath>

#include "CRayMath.h"
#include "MathDefs.h"
namespace CRay {
CRAYSTAL_DEVICE_HOST Float3 reflect(const Float3& incident,
                                    const Float3& normal) {
    return glm::reflect(incident, normal);
}

CRAYSTAL_DEVICE_HOST Float acosSafe(Float v) {
    return std::acos(glm::clamp<Float>(v, kEps, 1.0 - kEps));
}

CRAYSTAL_DEVICE_HOST Float asinSafe(Float v) {
    return std::asin(glm::clamp<Float>(v, kEps, 1.0 - kEps));
}

union FloatIntUnion {
    float f;
    int i;
};

CRAYSTAL_DEVICE_HOST int floatAsInt(float v) {
    FloatIntUnion u;
    u.f = v;
    return u.i;
}

CRAYSTAL_DEVICE_HOST float intAsFloat(int v) {
    FloatIntUnion u;
    u.i = v;
    return u.f;
}

}  // namespace CRay
