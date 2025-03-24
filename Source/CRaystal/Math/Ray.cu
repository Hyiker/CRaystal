#include "CRayMath.h"
#include "Ray.h"
namespace CRay {

// From ray tracing-gem chapt.6
constexpr float kOrigin = 1.0f / 32.0f;
constexpr float kFltScale = 1.0f / 65535.0f;
constexpr float kIntScale = 256.0f;

// Normal points outward for rays exiting the surface, else is flipped.
CRAYSTAL_DEVICE_HOST Float3 offsetRayOrigin(const Float3p32 p,
                                                   const Float3p32 n) {
    Int3 ofI(kIntScale * n.x, kIntScale * n.y, kIntScale * n.z);

    Float3p32 pI(intAsFloat(floatAsInt(p.x) + ((p.x < 0) ? -ofI.x : ofI.x)),
                 intAsFloat(floatAsInt(p.y) + ((p.y < 0) ? -ofI.y : ofI.y)),
                 intAsFloat(floatAsInt(p.z) + ((p.z < 0) ? -ofI.z : ofI.z)));

    return Float3(std::abs(p.x) < kOrigin ? p.x + kFltScale * n.x : pI.x,
                  std::abs(p.y) < kOrigin ? p.y + kFltScale * n.y : pI.y,
                  std::abs(p.z) < kOrigin ? p.z + kFltScale * n.z : pI.z);
}

CRAYSTAL_DEVICE_HOST void Ray::offsetOrigin(const Float3& normal) {
    origin = offsetRayOrigin(origin, normal);
}

}  // namespace CRay
