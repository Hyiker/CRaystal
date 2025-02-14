#include "DebugUtils.h"

namespace CRay {

CRAYSTAL_DEVICE_HOST Float3 pseudoColor(uint32_t seed) {
    seed = ((seed + 25605259) * 906882281) ^ 549722321;
    Float3 rgb;
    rgb.r = Float((seed & 0xFF) / 255.f);
    rgb.g = Float((seed >> 8 & 0xFF) / 255.f);
    rgb.b = Float((seed >> 16 & 0xFF) / 255.f);
    return rgb;
}

}  // namespace CRay
