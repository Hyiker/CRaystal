#include "Intersection.h"

namespace CRay {
CRAYSTAL_DEVICE_HOST Intersection::Intersection(Float3 posW, Float3 faceNormal,
                                                Float3 shadingNormal,
                                                Float3 viewW)
    : posW(posW), faceNormal(faceNormal), viewW(viewW) {
    isFrontFacing = dot(viewW, faceNormal) > 0.0;
    frame = Frame(isFrontFacing ? shadingNormal: -shadingNormal);
}

}  // namespace CRay
