#pragma once
#include "Frame.h"
#include "Hit.h"
#include "Macros.h"
#include "Vec.h"

namespace CRay {

/** Shading related intersection data.
 */
struct CRAYSTAL_API Intersection {
    Float3 posW;         ///< World space position.
    Frame frame;         ///< Shading frame.
    Float3 faceNormal;   ///< Geometry face normal.
    Float3 viewW;        ///< view direction.
    Float2 texCrd;       ///< Texture coordinates.
    bool isFrontFacing;  ///< True is dot(viewW, faceNormal) > 0.

    CRAYSTAL_DEVICE_HOST Intersection(Float3 posW, Float3 faceNormal,
                                      Float3 shadingNormal, Float3 viewW,
                                      Float2 texCrd);

    CRAYSTAL_DEVICE_HOST Float3 getOrientedFaceNormal() const {
        return isFrontFacing ? faceNormal : -faceNormal;
    }
};

}  // namespace CRay
