#pragma once

#include <cinttypes>

#include "Core/Macros.h"
#include "Core/Vec.h"
#include "Math/Ray.h"

namespace CRay {

enum class HitType : uint32_t { Sphere };

using PrimitiveID = uint32_t;

struct CRAYSTAL_API HitInfo {
    Float2 barycentric;
    HitType type;
    PrimitiveID primitiveIndex;

    CRAYSTAL_DEVICE_HOST Float3 getBarycentricWeights() const {
        return Float3(1.f - barycentric.x - barycentric.y, barycentric.x,
                      barycentric.y);
    }
};

struct CRAYSTAL_API RayHit {
    Ray ray;               ///< Ray data.
    HitInfo hitInfo;       ///< Geometry related hit info.
    Float hitT = kFltInf;  ///< Ray hit T, valid if != inf.
};

}  // namespace CRay
