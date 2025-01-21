#pragma once

#include "Core/Hit.h"
#include "Core/Macros.h"
#include "Math/Ray.h"

namespace CRay {

template <typename T>
struct ShapeTraits;

/** Intersect ray with shape, returns whether intersected with shape.
 *  rayHit will only be updated new hit is closer.
 */
template <typename Shape>
CRAYSTAL_DEVICE_HOST bool intersectShape(PrimitiveID id, const Shape& shape,
                                         RayHit& rayHit) {
    HitInfo hit;
    Float hitT;
    const auto& ray = rayHit.ray;
    bool isHit = ShapeTraits<Shape>::intersect(shape, ray, hit, hitT);
    hit.primitiveIndex = id;

    // Return false if not between tMin and tMax
    if (hitT < ray.tMin || hitT > ray.tMax) return false;

    // Return isHit, but set rayHit info by closest hit
    if (isHit && hitT < rayHit.hitT) {
        rayHit.hitInfo = hit;
        rayHit.hitT = rayHit.hitT;
    }

    return isHit;
}

}  // namespace CRay
