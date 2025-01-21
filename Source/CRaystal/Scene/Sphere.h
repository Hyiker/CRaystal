#pragma once

#include "Core/Macros.h"
#include "Shape.h"

namespace CRay {
struct CRAYSTAL_API Sphere {
    Float3 center = Float3(0);
    Float radius = 1.f;

    Sphere() = default;

    CRAYSTAL_DEVICE_HOST bool intersect(const Ray& ray, HitInfo& hitInfo,
                                        Float& hitT) const;
};

template <>
struct CRAYSTAL_API ShapeTraits<Sphere> {
    static CRAYSTAL_DEVICE_HOST bool intersect(const Sphere& sphere,
                                               const Ray& ray, HitInfo& hitInfo,
                                               Float& hitT) {
        hitInfo.type = HitType::Sphere;
        return sphere.intersect(ray, hitInfo, hitT);
    }
};

}  // namespace CRay
