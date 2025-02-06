#include <thrust/swap.h>

#include "AABB.h"
#include "Ray.h"
namespace CRay {
CRAYSTAL_DEVICE_HOST bool AABB::intersect(const Ray& ray, Float& hitT) const {
    Float tMin = ray.tMin;
    Float tMax = ray.tMax;

    [[unroll]]
    for (int i = 0; i < 3; ++i) {
        if (std::abs(pMax[i] - pMin[i]) < kFltEps) {
            if (std::abs(ray.direction[i]) < kFltEps) {
                if (std::abs(ray.origin[i] - pMin[i]) > kFltEps) {
                    return false;
                }
                continue;
            }
            Float t = (pMin[i] - ray.origin[i]) / ray.direction[i];
            tMin = std::max(tMin, t);
            tMax = std::min(tMax, t);
        } else {
            Float invD = Float(1) / ray.direction[i];
            Float t0 = (pMin[i] - ray.origin[i]) * invD;
            Float t1 = (pMax[i] - ray.origin[i]) * invD;

            if (invD < Float(0)) {
                thrust::swap(t0, t1);
            }

            tMin = std::max(tMin, t0);
            tMax = std::min(tMax, t1);
        }

        if (tMin > tMax) {
            return false;
        }
    }

    if (tMin > ray.tMax || tMax < ray.tMin) {
        return false;
    }

    hitT = tMin;
    return true;
}

}  // namespace CRay
