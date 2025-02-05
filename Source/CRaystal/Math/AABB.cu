#include "AABB.h"
#include "Ray.h"
namespace CRay {
CRAYSTAL_DEVICE_HOST bool AABB::intersect(const Ray& ray, Float& hitT) const {
    float tMin = ray.tMin;
    float tMax = ray.tMax;

    [[unroll]]
    for (int i = 0; i < 3; ++i) {
        float invD = 1.f / ray.direction[i];
        float t0 = (pMin[i] - ray.origin[i]) * invD;
        float t1 = (pMax[i] - ray.origin[i]) * invD;

        if (invD < 0.0f) {
            float temp = t0;
            t0 = t1;
            t1 = temp;
        }

        tMin = t0 > tMin ? t0 : tMin;
        tMax = t1 < tMax ? t1 : tMax;

        if (tMin > tMax) {
            return false;
        }
    }

    if (tMin <= ray.tMax && tMax >= ray.tMin) {
        hitT = tMin;
        return true;
    }

    return false;
}

}  // namespace CRay
