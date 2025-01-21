#include <thrust/swap.h>

#include "Math/MathDefs.h"
#include "Sphere.h"
namespace CRay {
bool Sphere::intersect(const Ray& ray, HitInfo& hitInfo, Float& hitT) const {
    Float3 oc = ray.origin - center;

    Float a = dot(ray.direction, ray.direction);
    Float b = 2.0f * dot(oc, ray.direction);
    Float c = dot(oc, oc) - radius * radius;

    Float discriminant = b * b - 4.0f * a * c;

    if (discriminant < 0) {
        return false;
    }

    Float sqrtd = sqrt(discriminant);
    Float t0 = (-b - sqrtd) / (2.0f * a);
    Float t1 = (-b + sqrtd) / (2.0f * a);

    if (t0 > t1) {
        thrust::swap(t0, t1);
    }

    if (t0 < ray.tMin || t0 > ray.tMax) {
        if (t1 < ray.tMin || t1 > ray.tMax) {
            return false;
        }
        t0 = t1;
    }

    hitT = t0;
    Float3 hitPoint = ray.origin + ray.direction * hitT;

    Float3 normal = normalize(hitPoint - center);

    Float theta = std::acos(-normal.y);
    Float phi = std::atan2(-normal.z, normal.x) + kPi;

    Float u = phi * (0.5f * kInvPi);
    Float v = theta * kInvPi;

    hitInfo.barycentric = Float2(u, v);

    return true;
}
}  // namespace CRay
