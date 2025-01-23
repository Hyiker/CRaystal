#include <thrust/swap.h>

#include "Math/MathDefs.h"
#include "Sphere.h"
namespace CRay {
CRAYSTAL_DEVICE bool SphereSOA::intersect(PrimitiveID id, const Ray& ray,
                                          HitInfo& hitInfo, Float& hitT) const {
    Float3 center = pCenter[id];
    Float radius = pRadius[id];

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

CRAYSTAL_DEVICE SphereData SphereSOA::getSphere(PrimitiveID id) const {
    SphereData data;
    data.center = pCenter[id];
    data.radius = pRadius[id];

    return data;
}

SphereManager::SphereManager(const std::vector<SphereData>& data) {
    int cnt = data.size();

    std::vector<Float3> center;
    std::vector<Float> radius;
    center.reserve(cnt);
    radius.reserve(cnt);

    for (const auto& d : data) {
        center.push_back(d.center);
        radius.push_back(d.radius);
    }

    mpCenterBuffer = std::make_unique<DeviceBuffer>(sizeof(Float3) * cnt);
    mpRadiusBuffer = std::make_unique<DeviceBuffer>(sizeof(Float) * cnt);

    mpCenterBuffer->copyFromHost(center.data());
    mpRadiusBuffer->copyFromHost(radius.data());

    mView.count = cnt;
    mView.pCenter = (Float3*)mpCenterBuffer->data();
    mView.pRadius = (Float*)mpRadiusBuffer->data();
}

}  // namespace CRay
