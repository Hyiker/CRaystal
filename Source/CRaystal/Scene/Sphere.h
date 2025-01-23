#pragma once

#include "Core/Buffer.h"
#include "Core/Macros.h"
#include "Shape.h"

namespace CRay {

struct CRAYSTAL_API SphereData {
    Float3 center = Float3(0);
    Float radius = 1.f;
};

class CRAYSTAL_API SphereManager : public ShapeManagerBase {
   public:
    using Ref = std::shared_ptr<SphereManager>;
    struct CRAYSTAL_API DeviceView {
        uint32_t count;

        Float3* pCenter;
        Float* pRadius;

        CRAYSTAL_DEVICE bool intersect(PrimitiveID id, const Ray& ray,
                                       HitInfo& hitInfo, Float& hitT) const;

        CRAYSTAL_DEVICE SphereData getSphere(PrimitiveID id) const;
    };

    SphereManager(const std::vector<SphereData>& data);

    DeviceView getDeviceView() const { return mView; }

   private:
    DeviceView mView;

    std::unique_ptr<DeviceBuffer> mpCenterBuffer;
    std::unique_ptr<DeviceBuffer> mpRadiusBuffer;
};

using SphereSOA = SphereManager::DeviceView;

template <>
struct CRAYSTAL_API ShapeSOATraits<SphereSOA> {
    static CRAYSTAL_DEVICE bool intersect(const SphereSOA& soa,
                                               PrimitiveID id, const Ray& ray,
                                               HitInfo& hitInfo, Float& hitT) {
        hitInfo.type = HitType::Sphere;
        return soa.intersect(id, ray, hitInfo, hitT);
    }
};

}  // namespace CRay
