#pragma once
#include <memory>

#include "Buffer.h"
#include "Macros.h"
#include "Scene/Scene.h"
#include "Vec.h"
namespace CRay {
class CRAYSTAL_API PathTraceIntegrator {
   public:
    using Ref = std::shared_ptr<PathTraceIntegrator>;

    struct DeviceView {
        uint32_t maxDepth = 8u;     ///< Path tracer max path length.
        Float rrThreshold = 0.15f;  ///< Russian roulette threshold value.
    };

    PathTraceIntegrator();

    void dispatch(Scene& scene, int spp) const;

    DeviceView* getDeviceView() const;

    ~PathTraceIntegrator() = default;

   private:
    DeviceView mView;

    std::unique_ptr<DeviceBuffer> mpConstDataBuffer;
};

using PathTraceIntegratorView = PathTraceIntegrator::DeviceView;

}  // namespace CRay
