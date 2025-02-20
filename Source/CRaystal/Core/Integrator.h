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

    struct Configs {
        uint32_t maxDepth = 10u;
        Float rrThreshold = 0.15f;
        bool useMIS = true;
    };

    struct DeviceView {
        uint32_t maxDepth = 10u;    ///< Path tracer max path length.
        Float rrThreshold = 0.05f;  ///< Russian roulette threshold value.
    };

    PathTraceIntegrator(Configs configs);

    void dispatch(Scene& scene, int spp) const;

    DeviceView* getDeviceView() const;

    ~PathTraceIntegrator() = default;

   private:
    Configs mConfigs;
    DeviceView mView;

    std::unique_ptr<DeviceBuffer> mpConstDataBuffer;
};

using PathTraceIntegratorView = PathTraceIntegrator::DeviceView;

}  // namespace CRay
