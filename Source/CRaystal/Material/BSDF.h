#pragma once
#include <variant>

#include "Core/Frame.h"
#include "Core/Macros.h"
#include "Core/Sampler.h"
#include "Core/Spectrum.h"
#include "Core/Vec.h"
#include "Utils/CVariant.h"

namespace CRay {

template <class Derived>
class BSDFBase {
   public:
    CRAYSTAL_DEVICE_HOST const Derived* derived() const {
        return static_cast<const Derived*>(this);
    }

    CRAYSTAL_DEVICE_HOST Derived* derived() {
        return static_cast<Derived*>(this);
    }

    /** Evaluate BSDF in local space.
     *  wo: observation direction
     *  wi: sampled light direction
     */
    CRAYSTAL_DEVICE_HOST Spectrum evaluate(const Float3& wo,
                                           const Float3& wi) const {
        // Apply \cos(\theta) here
        return derived()->evaluateImpl(wo, wi) * wi.z;
    }

    /** Sample the outgoing direction in local space.
     */
    CRAYSTAL_DEVICE_HOST Float3 sample(const Float3& wo, const Float2& u,
                                       Float& pdf) const {
        return derived()->sampleImpl(wo, u, pdf);
    }

    /** Sample the outgoing direction in local space and evaluate BSDF value.
     */
    CRAYSTAL_DEVICE_HOST Spectrum sampleEvaluate(const Float3& wo,
                                                 const Float2& u, Float3& wi,
                                                 Float& pdf) const {
        wi = sample(wo, u, pdf);
        return evaluate(wo, wi);
    }
};

class CRAYSTAL_API LambertianBSDF : public BSDFBase<LambertianBSDF> {
   public:
    CRAYSTAL_DEVICE_HOST LambertianBSDF() = default;

    CRAYSTAL_DEVICE_HOST LambertianBSDF(Spectrum kd);

    CRAYSTAL_DEVICE_HOST Spectrum evaluateImpl(const Float3& wo,
                                               const Float3& wi) const;

    CRAYSTAL_DEVICE_HOST Float3 sampleImpl(const Float3& wo, const Float2& u,
                                           Float& pdf) const;

   private:
    Spectrum mDiffuse;
};

class CRAYSTAL_API PrincipledBSDF : public BSDFBase<PrincipledBSDF> {
   public:
    CRAYSTAL_DEVICE_HOST PrincipledBSDF() = default;

    CRAYSTAL_DEVICE_HOST Spectrum evaluateImpl(const Float3& wo,
                                               const Float3& wi) const;

    CRAYSTAL_DEVICE_HOST Float3 sampleImpl(const Float3& wo, const Float2& u,
                                           Float& pdf) const;

   private:
};

using BSDFVariant = CVariant<LambertianBSDF, PrincipledBSDF>;

class CRAYSTAL_API BSDF {
   public:
    CRAYSTAL_DEVICE_HOST BSDF(BSDFVariant component, Frame frame);

    CRAYSTAL_DEVICE_HOST Spectrum evaluate(const Float3& wo,
                                           const Float3& wi) const;

    CRAYSTAL_DEVICE_HOST Float3 sample(Sampler& sampler, const Float3& wo,
                                       Float& pdf) const;

    CRAYSTAL_DEVICE_HOST Spectrum sampleEvaluate(Sampler& sampler,
                                                 const Float3& wo, Float3& wi,
                                                 Float& pdf) const;

   private:
    BSDFVariant mComponent;
    Frame mFrame;
};

}  // namespace CRay
