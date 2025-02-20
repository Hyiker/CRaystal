#pragma once
#include <variant>

#include "Core/Frame.h"
#include "Core/Intersection.h"
#include "Core/Macros.h"
#include "Core/Material.h"
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
        return derived()->evaluateImpl(wo, wi) * std::abs(wi.z);
    }

    /** Evaluate the pdf of given wi, wo.
     */
    CRAYSTAL_DEVICE_HOST Float evaluatePdf(const Float3& wo,
                                           const Float3& wi) const {
        return derived()->evaluatePdfImpl(wo, wi);
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

/** Lambertian diffuse BSDF.
 */
class CRAYSTAL_API LambertianBRDF : public BSDFBase<LambertianBRDF> {
   public:
    CRAYSTAL_DEVICE_HOST LambertianBRDF() = default;

    CRAYSTAL_DEVICE_HOST LambertianBRDF(Spectrum kd);

    CRAYSTAL_DEVICE_HOST Spectrum evaluateImpl(const Float3& wo,
                                               const Float3& wi) const;

    CRAYSTAL_DEVICE_HOST Float evaluatePdfImpl(const Float3& wo,
                                               const Float3& wi) const;

    CRAYSTAL_DEVICE_HOST Float3 sampleImpl(const Float3& wo, const Float2& u,
                                           Float& pdf) const;

   private:
    Spectrum mDiffuse;
};

/** Simplified version of Disney principled BRDF(2012), no transmission
 * included. In fact this is the same as Cook-Torrance BRDF.
 */
class CRAYSTAL_API PrincipledBRDF : public BSDFBase<PrincipledBRDF> {
   public:
    CRAYSTAL_DEVICE_HOST PrincipledBRDF() = default;

    /** Construct from PBR params.
     */
    CRAYSTAL_DEVICE_HOST PrincipledBRDF(const Spectrum& baseColor,
                                        Float roughness, Float metallic,
                                        Float anisotropic);

    /** Construct from wavefront obj mtl params.
     */
    CRAYSTAL_DEVICE_HOST PrincipledBRDF(const Spectrum& kd, const Spectrum& ks,
                                        Float specularFactor);

    CRAYSTAL_DEVICE_HOST Spectrum evaluateImpl(const Float3& wo,
                                               const Float3& wi) const;

    CRAYSTAL_DEVICE_HOST Float evaluatePdfImpl(const Float3& wo,
                                               const Float3& wi) const;

    CRAYSTAL_DEVICE_HOST Float3 sampleImpl(const Float3& wo, const Float2& u,
                                           Float& pdf) const;

   private:
    /** Evaluate the diffuse component, thetaD is the angle between view and
     *  half-vector.
     */
    CRAYSTAL_DEVICE_HOST Spectrum evaluateDiffuse(Float cosThetaV,
                                                  Float cosThetaL,
                                                  Float cosThetaD) const;

    CRAYSTAL_DEVICE_HOST Spectrum evaluateSpecular(Float cosThetaV,
                                                   Float cosThetaL,
                                                   Float cosThetaH,
                                                   Float cosThetaD) const;

    Spectrum mBaseColor;  ///< Base color.
    Float mRoughness;     ///< Roughness, controls diffuse and specular.
    Float mMetallic;      ///< 0: dielectric, 1: metal.
    Float mAnisotropic;   ///< 0: Isotropic, 1: Full anisotropic.

    Float mDiffuseProb = 1.0;
    Float mSpecularProb = 1.0;
};

using BSDFVariant = CVariant<LambertianBRDF, PrincipledBRDF>;

class CRAYSTAL_API BSDF {
   public:
    CRAYSTAL_DEVICE_HOST BSDF(BSDFVariant component, Frame frame);

    CRAYSTAL_DEVICE_HOST Spectrum evaluate(const Float3& wo,
                                           const Float3& wi) const;

    CRAYSTAL_DEVICE_HOST Float3 sample(Sampler& sampler, const Float3& wo,
                                       Float& pdf) const;

    CRAYSTAL_DEVICE_HOST Float evaluatePdf(const Float3& wo,
                                           const Float3& wi) const;

    CRAYSTAL_DEVICE_HOST Spectrum sampleEvaluate(Sampler& sampler,
                                                 const Float3& wo, Float3& wi,
                                                 Float& pdf) const;

   private:
    BSDFVariant mComponent;
    Frame mFrame;
};

CRAYSTAL_API CRAYSTAL_DEVICE BSDF getBSDF(const MaterialView& materialSystem,
                                          const MaterialData& material,
                                          const Intersection& it);

}  // namespace CRay
