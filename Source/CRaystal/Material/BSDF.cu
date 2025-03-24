#include "BSDF.h"
#include "Math/MathDefs.h"
#include "Math/Sampling.h"
namespace CRay {

CRAYSTAL_DEVICE BSDF getBSDF(const MaterialView& materialSystem,
                             const MaterialData& material,
                             const Intersection& intersection) {
    BSDFVariant bsdfVar;

    Spectrum diffuse = material.diffuseRefl;
    if (material.hasDiffuseTexture()) {
        diffuse = Spectrum(materialSystem.getTexture(material.diffuseHandle)
                               .sample(intersection.texCrd));
    }

    switch (material.type) {
        case MaterialType::FullDiffuse: {
            bsdfVar.emplace<LambertianBRDF>(diffuse);
        } break;
        case MaterialType::Principled: {
            bsdfVar.emplace<PrincipledBRDF>(diffuse, material.specularRefl,
                                            material.shininess);
        } break;
        case MaterialType::Phong: {
            bsdfVar.emplace<PhongBRDF>(diffuse, material.specularRefl,
                                       material.shininess);
        } break;
    }

    return BSDF(bsdfVar, intersection.frame);
}

CRAYSTAL_DEVICE_HOST BSDF::BSDF(BSDFVariant component, Frame frame)
    : mComponent(component), mFrame(frame) {}

template <typename Ret, typename Func, int I = 0>
CRAYSTAL_DEVICE_HOST Ret dispatchBSDF(const BSDFVariant& variant, Func&& func,
                                      Ret defaultValue = Ret()) {
    if constexpr (I >= BSDFVariant::type_count)
        return defaultValue;
    else {
        if (auto pBSDF = variant.template get_if<BSDFVariant::type_at_t<I>>())
            return func(*pBSDF);
        else
            return dispatchBSDF<Ret, Func, I + 1>(
                variant, std::forward<Func>(func), defaultValue);
    }
}

CRAYSTAL_DEVICE_HOST Spectrum BSDF::evaluate(const Float3& wo,
                                             const Float3& wi) const {
    const Float3 woLocal = mFrame.toLocal(wo);
    const Float3 wiLocal = mFrame.toLocal(wi);

    return dispatchBSDF<Spectrum>(
        mComponent,
        [&](const auto& bsdf) { return bsdf.evaluate(woLocal, wiLocal); },
        Spectrum(0.f));
}

CRAYSTAL_DEVICE_HOST Float3 BSDF::sample(Sampler& sampler, const Float3& wo,
                                         Float& pdf) const {
    const Float3 woLocal = mFrame.toLocal(wo);
    const Float2 u = sampler.nextSample2D();
    pdf = 0.0;

    return dispatchBSDF<Float3>(
        mComponent,
        [&](const auto& bsdf) {
            return mFrame.toWorld(bsdf.sample(woLocal, u, pdf));
        },
        Float3(0.f));
}

CRAYSTAL_DEVICE_HOST Float BSDF::evaluatePdf(const Float3& wo,
                                             const Float3& wi) const {
    const Float3 woLocal = mFrame.toLocal(wo);
    const Float3 wiLocal = mFrame.toLocal(wi);

    return dispatchBSDF<Float>(
        mComponent,
        [&](const auto& bsdf) { return bsdf.evaluatePdf(woLocal, wiLocal); },
        Float(0.f));
}

CRAYSTAL_DEVICE_HOST Spectrum BSDF::sampleEvaluate(Sampler& sampler,
                                                   const Float3& wo, Float3& wi,
                                                   Float& pdf) const {
    const Float3 woLocal = mFrame.toLocal(wo);
    Float3 wiLocal;
    pdf = 0.0;

    return dispatchBSDF<Spectrum>(
        mComponent,
        [&](const auto& bsdf) {
            auto ret = bsdf.sampleEvaluate(woLocal, sampler.nextSample2D(),
                                           wiLocal, pdf);
            wi = mFrame.toWorld(wiLocal);
            return ret;
        },
        Spectrum(0.f));
}
}  // namespace CRay
