#include "BSDF.h"
#include "Math/MathDefs.h"
#include "Math/Sampling.h"
namespace CRay {

CRAYSTAL_DEVICE_HOST LambertianBSDF::LambertianBSDF(Spectrum kd)
    : mDiffuse(kd) {}

CRAYSTAL_DEVICE_HOST Spectrum
LambertianBSDF::evaluateImpl(const Float3& wo, const Float3& wi) const {
    return mDiffuse * kInvPi;
}

CRAYSTAL_DEVICE_HOST Float3 LambertianBSDF::sampleImpl(const Float3& wo,
                                                       const Float2& u,
                                                       Float& pdf) const {
    Float3 wi = cosineWeightSampleHemisphere(u);
    pdf = cosineWeightSampleHemispherePdf(wi);
    return wi;
}

CRAYSTAL_DEVICE_HOST Spectrum
PrincipledBSDF::evaluateImpl(const Float3& wo, const Float3& wi) const {
    return Spectrum(0.f);
}

CRAYSTAL_DEVICE_HOST Float3 PrincipledBSDF::sampleImpl(const Float3& wo,
                                                       const Float2& u,
                                                       Float& pdf) const {
    Float3 wi = cosineWeightSampleHemisphere(u);
    pdf = cosineWeightSampleHemispherePdf(wi);
    return wi;
}

CRAYSTAL_DEVICE_HOST BSDF::BSDF(BSDFVariant component, Frame frame)
    : mComponent(component), mFrame(frame) {}

template <typename Ret, typename Func>
CRAYSTAL_DEVICE_HOST Ret dispatchBSDF(BSDFVariant variant, Func&& func,
                                      Ret defaultValue = Ret()) {
    if (const auto* lambertian = get_if<LambertianBSDF>(&variant)) {
        return func(*lambertian);
    } else if (const auto* principled = get_if<PrincipledBSDF>(&variant)) {
        return func(*principled);
    }
    return defaultValue;
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
