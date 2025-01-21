#pragma once
#include <array>

#include "Enum.h"
#include "Macros.h"
#include "Vec.h"

namespace CRay {

CRAYSTAL_API Float3 XYZ2RGB(const Float3& xyz);

CRAYSTAL_API Float3 RGB2XYZ(const Float3& rgb);

template <int N, typename T>
class SpectrumBase {
   public:
    CRAYSTAL_DEVICE_HOST SpectrumBase() {
        for (int i = 0; i < N; i++) {
            mData[i] = 0.f;
        }
    }

    CRAYSTAL_DEVICE_HOST explicit SpectrumBase(Float value) {
        for (int i = 0; i < N; i++) {
            mData[i] = value;
        }
    }

    CRAYSTAL_HOST explicit SpectrumBase(std::span<const Float, N> values) {
        for (int i = 0; i < N; i++) {
            mData[i] = values[i];
        }
    }

    CRAYSTAL_DEVICE_HOST SpectrumBase(const SpectrumBase& other) {
        for (int i = 0; i < N; i++) {
            mData[i] = other.mData[i];
        }
    }

    CRAYSTAL_DEVICE_HOST explicit SpectrumBase(SpectrumBase&& other) noexcept {
        for (int i = 0; i < N; i++) {
            mData[i] = std::move(other.mData[i]);
        }
    }

    CRAYSTAL_DEVICE_HOST static consteval int size() { return N; }

    CRAYSTAL_DEVICE_HOST Float& operator[](int i) { return mData[i]; }

    CRAYSTAL_DEVICE_HOST Float operator[](int i) const { return mData[i]; }

    CRAYSTAL_DEVICE_HOST T& operator=(const T& other) {
        std::memcpy(mData, other.mData, sizeof(mData));
        return *static_cast<T*>(this);
    }

    CRAYSTAL_DEVICE_HOST T& operator=(T&& other) noexcept {
        for (int i = 0; i < N; i++) {
            mData[i] = std::move(other.mData[i]);
        }
        return *static_cast<T*>(this);
    }

    CRAYSTAL_DEVICE_HOST T operator+(const T& other) const {
        T result(0);
        for (int i = 0; i < N; ++i) {
            result.mData[i] = mData[i] + other.mData[i];
        }
        return result;
    }

    CRAYSTAL_DEVICE_HOST T operator-(const T& other) const {
        T result(0);
        for (int i = 0; i < N; ++i) {
            result.mData[i] = mData[i] - other.mData[i];
        }
        return result;
    }

    CRAYSTAL_DEVICE_HOST T operator*(const T& other) const {
        T result(0);
        for (int i = 0; i < N; ++i) {
            result.mData[i] = mData[i] * other.mData[i];
        }
        return result;
    }

    CRAYSTAL_DEVICE_HOST T operator/(const T& other) const {
        T result(0);
        for (int i = 0; i < N; ++i) {
            result.mData[i] = mData[i] / other.mData[i];
        }
        return result;
    }

    CRAYSTAL_DEVICE_HOST T& operator+=(const T& other) {
        for (int i = 0; i < N; ++i) {
            mData[i] += other.mData[i];
        }
        return *static_cast<T*>(this);
    }

    CRAYSTAL_DEVICE_HOST T& operator-=(const T& other) {
        for (int i = 0; i < N; ++i) {
            mData[i] -= other.mData[i];
        }
        return *static_cast<T*>(this);
    }

    CRAYSTAL_DEVICE_HOST T& operator*=(const T& other) {
        for (int i = 0; i < N; ++i) {
            mData[i] *= other.mData[i];
        }
        return *static_cast<T*>(this);
    }

    CRAYSTAL_DEVICE_HOST T& operator/=(const T& other) {
        for (int i = 0; i < N; ++i) {
            mData[i] /= other.mData[i];
        }
        return *static_cast<T*>(this);
    }

    CRAYSTAL_DEVICE_HOST T operator*(Float scalar) const {
        T result(0);
        for (int i = 0; i < N; ++i) {
            result.mData[i] = mData[i] * scalar;
        }
        return result;
    }

    CRAYSTAL_DEVICE_HOST T operator/(Float scalar) const {
        T result(0);
        for (int i = 0; i < N; ++i) {
            result.mData[i] = mData[i] / scalar;
        }
        return result;
    }

    friend CRAYSTAL_DEVICE_HOST T operator*(Float scalar, const T& spectrum) {
        T result(0);
        for (int i = 0; i < N; ++i) {
            result.mData[i] = scalar * spectrum.mData[i];
        }
        return result;
    }

    friend CRAYSTAL_DEVICE_HOST T operator/(Float scalar, const T& spectrum) {
        T result(0);
        for (int i = 0; i < N; ++i) {
            result.mData[i] = scalar / spectrum.mData[i];
        }
        return result;
    }

    CRAYSTAL_DEVICE_HOST T exp() const {
        T result(0);
        for (int i = 0; i < N; ++i) {
            result.mData[i] = std::exp(mData[i]);
        }
        return result;
    }

    CRAYSTAL_DEVICE_HOST T pow(Float exponent) const {
        T result(0);
        for (int i = 0; i < N; ++i) {
            result.mData[i] = std::pow(mData[i], exponent);
        }
        return result;
    }

    CRAYSTAL_DEVICE_HOST T sqrt() const {
        T result(0);
        for (int i = 0; i < N; ++i) {
            result.mData[i] = std::sqrt(mData[i]);
        }
        return result;
    }

   protected:
    Float mData[N];
};

/** RGB Spectrum stores linear srgb values.
 */
class CRAYSTAL_API RGBSpectrum : public SpectrumBase<3, RGBSpectrum> {
   public:
    using SpectrumBase::SpectrumBase;
    using SpectrumBase::operator=;

    CRAYSTAL_DEVICE_HOST explicit RGBSpectrum(Float3 rgb) {
        mData[0] = rgb.r;
        mData[1] = rgb.g;
        mData[2] = rgb.b;
    }

    CRAYSTAL_DEVICE_HOST RGBSpectrum(const RGBSpectrum& spectrum)
        : SpectrumBase(spectrum) {}

    CRAYSTAL_HOST Float3 toXYZ() const;

    CRAYSTAL_HOST Float3 toRGB() const;

    CRAYSTAL_HOST static RGBSpectrum fromRGB(Float3 rgb, bool isIllum = false);

    CRAYSTAL_HOST static void initialize();
};

// CIE configurations from
// https://github.com/search?q=repo:mitsuba-renderer/mitsuba3%20spectrum&type=code

constexpr Float kCIEMin = 360.f;
constexpr Float kCIEMax = 830.f;
constexpr int kCIESampleCount = 95;

static_assert(kCIESampleCount <= int(kCIEMax - kCIEMin) + 1,
              "kCIESampleCount is greater than the actual sample count.");

// CIE standard observer Y-integration
constexpr Float kCIENormalizationY = Float(1.0 / 106.8569131841719);

/** Broad spectrum stores kCIESampleCount uniform sampled wavelength data.
 */
class CRAYSTAL_API BroadSpectrum
    : public SpectrumBase<kCIESampleCount, BroadSpectrum> {
   public:
    using SpectrumBase::SpectrumBase;
    using SpectrumBase::operator=;

    CRAYSTAL_DEVICE_HOST BroadSpectrum(const BroadSpectrum& spectrum)
        : SpectrumBase(spectrum) {}

    CRAYSTAL_HOST Float3 toXYZ() const;

    CRAYSTAL_HOST Float3 toRGB() const;

    CRAYSTAL_HOST static BroadSpectrum fromRGB(Float3 rgb,
                                               bool isIllum = false);

    CRAYSTAL_HOST static void initialize();
};

using Spectrum = RGBSpectrum;
// TODO: support Broad sampled spectrum
// #if CRAYSTAL_SPECTRUM
// using Spectrum = RGBSpectrum;
// #else
// using Spectrum = RGBSpectrum;
// #endif

}  // namespace CRay
