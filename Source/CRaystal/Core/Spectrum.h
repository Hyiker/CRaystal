#pragma once
#include <array>

#include "Macros.h"
#include "Vec.h"

namespace CRay {

enum class SpectrumType { Reflectance, Illuminant };

CRAYSTAL_API Float3 XYZ2RGB(const Float3& xyz);

CRAYSTAL_API Float3 RGB2XYZ(const Float3& rgb);

template <int N, typename T>
class SpectrumBase {
   public:
    CRAYSTAL_DEVICE_HOST SpectrumBase() : SpectrumBase(Float(0)) {}

    CRAYSTAL_DEVICE_HOST SpectrumBase(Float value) : mData(value) {
        mData.fill(value);
    }

    CRAYSTAL_DEVICE_HOST SpectrumBase(const SpectrumBase& other) {
        mData = other.mData;
    }

    CRAYSTAL_DEVICE_HOST explicit SpectrumBase(SpectrumBase&& other) noexcept {
        mData = std::move(other.mData);
    }

    consteval int size() { return N; }

    CRAYSTAL_DEVICE_HOST Float& operator[](int i) { return mData[i]; }

    CRAYSTAL_DEVICE_HOST Float operator[](int i) const { return mData[i]; }

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

    CRAYSTAL_DEVICE_HOST Float3 toXYZ() const {
        return static_cast<const T*>(this)->toXYZImpl();
    }

    CRAYSTAL_DEVICE_HOST Float3 toRGB() const { return XYZ2RGB(toXYZ()); }

   protected:
    std::array<Float, N> mData;
};

class CRAYSTAL_API RGBSpectrum : public SpectrumBase<3, RGBSpectrum> {
   public:
    using SpectrumBase::SpectrumBase;

    CRAYSTAL_DEVICE_HOST Float3 toXYZImpl() const;

    CRAYSTAL_DEVICE_HOST Float3 toRGBImpl() const;
};

// CIE configurations from
// https://github.com/search?q=repo:mitsuba-renderer/mitsuba3%20spectrum&type=code

constexpr Float kCIEMin = 360.f;
constexpr Float kCIEMax = 830.f;
constexpr int kCIESampleCount = 95;

// Normalization factor to ensure the spectrum value integrates to a luminance
// of 1.0
constexpr Float kCIENormalizationY = Float(1.0 / 106.7502593994140625);

// D65 illuminant data sample from CIE, normalized relative to the power at
// 560nm.
constexpr float kD65Table[kCIESampleCount] = {
    46.6383f, 49.3637f, 52.0891f, 51.0323f, 49.9755f, 52.3118f, 54.6482f,
    68.7015f, 82.7549f, 87.1204f, 91.486f,  92.4589f, 93.4318f, 90.057f,
    86.6823f, 95.7736f, 104.865f, 110.936f, 117.008f, 117.41f,  117.812f,
    116.336f, 114.861f, 115.392f, 115.923f, 112.367f, 108.811f, 109.082f,
    109.354f, 108.578f, 107.802f, 106.296f, 104.79f,  106.239f, 107.689f,
    106.047f, 104.405f, 104.225f, 104.046f, 102.023f, 100.0f,   98.1671f,
    96.3342f, 96.0611f, 95.788f,  92.2368f, 88.6856f, 89.3459f, 90.0062f,
    89.8026f, 89.5991f, 88.6489f, 87.6987f, 85.4936f, 83.2886f, 83.4939f,
    83.6992f, 81.863f,  80.0268f, 80.1207f, 80.2146f, 81.2462f, 82.2778f,
    80.281f,  78.2842f, 74.0027f, 69.7213f, 70.6652f, 71.6091f, 72.979f,
    74.349f,  67.9765f, 61.604f,  65.7448f, 69.8856f, 72.4863f, 75.087f,
    69.3398f, 63.5927f, 55.0054f, 46.4182f, 56.6118f, 66.8054f, 65.0941f,
    63.3828f, 63.8434f, 64.304f,  61.8779f, 59.4519f, 55.7054f, 51.959f,
    54.6998f, 57.4406f, 58.8765f, 60.3125f};

class CRAYSTAL_API BroadSpectrum
    : public SpectrumBase<kCIESampleCount, BroadSpectrum> {
   public:
    using SpectrumBase::SpectrumBase;

    CRAYSTAL_DEVICE_HOST Float3 toXYZImpl() const;

    CRAYSTAL_DEVICE_HOST Float3 toRGBImpl() const;
};

#if CRAYSTAL_SPECTRUM
using Spectrum = BroadSpectrum;
#else
using Spectrum = RGBSpectrum;
#endif

}  // namespace CRay
