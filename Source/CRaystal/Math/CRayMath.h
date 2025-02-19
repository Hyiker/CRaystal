#pragma once
#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>
#include <type_traits>

#include "Core/Macros.h"
#include "Core/Vec.h"

namespace CRay {

template <class T>
concept FpType = std::is_floating_point_v<T>;

template <typename T>
CRAYSTAL_DEVICE_HOST Float dot(const T& v0, const T& v1) {
    return glm::dot(v0, v1);
}

template <typename T>
CRAYSTAL_DEVICE_HOST Float absDot(const T& v0, const T& v1) {
    return glm::abs(glm::dot(v0, v1));
}

template <typename T>
CRAYSTAL_DEVICE_HOST T saturate(T value) {
    return std::clamp<T>(value, 0, 1);
}

template <typename T>
CRAYSTAL_DEVICE_HOST T lerp(const T& left, const T& right, Float t) {
    return left * (1.0f - t) + right * t;
}

template <typename T>
CRAYSTAL_DEVICE_HOST constexpr T degrees(const T& val) {
    return glm::degrees(val);
}

template <typename T>
CRAYSTAL_DEVICE_HOST constexpr T radians(const T& val) {
    return glm::radians(val);
}

CRAYSTAL_DEVICE_HOST Float3 reflect(const Float3& incident,
                                    const Float3& normal);

template <FpType T>
CRAYSTAL_DEVICE_HOST constexpr bool isNaN(T val) {
    return ::isnan(val);
}

template <FpType T>
CRAYSTAL_DEVICE_HOST constexpr bool isInfinite(T val) {
    return ::isinf(val);
}

template <FpType T>
CRAYSTAL_DEVICE_HOST void sinCos(T x, T* sPtr, T* cPtr) {
    *sPtr = std::sin(x);
    *cPtr = std::cos(x);
}

#if CRAYSTAL_GCC
template <>
CRAYSTAL_DEVICE_HOST void sinCos(float x, float* sPtr, float* cPtr) {
    ::sincosf(x, sPtr, cPtr);
}

template <>
CRAYSTAL_DEVICE_HOST void sinCos(double x, double* sPtr, double* cPtr) {
    ::sincospi(x, sPtr, cPtr);
}
#endif

CRAYSTAL_API CRAYSTAL_DEVICE_HOST int floatAsInt(float v);

CRAYSTAL_API CRAYSTAL_DEVICE_HOST float intAsFloat(int v);

}  // namespace CRay
