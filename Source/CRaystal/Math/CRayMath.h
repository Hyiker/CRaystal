#pragma once
#include <algorithm>

#include "Core/Macros.h"
#include <glm/glm.hpp>

namespace CRay {

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

}  // namespace CRay
