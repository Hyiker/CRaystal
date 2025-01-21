#pragma once

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>

#include "Core/Macros.h"

namespace CRay {
constexpr Float kPi = 3.14159265358979323846f;
constexpr Float kInvPi = 1.f / kPi;

constexpr Float kFltInf = std::numeric_limits<Float>::infinity();
}  // namespace CRay
