#pragma once

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>

#include "Core/Macros.h"

namespace CRay {

constexpr Float kEps = 1e-6f;
constexpr Float kPi = 3.14159265358979323846f;
constexpr Float kInvPi = 1.f / kPi;
constexpr Float kInv2Pi = 0.5f / kPi;
constexpr Float kInv4Pi = 0.25f / kPi;

constexpr Float kFltInf = std::numeric_limits<Float>::infinity();
constexpr Float kFltEps = std::numeric_limits<Float>::epsilon();
}  // namespace CRay
