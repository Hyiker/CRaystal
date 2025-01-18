#pragma once
#include <algorithm>

#include "Macros.h"
#include <glm/glm.hpp>

namespace CRay {
using UInt = uint32_t;
using UInt2 = glm::uvec2;
using UInt3 = glm::uvec3;
using UInt4 = glm::uvec4;

using Int2 = glm::ivec2;
using Int3 = glm::ivec3;
using Int4 = glm::ivec4;

using Bool2 = glm::bvec2;
using Bool3 = glm::bvec3;
using Bool4 = glm::bvec4;

// precision 32 float
using Float2p32 = glm::vec2;
using Float3p32 = glm::vec3;
using Float4p32 = glm::vec4;

using Float3x3p32 = glm::mat3x3;
using Float4x4p32 = glm::mat4x4;
using Float3x4p32 = glm::mat3x4;

using Quat32 = glm::qua<glm::f32>;

// precision 64 float
using Float2p64 = glm::dvec2;
using Float3p64 = glm::dvec3;
using Float4p64 = glm::dvec4;

using Float3x3p64 = glm::dmat3x3;
using Float4x4p64 = glm::dmat4x4;
using Float3x4p64 = glm::dmat3x4;

using Quat64 = glm::qua<glm::f64>;

#if CRAYSTAL_ENABLE_F64
using Float2 = Float2p64;
using Float3 = Float3p64;
using Float4 = Float4p64;

using Float3x3 = Float3x3p64;
using Float4x4 = Float4x4p64;
using Float3x4 = Float3x4p64;

using Quat = Quat64;
#else
using Float2 = Float2p32;
using Float3 = Float3p32;
using Float4 = Float4p32;

using Float3x3 = Float3x3p32;
using Float4x4 = Float4x4p32;
using Float3x4 = Float3x4p32;

using Quat = Quat32;
#endif

CRAYSTAL_API CRAYSTAL_DEVICE_HOST inline Float3 cross(const Float3& v0,
                                                      const Float3& v1) {
    return glm::cross(v0, v1);
}

template <typename T>
CRAYSTAL_DEVICE_HOST bool any(const T& v) {
    return glm::any(v);
}

template <typename T>
CRAYSTAL_DEVICE_HOST bool all(const T& v) {
    return glm::all(v);
}

// Transform functions use Right-handed, y-up coordinates

/**
 * Builds a right-handed look-at view matrix.
 * This matrix defines a view transformation where the camera is positioned at
 * `pos`, looking towards `target`, with the `up` vector specifying the camera's
 * orientation.
 *
 * @param pos The position of the camera in world space coordinates.
 * @param target The point in world space coordinates that the camera is looking
 * at.
 * @param up The direction vector representing the upward orientation of the
 * camera.
 * @return A 4x4 transformation matrix representing the view matrix.
 */
CRAYSTAL_API CRAYSTAL_DEVICE_HOST Float4x4 lookAt(const Float3& pos,
                                                  const Float3& target,
                                                  const Float3& up);

/**
 * Creates a matrix for a right-handed, symmetric perspective-view frustum.
 * This matrix is useful for projecting 3D scenes onto a 2D surface, typically
 * used in graphics rendering for creating perspective depth. The frustum
 * defines a visible area based on the field of view and aspect ratio, with near
 * and far clipping planes in Direct3D clip volume coordinates (0 to +1).
 *
 * @param fovY Field of view in the Y axis, in radians. This defines the
 * vertical angle of the view.
 * @param aspectRatio The aspect ratio of the view window, defined as width
 * divided by height.
 * @param near The distance to the near clipping plane, which must be positive.
 * @param far The distance to the far clipping plane, which must be greater than
 * `near`.
 * @return A 4x4 transformation matrix representing the perspective projection
 * matrix.
 */
CRAYSTAL_API Float4x4 perspective(Float fovY, Float aspectRatio, Float near,
                                  Float far);

template <typename T>
T transpose(T mat) {
    return glm::transpose(mat);
}

template <typename T>
T inverse(T mat) {
    return glm::inverse(mat);
}

}  // namespace CRay
