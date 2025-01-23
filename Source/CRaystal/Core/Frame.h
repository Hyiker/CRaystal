#pragma once
#include "Macros.h"
#include "Vec.h"
namespace CRay {

/** Local frame utility class.
 */
struct CRAYSTAL_API Frame {
    Float3 T = Float3(1, 0, 0);  ///< Tangent
    Float3 B = Float3(0, 1, 0);  ///< Bitangent
    Float3 N = Float3(0, 0, 1);  ///< Normal

    CRAYSTAL_DEVICE_HOST Frame() = default;

    CRAYSTAL_DEVICE_HOST Frame(Float3 normal);
    CRAYSTAL_DEVICE_HOST Frame(Float3 normal, Float3 tangent);

    static CRAYSTAL_DEVICE_HOST Float3 createTangentSafe(const Float3& normal);

    CRAYSTAL_DEVICE_HOST Float3 toLocal(const Float3& world) const;
    CRAYSTAL_DEVICE_HOST Float3 toWorld(const Float3& local) const;
};

}  // namespace CRay
