#include "Frame.h"
#include "Math/CRayMath.h"
#include "Math/MathDefs.h"
namespace CRay {

CRAYSTAL_DEVICE_HOST Frame::Frame(Float3 normal)
    : Frame(normal, createTangentSafe(normal)) {}

CRAYSTAL_DEVICE_HOST Frame::Frame(Float3 normal, Float3 tangent) {
    N = normal;
    T = tangent;
    B = cross(N, T);
}

CRAYSTAL_DEVICE_HOST Float3 Frame::createTangentSafe(const Float3& normal) {
    Float3 tangent = Float3(0, 1, 0);
    if (1.0 - absDot(normal, tangent) < kEps) {
        tangent = Float3(1, 0, 0);
    }
    Float3 bitangent = cross(normal, tangent);
    return cross(bitangent, normal);
}

CRAYSTAL_DEVICE_HOST Float3 Frame::toLocal(const Float3& world) const {
    return Float3x3(T.x, B.x, N.x, T.y, B.y, N.y, T.z, B.z, N.z) * world;
}

CRAYSTAL_DEVICE_HOST Float3 Frame::toWorld(const Float3& local) const {
    return Float3x3(T, B, N) * local;
}

}  // namespace CRay
