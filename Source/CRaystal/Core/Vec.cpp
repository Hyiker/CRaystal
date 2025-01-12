#include "Vec.h"

#include <glm/gtc/matrix_transform.hpp>

namespace CRay {

Float4x4 lookAt(const Float3& pos, const Float3& target, const Float3& up) {
    return glm::lookAtRH(pos, target, up);
}

Float4x4 perspective(Float fovY, Float aspectRatio, Float near, Float far) {
    return glm::perspectiveRH_ZO(fovY, aspectRatio, near, far);
}

}  // namespace CRay
