#pragma once

#include <vector>

#include "Core/Buffer.h"
#include "Core/Macros.h"
#include "Shape.h"

namespace CRay {

struct CRAYSTAL_API VertexData {
    Float3 position;
    Float3 normal;
    Float2 texCrd;
};

struct CRAYSTAL_API TriangleData {
    VertexData vData[3];

    CRAYSTAL_DEVICE_HOST Float3 getFaceNormal() const;
};

struct CRAYSTAL_API MeshDesc {
    uint32_t indexOffset;
    uint32_t indexCount;
    uint32_t vertexOffset;
};

/** Host side mesh data storage.
 */
struct CRAYSTAL_API MeshData {
    std::vector<Float3> position;
    std::vector<Float3> normal;
    std::vector<Float2> texCrd;
    std::vector<uint32_t> index;
};

class CRAYSTAL_API TriangleMeshManager : public ShapeManagerBase {
   public:
    using Ref = std::shared_ptr<TriangleMeshManager>;
    struct CRAYSTAL_API DeviceView {
        uint32_t nMesh;
        MeshDesc* pMeshDescs;

        // Global vertex buffer
        Float3* pPosition;
        Float3* pNormal;
        Float2* pTexCrd;
        // Global vertex buffer
        uint32_t* pIndex;

        CRAYSTAL_DEVICE bool intersect(PrimitiveID id, const Ray& ray,
                                       HitInfo& hitInfo, Float& hitT) const;

        CRAYSTAL_DEVICE TriangleData getTriangle(PrimitiveID id) const;
    };

    TriangleMeshManager(const std::vector<MeshData>& meshData);

    DeviceView getDeviceView() const { return mView; }

   private:
    DeviceView mView;

    std::unique_ptr<DeviceBuffer> mpMeshDescBuffer;
    std::unique_ptr<DeviceBuffer> mpPositionBuffer;
    std::unique_ptr<DeviceBuffer> mpNormalBuffer;
    std::unique_ptr<DeviceBuffer> mpTexCrdBuffer;
    std::unique_ptr<DeviceBuffer> mpIndexBuffer;
};

using TriangleMeshSOA = TriangleMeshManager::DeviceView;

template <>
struct CRAYSTAL_API ShapeSOATraits<TriangleMeshSOA> {
    static CRAYSTAL_DEVICE bool intersect(const TriangleMeshSOA& soa,
                                          PrimitiveID id, const Ray& ray,
                                          HitInfo& hitInfo, Float& hitT) {
        hitInfo.type = HitType::Triangle;
        return soa.intersect(id, ray, hitInfo, hitT);
    }
};

}  // namespace CRay
