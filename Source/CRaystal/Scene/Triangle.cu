#include <algorithm>
#include <iterator>

#include "Core/Error.h"
#include "Math/MathDefs.h"
#include "Triangle.h"
namespace CRay {

CRAYSTAL_DEVICE_HOST Float3 TriangleData::getFaceNormal() const {
    Float3 edge1 = vData[1].position - vData[0].position;
    Float3 edge2 = vData[2].position - vData[0].position;

    Float3 normal = cross(edge1, edge2);

    return normalize(normal);
}

CRAYSTAL_DEVICE bool TriangleMeshSOA::intersect(PrimitiveID id, const Ray& ray,
                                                HitInfo& hitInfo,
                                                Float& hitT) const {
    uint32_t iIndex = id * 3;
    // Fetch positions
    uint32_t vIndex[3]{pIndex[iIndex], pIndex[iIndex + 1], pIndex[iIndex + 2]};

    Float3 v0 = pPosition[vIndex[0]];
    Float3 v1 = pPosition[vIndex[1]];
    Float3 v2 = pPosition[vIndex[2]];

    // Möller-Trumbore ray-triangle intersection
    Float3 e1 = v1 - v0;
    Float3 e2 = v2 - v0;
    Float3 p = cross(ray.direction, e2);
    Float det = dot(e1, p);

    if (std::abs(det) < 1e-6f) return false;

    float inv_det = 1.0f / det;
    Float3 tvec = ray.origin - v0;
    Float u = dot(tvec, p) * inv_det;
    if (u < 0.0f || u > 1.0f) return false;

    Float3 q = cross(tvec, e1);
    Float v = dot(ray.direction, q) * inv_det;
    if (v < 0.0f || u + v > 1.0f) return false;

    Float t = dot(e2, q) * inv_det;
    if (t <= 0.0) return false;

    hitInfo.barycentric = Float2(u, v);
    hitT = t;

    return true;
}

CRAYSTAL_DEVICE TriangleData
TriangleMeshSOA::getTriangle(PrimitiveID id) const {
    uint32_t iIndex = id * 3;

    TriangleData data;
    uint32_t vIndex[3]{pIndex[iIndex], pIndex[iIndex + 1], pIndex[iIndex + 2]};

    [[unroll]]
    for (int i = 0; i < 3; i++) {
        data.vData[i].position = pPosition[vIndex[i]];
        data.vData[i].normal = pNormal[vIndex[i]];
        data.vData[i].texCrd = pTexCrd[vIndex[i]];
    }

    return data;
}

CRAYSTAL_DEVICE MeshDesc TriangleMeshSOA::getMeshDesc(PrimitiveID id) const {
    return pMeshDescs[pPrimitiveToMesh[id]];
}

TriangleMeshManager::TriangleMeshManager(
    const std::vector<MeshData>& meshData) {
    std::vector<Float3> position;
    std::vector<Float3> normal;
    std::vector<Float2> texCrd;
    std::vector<uint32_t> index;
    std::vector<MeshDesc> descs;

    for (const MeshData& mesh : meshData) {
        MeshDesc desc;
        desc.indexCount = mesh.index.size();
        desc.indexOffset = index.size();
        desc.vertexOffset = position.size();
        desc.materialID = mesh.materialID;

        position.insert(position.end(), mesh.position.begin(),
                        mesh.position.end());
        normal.insert(normal.end(), mesh.normal.begin(), mesh.normal.end());
        texCrd.insert(texCrd.end(), mesh.texCrd.begin(), mesh.texCrd.end());
        // TODO: use better indexing
        std::transform(mesh.index.begin(), mesh.index.end(),
                       std::back_inserter(index),
                       [offset = desc.vertexOffset](uint32_t value) {
                           return value + offset;
                       });
        descs.push_back(desc);
    }

    std::vector<uint32_t> primitiveToMesh(index.size() / 3);
    uint32_t currentMeshID = 0;
    for (const auto& desc : descs) {
        uint32_t startPrimitive = desc.indexOffset / 3u;
        uint32_t primitiveCount = desc.indexCount / 3u;

        std::fill_n(primitiveToMesh.begin() + startPrimitive, primitiveCount,
                    currentMeshID);

        currentMeshID++;
    };

    mpMeshDescBuffer =
        std::make_unique<DeviceBuffer>(sizeof(MeshDesc) * descs.size());
    mpPositionBuffer =
        std::make_unique<DeviceBuffer>(sizeof(Float3) * position.size());
    mpNormalBuffer =
        std::make_unique<DeviceBuffer>(sizeof(Float3) * normal.size());
    mpTexCrdBuffer =
        std::make_unique<DeviceBuffer>(sizeof(Float2) * texCrd.size());
    mpIndexBuffer =
        std::make_unique<DeviceBuffer>(sizeof(uint32_t) * index.size());
    mpPrimitiveToMeshBuffer = std::make_unique<DeviceBuffer>(
        sizeof(uint32_t) * primitiveToMesh.size());

    mpMeshDescBuffer->copyFromHost(descs.data());
    mpPositionBuffer->copyFromHost(position.data());
    mpNormalBuffer->copyFromHost(normal.data());
    mpTexCrdBuffer->copyFromHost(texCrd.data());
    mpIndexBuffer->copyFromHost(index.data());
    mpPrimitiveToMeshBuffer->copyFromHost(primitiveToMesh.data());

    mView.nMesh = descs.size();
    mView.pMeshDescs = (MeshDesc*)mpMeshDescBuffer->data();
    mView.pPosition = (Float3*)mpPositionBuffer->data();
    mView.pNormal = (Float3*)mpNormalBuffer->data();
    mView.pTexCrd = (Float2*)mpTexCrdBuffer->data();
    mView.pIndex = (uint32_t*)mpIndexBuffer->data();
    mView.pPrimitiveToMesh = (uint32_t*)mpPrimitiveToMeshBuffer->data();
}

}  // namespace CRay
