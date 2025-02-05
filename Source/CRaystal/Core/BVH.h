#pragma once
#include <memory>
#include <variant>
#include <vector>

#include "Buffer.h"
#include "Macros.h"
#include "Math/AABB.h"
#include "Math/Ray.h"
#include "Object.h"
#include "Scene/Triangle.h"

namespace CRay {

struct SceneData;

struct CRAYSTAL_API BVHNode {
    struct LeafProp {
        uint32_t leafOffset;
        uint32_t leafCount;
    };

    struct InternalProp {
        uint32_t left;
        uint32_t right;
    };

    std::variant<LeafProp, InternalProp> props;
    AABB bounds;
};

struct CRAYSTAL_API BLAS {
    AABB bounds;
    uint32_t rootIndex;
};

struct CRAYSTAL_API BVHData {
    BLAS blas;

    BVHNode* blasNodes;

    CRAYSTAL_DEVICE bool intersect(const TriangleMeshSOA& meshSOA,
                                   RayHit& rayHit) const;
};

class CRAYSTAL_API BVH {
   public:
    using Ref = std::shared_ptr<BVH>;
    using DeviceView = BVHData;

    BVH();

    /** Build BVH from scene data.
     */
    void build(SceneData& data);

    DeviceView getDeviceView() const;

    ~BVH();

   private:
    struct BuildStats {
        uint32_t maxDepth = 0u;
        uint32_t maxLeafNodeCount = 0u;

        uint32_t nodeCount = 0u;
    };

    /** Auxiliary build triangle data.
     */
    struct BuildTriangle {
        Float3 centroid;
        AABB bounds;
        uint32_t meshIndex;
        uint32_t triIndex;
    };

    struct SAHBin {
        AABB bounds;
        uint32_t triCount = 0;
    };

    void gatherTriangles(const SceneData& data);

    uint32_t buildRecursive(std::vector<uint32_t>& indices, uint32_t start,
                            uint32_t count, uint32_t depth, AABB& nodeBounds);

    std::pair<uint32_t, float> findBestSplit(
        const std::vector<uint32_t>& indices, uint32_t start, uint32_t count,
        const AABB& nodeBounds);

    float evaluateSAH(const AABB& nodeBounds, const AABB& leftBounds,
                      const AABB& rightBounds, int leftCount, int rightCount);

    void createFinalBVH(SceneData& data,
                        const std::vector<uint32_t>& finalIndices,
                        const AABB& rootBounds, uint32_t rootIndex);

    void reorderMeshData(SceneData& data,
                         const std::vector<uint32_t>& finalIndices);

    static constexpr int NUM_BINS = 32;
    static constexpr uint32_t MIN_TRIS_PER_LEAF = 4;
    static constexpr uint32_t MAX_DEPTH = 20;

    // Construction internal data
    std::vector<BVHNode> mNodes;
    std::vector<BuildTriangle> mBuildTris;
    BuildStats mBuildStats;

    // View data
    DeviceView mView;

    // GPU Buffers
    std::unique_ptr<DeviceBuffer> mpDeviceNodeData;
};

}  // namespace CRay
