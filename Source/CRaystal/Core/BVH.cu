#include <algorithm>
#include <numeric>

#include "BVH.h"
#include "Scene/Scene.h"
namespace CRay {

struct StackEntry {
    uint32_t nodeIndex;
    float tMin;
};

CRAYSTAL_DEVICE bool BVHData::intersect(const TriangleMeshSOA& meshSOA,
                                        RayHit& rayHit) const {
    Float _;
    if (!blas.bounds.intersect(rayHit.ray, _)) {
        return false;
    }

    constexpr int MAX_STACK = 64;
    StackEntry stack[MAX_STACK];
    int stackPtr = 0;

    stack[stackPtr++] = {blas.rootIndex, 0.0f};

    bool isHit = false;
    float closestT = rayHit.hitT;

    while (stackPtr > 0) {
        const StackEntry entry = stack[--stackPtr];
        const BVHNode& node = blasNodes[entry.nodeIndex];

        if (entry.tMin > closestT) {
            continue;
        }

        if (std::holds_alternative<BVHNode::LeafProp>(node.props)) {
            const auto& leafProp = std::get<BVHNode::LeafProp>(node.props);
            for (uint32_t i = 0; i < leafProp.leafCount; ++i) {
                PrimitiveID primID = leafProp.leafOffset + i;

                if (intersectShape(primID, meshSOA, rayHit)) {
                    isHit = true;
                    // closestT = rayHit.hitT;
                }
            }
        } else {
            const auto& internalProp =
                std::get<BVHNode::InternalProp>(node.props);
            const BVHNode& leftChild = blasNodes[internalProp.left];
            const BVHNode& rightChild = blasNodes[internalProp.right];

            float leftT = kFltInf;
            float rightT = kFltInf;
            bool hitLeft = leftChild.bounds.intersect(rayHit.ray, leftT);
            bool hitRight = rightChild.bounds.intersect(rayHit.ray, rightT);

            if (hitLeft && hitRight) {
                if (leftT > rightT) {
                    if (leftT < closestT) {
                        stack[stackPtr++] = {internalProp.left, leftT};
                    }
                    if (rightT < closestT) {
                        stack[stackPtr++] = {internalProp.right, rightT};
                    }
                } else {
                    if (rightT < closestT) {
                        stack[stackPtr++] = {internalProp.right, rightT};
                    }
                    if (leftT < closestT) {
                        stack[stackPtr++] = {internalProp.left, leftT};
                    }
                }
            } else if (hitLeft && leftT < closestT) {
                stack[stackPtr++] = {internalProp.left, leftT};
            } else if (hitRight && rightT < closestT) {
                stack[stackPtr++] = {internalProp.right, rightT};
            }
        }
    }

    return isHit;
}

BVH::BVH() {}

void BVH::build(SceneData& data) {
    logInfo("Building BVH");
    // 1. Collect triangle data
    gatherTriangles(data);

    // 2. Prepare data
    std::vector<uint32_t> indices(mBuildTris.size());
    std::iota(indices.begin(), indices.end(), 0);

    mNodes.clear();
    mNodes.reserve(2 * mBuildTris.size());

    // 3. Recursive construction
    AABB rootBounds;
    uint32_t rootIndex =
        buildRecursive(indices, 0, indices.size(), 0, rootBounds);

    // 4. Reorder mesh data
    createFinalBVH(data, indices, rootBounds, rootIndex);

    mNodes.clear();
    mBuildTris.clear();

    logInfo(
        "BVH build stats:\n\t\t\tMax depth: {}\n\t\t\tMax leaf node: "
        "{}\n\t\t\tNode count: {}",
        mBuildStats.maxDepth, mBuildStats.maxLeafNodeCount,
        mBuildStats.nodeCount);
}

BVH::DeviceView BVH::getDeviceView() const { return mView; }

void BVH::gatherTriangles(const SceneData& data) {
    size_t totalTris = 0;
    for (const auto& mesh : data.meshes) {
        totalTris += mesh.index.size() / 3;
    }

    mBuildTris.clear();
    mBuildTris.reserve(totalTris);

    for (size_t meshIdx = 0; meshIdx < data.meshes.size(); ++meshIdx) {
        const auto& mesh = data.meshes[meshIdx];
        for (size_t i = 0; i < mesh.index.size(); i += 3) {
            BuildTriangle tri;
            Float3 v0 = mesh.position[mesh.index[i]];
            Float3 v1 = mesh.position[mesh.index[i + 1]];
            Float3 v2 = mesh.position[mesh.index[i + 2]];

            tri.centroid = (v0 + v1 + v2) / 3.0f;
            tri.bounds |= v0;
            tri.bounds |= v1;
            tri.bounds |= v2;
            tri.meshIndex = meshIdx;
            tri.triIndex = i / 3;

            mBuildTris.push_back(tri);
        }
    }
}

uint32_t BVH::buildRecursive(std::vector<uint32_t>& indices, uint32_t start,
                             uint32_t count, uint32_t depth, AABB& nodeBounds) {
    nodeBounds = AABB();
    for (uint32_t i = start; i < start + count; ++i) {
        nodeBounds |= mBuildTris[indices[i]].bounds;
    }

    BVHNode node;
    node.bounds = nodeBounds;

    // Leaf node
    if (count <= MIN_TRIS_PER_LEAF || depth >= MAX_DEPTH) {
        BVHNode::LeafProp leaf;
        leaf.leafOffset = start;
        leaf.leafCount = count;
        node.props = leaf;
        mNodes.push_back(node);
        mBuildStats.maxLeafNodeCount =
            std::max<uint32_t>(mBuildStats.maxLeafNodeCount, count);
        return mNodes.size() - 1;
    }

    auto split = findBestSplit(indices, start, count, nodeBounds);
    uint32_t splitAxis = split.first;
    float splitPos = split.second;

    // Split triangles
    auto mid = std::partition(
        indices.begin() + start, indices.begin() + start + count,
        [&](uint32_t idx) {
            return mBuildTris[idx].centroid[splitAxis] < splitPos;
        });

    uint32_t leftCount = std::distance(indices.begin() + start, mid);

    // Create leaf if split failed
    if (leftCount == 0 || leftCount == count) {
        BVHNode::LeafProp leaf;
        leaf.leafOffset = start;
        leaf.leafCount = count;
        node.props = leaf;
        mNodes.push_back(node);
        mBuildStats.maxLeafNodeCount =
            std::max<uint32_t>(mBuildStats.maxLeafNodeCount, count);
        return mNodes.size() - 1;
    }

    // Construct children recursively
    AABB leftBounds, rightBounds;
    uint32_t leftChild =
        buildRecursive(indices, start, leftCount, depth + 1, leftBounds);
    uint32_t rightChild = buildRecursive(
        indices, start + leftCount, count - leftCount, depth + 1, rightBounds);

    BVHNode::InternalProp internal;
    internal.left = leftChild;
    internal.right = rightChild;
    node.props = internal;
    mNodes.push_back(node);

    mBuildStats.maxDepth = std::max(mBuildStats.maxDepth, depth);
    return mNodes.size() - 1;
}

std::pair<uint32_t, Float> BVH::findBestSplit(
    const std::vector<uint32_t>& indices, uint32_t start, uint32_t count,
    const AABB& nodeBounds) {
    Float bestCost = kFltInf;
    uint32_t bestAxis = 0;
    Float bestSplit = 0.0f;

    // Split each axis
    for (uint32_t axis = 0; axis < 3; ++axis) {
        // Init bins
        std::array<SAHBin, NUM_BINS> bins;
        Float3 extent = nodeBounds.diagonal();
        Float scale = NUM_BINS / extent[axis];

        // Assign triangle to bins
        for (uint32_t i = start; i < start + count; ++i) {
            const auto& tri = mBuildTris[indices[i]];
            int binIndex = std::min(
                NUM_BINS - 1,
                static_cast<int>((tri.centroid[axis] - nodeBounds.pMin[axis]) *
                                 scale));
            bins[binIndex].bounds |= tri.bounds;
            bins[binIndex].triCount++;
        }

        // Scan best split from left to right
        std::array<AABB, NUM_BINS - 1> leftBounds;
        std::array<AABB, NUM_BINS - 1> rightBounds;
        std::array<int, NUM_BINS - 1> leftCount{0};
        std::array<int, NUM_BINS - 1> rightCount{0};

        // Left accumulate
        AABB currentLeft;
        int currentLeftCount = 0;
        for (int i = 0; i < NUM_BINS - 1; ++i) {
            currentLeft |= bins[i].bounds;
            currentLeftCount += bins[i].triCount;
            leftBounds[i] = currentLeft;
            leftCount[i] = currentLeftCount;
        }

        // Right accumulate
        AABB currentRight;
        int currentRightCount = 0;
        for (int i = NUM_BINS - 1; i > 0; --i) {
            currentRight |= bins[i].bounds;
            currentRightCount += bins[i].triCount;
            rightBounds[i - 1] = currentRight;
            rightCount[i - 1] = currentRightCount;
        }

        // Evaluate possible splits
        for (int i = 0; i < NUM_BINS - 1; ++i) {
            Float splitPos =
                nodeBounds.pMin[axis] + (i + 1) * extent[axis] / NUM_BINS;

            Float cost = evaluateSAH(nodeBounds, leftBounds[i], rightBounds[i],
                                     leftCount[i], rightCount[i]);

            if (cost < bestCost) {
                bestCost = cost;
                bestAxis = axis;
                bestSplit = splitPos;
            }
        }
    }
    return {bestAxis, bestSplit};
}

Float calcAABBSurfaceArea(const AABB& aabb) {
    Float3 xyz = aabb.diagonal();
    return (xyz.x * xyz.y + xyz.x * xyz.z + xyz.y * xyz.z) * 2.0;
}

Float BVH::evaluateSAH(const AABB& nodeBounds, const AABB& leftBounds,
                       const AABB& rightBounds, int leftCount, int rightCount) {
    const float traversalCost = 1.0f;
    const float intersectionCost = 1.0f;

    Float leftSA = calcAABBSurfaceArea(leftBounds);
    Float rightSA = calcAABBSurfaceArea(rightBounds);
    Float rootSA = calcAABBSurfaceArea(nodeBounds);

    return traversalCost + intersectionCost *
                               (leftCount * leftSA + rightCount * rightSA) /
                               rootSA;
}

void BVH::createFinalBVH(SceneData& data,
                         const std::vector<uint32_t>& finalIndices,
                         const AABB& rootBounds, uint32_t rootIndex) {
    mpDeviceNodeData =
        std::make_unique<DeviceBuffer>(sizeof(BVHNode) * mNodes.size());

    mpDeviceNodeData->copyFromHost(mNodes.data());

    mView.blas.bounds = rootBounds;
    mView.blas.rootIndex = rootIndex;

    // Create node data
    mView.blasNodes = (BVHNode*)mpDeviceNodeData->data();

    reorderMeshData(data, finalIndices);

    mBuildStats.nodeCount = mNodes.size();
}

void BVH::reorderMeshData(SceneData& data,
                          const std::vector<uint32_t>& finalIndices) {
    // Create reordered data for each mesh
    std::vector<MeshData> reorderedMeshes(data.meshes.size());

    for (size_t i = 0; i < finalIndices.size(); ++i) {
        const auto& tri = mBuildTris[finalIndices[i]];
        auto& srcMesh = data.meshes[tri.meshIndex];
        auto& dstMesh = reorderedMeshes[tri.meshIndex];

        for (int j = 0; j < 3; ++j) {
            uint32_t srcIdx = srcMesh.index[tri.triIndex * 3 + j];

            dstMesh.position.push_back(srcMesh.position[srcIdx]);
            dstMesh.normal.push_back(srcMesh.normal[srcIdx]);
            dstMesh.texCrd.push_back(srcMesh.texCrd[srcIdx]);
            dstMesh.index.push_back(dstMesh.position.size() - 1);
        }
    }

    data.meshes = std::move(reorderedMeshes);
}

BVH::~BVH() {}

}  // namespace CRay
