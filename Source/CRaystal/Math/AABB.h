#pragma once
#include <limits>
#include <utility>

#include "Core/Macros.h"
#include "Core/Vec.h"
#include "MathDefs.h"
namespace CRay {
struct Ray;
/** Axis-aligned bounding box.
 */
struct CRAYSTAL_API AABB {
    Float3 pMin;  ///< Minimum corner.
    Float3 pMax;  ///< Maximum corner.

    struct CRAYSTAL_API iterator {
        Float3 current;
        int index;

        iterator(Float3 min, const Float3& max, int idx)
            : current(std::move(min)), index(idx) {
            if (index < 8) {
                if (index & 1) current.x = max.x;
                if (index & 2) current.y = max.y;
                if (index & 4) current.z = max.z;
            }
        }

        Float3 operator*() const { return current; }

        iterator& operator++() {
            ++index;
            return *this;
        }

        bool operator!=(const iterator& other) const {
            return index != other.index;
        }
    };

    [[nodiscard]] iterator begin() const { return {pMin, pMax, 0}; }
    [[nodiscard]] iterator end() const { return {pMin, pMax, 8}; }

    /** Construct an empty AABB.
     */
    CRAYSTAL_DEVICE_HOST AABB()
        : pMin(Float3(kFltInf)), pMax(Float3(-kFltInf)) {}
    /** Construct an AABB from two corners.
     * @param min Minimum corner.
     * @param max Maximum corner.
     */
    CRAYSTAL_DEVICE_HOST AABB(Float3 min, Float3 max)
        : pMin(std::move(min)), pMax(std::move(max)) {}
    /** Construct an AABB from a point.
     * @param p Point.
     */
    CRAYSTAL_DEVICE_HOST explicit AABB(const Float3& p) : pMin(p), pMax(p) {}

    // AABB boolean operations
    /** Expand the AABB to include a point.
     * @param p Point.
     */
    [[nodiscard]] CRAYSTAL_DEVICE_HOST AABB include(const Float3& p) const {
        AABB ret = *this;
        ret.pMin = min(pMin, p);
        ret.pMax = max(pMax, p);
        return ret;
    }
    /** Expand the AABB to include another AABB.
     * @param aabb AABB.
     */
    [[nodiscard]] CRAYSTAL_DEVICE_HOST AABB include(const AABB& aabb) const {
        AABB ret = *this;
        ret.pMin = min(pMin, aabb.pMin);
        ret.pMax = max(pMax, aabb.pMax);
        return ret;
    }

    /** Get the intersection of two AABBs.
     */
    [[nodiscard]] CRAYSTAL_DEVICE_HOST AABB intersect(const AABB& aabb) const {
        return {max(pMin, aabb.pMin), min(pMax, aabb.pMax)};
    }

    /** Check if the AABB is empty.
     */
    [[nodiscard]] CRAYSTAL_DEVICE_HOST bool isEmpty() const {
        return pMin.x > pMax.x || pMin.y > pMax.y || pMin.z > pMax.z;
    }
    /** Get the center of the AABB.
     */
    [[nodiscard]] CRAYSTAL_DEVICE_HOST Float3 center() const {
        return (pMin + pMax) / Float(2.0);
    }
    /** Get the diagonal of the AABB.
     */
    [[nodiscard]] CRAYSTAL_DEVICE_HOST Float3 diagonal() const {
        return pMax - pMin;
    }
    /** Check if a point is inside the AABB.
     */
    [[nodiscard]] CRAYSTAL_DEVICE_HOST bool inside(const Float3& p) const {
        return p.x >= pMin.x && p.x <= pMax.x && p.y >= pMin.y &&
               p.y <= pMax.y && p.z >= pMin.z && p.z <= pMax.z;
    }

    CRAYSTAL_DEVICE_HOST AABB operator|(const AABB& aabb) const {
        return include(aabb);
    }
    CRAYSTAL_DEVICE_HOST AABB operator|(const Float3& p) const {
        return include(p);
    }
    CRAYSTAL_DEVICE_HOST AABB operator&(const AABB& aabb) const {
        return intersect(aabb);
    }
    CRAYSTAL_DEVICE_HOST AABB& operator|=(const AABB& aabb) {
        *this = include(aabb);
        return *this;
    }
    CRAYSTAL_DEVICE_HOST AABB& operator|=(const Float3& p) {
        *this = include(p);
        return *this;
    }

    CRAYSTAL_DEVICE_HOST AABB& operator&=(const AABB& aabb) {
        *this = intersect(aabb);
        return *this;
    }

    CRAYSTAL_DEVICE_HOST [[nodiscard]] bool operator==(const AABB& aabb) const {
        return pMin == aabb.pMin && pMax == aabb.pMax;
    }
    CRAYSTAL_DEVICE_HOST [[nodiscard]] bool operator!=(const AABB& aabb) const {
        return !(aabb == *this);
    }

    CRAYSTAL_DEVICE_HOST bool intersect(const Ray& ray, Float& hitT) const;
};
}  // namespace CRay
