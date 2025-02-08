#pragma once
#include <cinttypes>

#include "Macros.h"
#include "Vec.h"
namespace CRay {

/** PCG pseudo random sampler.
 */
class CRAYSTAL_API Sampler {
   public:
    CRAYSTAL_DEVICE_HOST Sampler(const UInt2& pixel, uint32_t sampleIndex);

    CRAYSTAL_DEVICE_HOST Float nextSample1D();

    CRAYSTAL_DEVICE_HOST Float2 nextSample2D();

   private:
    CRAYSTAL_DEVICE_HOST uint32_t nextState();

    CRAYSTAL_DEVICE_HOST Float stateToFloat(uint32_t state);

    uint32_t mState;
    uint32_t mIncrement;
    uint32_t mSampleIndex;
};

}  // namespace CRay
