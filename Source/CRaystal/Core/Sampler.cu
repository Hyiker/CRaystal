#include "Sampler.h"

namespace CRay {
CRAYSTAL_DEVICE_HOST Sampler::Sampler(const UInt2& pixel, uint32_t sampleIndex)
    : mSampleIndex(sampleIndex) {
    mState = pixel.x * 2654435761u + pixel.y * 1597334677u + sampleIndex;
    mIncrement = pixel.x * 1234567891u + pixel.y * 987654321u + 1;
    for (int i = 0; i < 4; ++i) {
        nextState();
    }
}

CRAYSTAL_DEVICE_HOST Float Sampler::nextSample1D() {
    return stateToFloat(nextState());
}

CRAYSTAL_DEVICE_HOST Float2 Sampler::nextSample2D() {
    return Float2(stateToFloat(nextState()), stateToFloat(nextState()));
}

CRAYSTAL_DEVICE_HOST uint32_t Sampler::nextState() {
    uint32_t oldState = mState;
    mState = oldState * 747796405u + mIncrement;

    oldState = ((oldState >> ((oldState >> 28u) + 4u)) ^ oldState) * 277803737u;
    return (oldState >> 22u) ^ oldState;
}

CRAYSTAL_DEVICE_HOST Float Sampler::stateToFloat(uint32_t state) {
    return Float(state) * (1.0f / 4294967296.0f);
}
}  // namespace CRay
