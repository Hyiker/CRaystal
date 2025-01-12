#include "Spectrum.h"

namespace CRay {

Float3 XYZ2RGB(const Float3& xyz) { return {}; }
Float3 RGB2XYZ(const Float3& rgb) { return {}; }

Float3 RGBSpectrum::toXYZImpl() const {
    return Float3(mData[0], mData[0], mData[0]);
}

Float3 RGBSpectrum::toRGBImpl() const { return XYZ2RGB(toXYZImpl()); }

Float3 BroadSpectrum::toXYZImpl() const {
    return Float3(mData[0], mData[0], mData[0]);
}

Float3 BroadSpectrum::toRGBImpl() const {
    return Float3(mData[0], mData[0], mData[0]);
}

}  // namespace CRay
