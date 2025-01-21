#include "Spectrum.h"

#include "SpectrumTabData.h"

namespace CRay {

BroadSpectrum gRGBRefl2SpectWhite;
BroadSpectrum gRGBRefl2SpectCyan;
BroadSpectrum gRGBRefl2SpectMagenta;
BroadSpectrum gRGBRefl2SpectYellow;
BroadSpectrum gRGBRefl2SpectRed;
BroadSpectrum gRGBRefl2SpectGreen;
BroadSpectrum gRGBRefl2SpectBlue;
BroadSpectrum gRGBIllum2SpectWhite;
BroadSpectrum gRGBIllum2SpectCyan;
BroadSpectrum gRGBIllum2SpectMagenta;
BroadSpectrum gRGBIllum2SpectYellow;
BroadSpectrum gRGBIllum2SpectRed;
BroadSpectrum gRGBIllum2SpectGreen;
BroadSpectrum gRGBIllum2SpectBlue;

Float3 XYZ2RGB(const Float3& xyz) { return kXYZ2SrgbMatrix * xyz; }
Float3 RGB2XYZ(const Float3& rgb) { return kSrgb2XYZMatrix * rgb; }

Float3 RGBSpectrum::toXYZ() const { return RGB2XYZ(toRGB()); }

Float3 RGBSpectrum::toRGB() const {
    return Float3(mData[0], mData[1], mData[2]);
}

RGBSpectrum RGBSpectrum::fromRGB(Float3 rgb, bool isIllum) {
    return RGBSpectrum(rgb);
}

void RGBSpectrum::initialize() {}

Float3 BroadSpectrum::toXYZ() const {
    Float3 xyz{};
    for (int i = 0; i < size(); i++) {
        xyz += Float3(kXYZTable[0][i], kXYZTable[1][i], kXYZTable[0][i]) *
               mData[i];
    }
    xyz *= kCIENormalizationY * kXYZStep;
    return xyz;
}

Float3 BroadSpectrum::toRGB() const { return XYZ2RGB(toXYZ()); }

BroadSpectrum BroadSpectrum::fromRGB(Float3 rgb, bool isIllum) { return {}; }

// static BroadSpectrum interpolateFromTabular(std::span<const double>
// wavelengths,
//                                             std::span<const double> values) {
//     CRAYSTAL_ASSERT(wavelengths.size() == values.size());

//     std::array<Float, kCIESampleCount> data;

//     Float wlMin = wavelengths.front();
//     Float wlMax = wavelengths.back();
//     Float step = (wlMax - wlMin) / (kCIESampleCount - 1);

//     int inputSampleCount = wavelengths.size();

//     auto wavelengthsClamped = [&](int index) -> Float {
//         if (index == -1) {
//             return wlMin - step;
//         } else if (index == inputSampleCount) {
//             return wlMax + step;
//         }
//         return wavelengths[index];
//     };

//     auto valuesClamped = [&](int index) {
//         return values[std::clamp(index, 0, inputSampleCount - 1)];
//     };

//     auto resample = [&](Float wavelen) -> Float {
//         if (wavelen + step / 2 <= wavelengths[0]) return values[0];
//         if (wavelen - step / 2 >= wavelengths[inputSampleCount - 1])
//             return values[inputSampleCount - 1];
//         if (inputSampleCount == 1) return values[0];

//         int start, end;
//         int target = wavelen - step;
//         if (target < wavelengths[0])
//             start = -1;
//         else {
//             auto it = std::lower_bound(wavelengths.begin(),
//             wavelengths.end(),
//                                        wavelen - step);
//             if (it != wavelengths.begin() &&
//                 (it == wavelengths.end() || *it > target)) {
//                 --it;
//             }
//             start = std::clamp<int>(std::distance(wavelengths.begin(), it),
//             0, wavelengths.size() - 2);
//         }

//         if (wavelen + step > wavelengths[inputSampleCount - 1])
//             end = inputSampleCount;
//         else {
//             end = start > 0 ? start : 0;
//             while (end < inputSampleCount && wavelen + step >
//             wavelengths[end])
//                 ++end;
//         }

//         if (end - start == 2 && wavelengthsClamped(start) <= target &&
//             wavelengths[start + 1] == wavelen &&
//             wavelengthsClamped(end) >= wavelen + step) {
//             return values[start + 1];
//         } else if (end - start == 1) {
//             Float t = (wavelen - wavelengthsClamped(start)) /
//                       (wavelengthsClamped(end) - wavelengthsClamped(start));
//             CRAYSTAL_ASSERT(t >= 0 && t <= 1);
//             return std::lerp(t, valuesClamped(start), valuesClamped(end));
//         } else {
//             return AverageSpectrumSamples(wavelengths, values,
//             inputSampleCount,
//                                           wavelen - step / 2,
//                                           wavelen + step / 2);
//         }
//     };

//     for (int outOffset = 0; outOffset < data.size(); ++outOffset) {
//         Float lambda = std::lerp<Float>(Float(outOffset) / (data.size() - 1),
//                                         wlMin, wlMax);
//         data[outOffset] = resample(lambda);
//     }

//     return BroadSpectrum(data);
// }

void BroadSpectrum::initialize() {
    // gRGBRefl2SpectWhite =
    //     interpolateFromTabular(RGB2SpectLambda, RGBRefl2SpectWhite);
    // gRGBRefl2SpectCyan =
    //     interpolateFromTabular(RGB2SpectLambda, RGBRefl2SpectCyan);
    // gRGBRefl2SpectMagenta =
    //     interpolateFromTabular(RGB2SpectLambda, RGBRefl2SpectMagenta);
    // gRGBRefl2SpectYellow =
    //     interpolateFromTabular(RGB2SpectLambda, RGBRefl2SpectYellow);
    // gRGBRefl2SpectRed =
    //     interpolateFromTabular(RGB2SpectLambda, RGBRefl2SpectRed);
    // gRGBRefl2SpectGreen =
    //     interpolateFromTabular(RGB2SpectLambda, RGBRefl2SpectGreen);
    // gRGBRefl2SpectBlue =
    //     interpolateFromTabular(RGB2SpectLambda, RGBRefl2SpectBlue);
    // gRGBIllum2SpectWhite =
    //     interpolateFromTabular(RGB2SpectLambda, RGBIllum2SpectWhite);
    // gRGBIllum2SpectCyan =
    //     interpolateFromTabular(RGB2SpectLambda, RGBIllum2SpectCyan);
    // gRGBIllum2SpectMagenta =
    //     interpolateFromTabular(RGB2SpectLambda, RGBIllum2SpectMagenta);
    // gRGBIllum2SpectYellow =
    //     interpolateFromTabular(RGB2SpectLambda, RGBIllum2SpectYellow);
    // gRGBIllum2SpectRed =
    //     interpolateFromTabular(RGB2SpectLambda, RGBIllum2SpectRed);
    // gRGBIllum2SpectGreen =
    //     interpolateFromTabular(RGB2SpectLambda, RGBIllum2SpectGreen);
    // gRGBIllum2SpectBlue =
    //     interpolateFromTabular(RGB2SpectLambda, RGBIllum2SpectBlue);
}

}  // namespace CRay
