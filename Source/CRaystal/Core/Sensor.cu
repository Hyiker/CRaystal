#include "Error.h"
#include "Sensor.h"
namespace CRay {

CRAYSTAL_DEVICE void atomicAdd(Spectrum& lhs, const Spectrum& rhs) {
    [[unroll]]
    for (int i = 0; i < lhs.size(); i++) {
        ::atomicAdd(&lhs[i], rhs[i]);
    }
}

CRAYSTAL_DEVICE Float evalTriangleFilter(Float filterRadius, Float2 xy) {
    return max(0.0f, filterRadius - abs(xy.x)) *
           max(0.0f, filterRadius - abs(xy.y));
}

CRAYSTAL_DEVICE void SensorData::addSample(const Spectrum& sample, Float2 xy) {
    // TODO: optimize me
    Int2 pMin = Int2(floor(xy - Float2(1.0f)));
    Int2 pMax = Int2(ceil(xy + Float2(1.0f)));

    pMin = max(pMin, Int2(0));
    pMax = min(pMax, Int2(size) - Int2(1));

    for (int y = pMin.y; y <= pMax.y; y++) {
        for (int x = pMin.x; x <= pMax.x; x++) {
            Float2 p = Float2(x, y) + Float2(0.5f);
            Float2 d = xy - p;

            Float w = evalTriangleFilter(1.0, d);

            uint32_t idx = getIndex(UInt2(x, y));
            atomicAdd(dataArray[idx], sample * w);
        }
    }
}

Sensor::Sensor(UInt2 size, uint32_t spp) {
    uint32_t area = size.x * size.y;

    mConstData.size = size;
    mConstData.weight = 1.f / Float(spp);

    mData.resize(area);

    mpDeviceConstData = std::make_unique<DeviceBuffer>(sizeof(SensorData));
    mpDeviceData = std::make_unique<DeviceBuffer>(area * sizeof(Spectrum));
    mConstData.dataArray = (Spectrum*)mpDeviceData->data();
}

Sensor::Sensor(Sensor&& other) noexcept
    : mConstData(std::move(other.mConstData)),
      mData(std::move(other.mData)),
      mpDeviceConstData(std::move(other.mpDeviceConstData)),
      mpDeviceData(std::move(other.mpDeviceData)) {}

Sensor& Sensor::operator=(Sensor&& other) noexcept {
    if (this != &other) {
        mConstData = std::move(other.mConstData);
        mData = std::move(other.mData);
        mpDeviceConstData = std::move(other.mpDeviceConstData);
        mpDeviceData = std::move(other.mpDeviceData);
    }
    return *this;
}

SensorData* Sensor::getDeviceView() const {
    return (SensorData*)mpDeviceConstData->data();
}

void Sensor::readbackDeviceData() {
    CRAYSTAL_ASSERT(mpDeviceData != nullptr);
    mpDeviceData->copyToHost(mData.data());
}

void Sensor::updateDeviceData() const {
    CRAYSTAL_ASSERT(mpDeviceConstData != nullptr);
    mpDeviceConstData->copyFromHost(&mConstData);
}

Image::Ref Sensor::createImage() const {
    UInt2 size = mConstData.size;
    Image::Ref pImage =
        std::make_shared<Image>(size.x, size.y, 3, ColorSpace::Linear);

    for (uint32_t y = 0; y < size.y; y++) {
        for (uint32_t x = 0; x < size.x; x++) {
            uint32_t index = mConstData.getIndex(UInt2(x, y));
            Float3 rgb = mData[index].toRGB();

            pImage->setPixel(x, y, 0, rgb.r);
            pImage->setPixel(x, y, 1, rgb.g);
            pImage->setPixel(x, y, 2, rgb.b);
        }
    }

    return pImage;
}

}  // namespace CRay
