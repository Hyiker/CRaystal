#include "Error.h"
#include "Sensor.h"
namespace CRay {
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
