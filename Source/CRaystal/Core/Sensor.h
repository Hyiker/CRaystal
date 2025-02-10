#pragma once
#include <memory>

#include "Buffer.h"
#include "Macros.h"
#include "Math/CRayMath.h"
#include "Math/Ray.h"
#include "Object.h"
#include "Spectrum.h"
#include "Utils/Image.h"
#include "Vec.h"

namespace CRay {

struct CRAYSTAL_API SensorData {
    Spectrum* dataArray;
    Float* weightArray;
    UInt2 size = UInt2(512u);  ///< Sensor film size.
    uint32_t spp = 1u;         ///< Samples per pixel.
    Float weight = 1.f;        ///< Sample base weight located at pixel + 0.5.

    CRAYSTAL_DEVICE_HOST uint32_t getIndex(UInt2 xy) const {
        return size.x * xy.y + xy.x;
    }

    /** Add sample to sensor with triangle filter.
     */
    CRAYSTAL_DEVICE void addSample(const Spectrum& sample, Float2 xy);
};

class CRAYSTAL_API Sensor : public HostObject {
   public:
    using Ref = std::shared_ptr<Sensor>;

    Sensor(UInt2 size, uint32_t spp = 1u);

    Sensor(const Sensor& other) = delete;
    Sensor(Sensor&& other) noexcept;
    Sensor& operator=(const Sensor& other) = delete;
    Sensor& operator=(Sensor&& other) noexcept;

    SensorData* getDeviceView() const;

    void readbackDeviceData() override;
    void updateDeviceData() const override;

    void setSPP(uint32_t value) {
        mConstData.spp = value;
        mConstData.weight = 1.0 / value;
    }

    uint32_t getSPP() const { return mConstData.spp; }

    UInt2 getSize() const { return mConstData.size; }

    /** Create rgb image from sensor data.
     */
    Image::Ref createImage() const;

    ~Sensor() = default;

   private:
    SensorData mConstData;
    std::vector<Spectrum> mData;
    std::vector<Float> mWeightData;

    std::unique_ptr<DeviceBuffer> mpDeviceConstData;
    std::unique_ptr<DeviceBuffer> mpDeviceData;
    std::unique_ptr<DeviceBuffer> mpDeviceWeight;
};

}  // namespace CRay
