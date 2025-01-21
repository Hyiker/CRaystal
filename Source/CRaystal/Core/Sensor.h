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
    UInt2 size = UInt2(512u);  ///< Sensor film size.
    Float weight = 1.f;        ///< Sample base weight located at pixel + 0.5.

    CRAYSTAL_DEVICE_HOST uint32_t getIndex(UInt2 xy) const {
        return size.x * xy.y + xy.x;
    }

    CRAYSTAL_DEVICE_HOST void addSample(const Spectrum& sample, Float2 xy) {
        dataArray[getIndex(UInt2(xy))] += sample * weight;
    }
};

class CRAYSTAL_API Sensor : public HostObject {
   public:
    using Ref = std::shared_ptr<Sensor>;

    Sensor(UInt2 size, uint32_t spp);

    Sensor(const Sensor& other) = delete;
    Sensor(Sensor&& other) noexcept;
    Sensor& operator=(const Sensor& other) = delete;
    Sensor& operator=(Sensor&& other) noexcept;

    SensorData* getDeviceView() const;

    void readbackDeviceData() override;
    void updateDeviceData() const override;

    uint32_t getSPP() const { return mSPP; }

    /** Create rgb image from sensor data.
     */
    Image::Ref createImage() const;

    ~Sensor() = default;

   private:
    uint32_t mSPP;

    SensorData mConstData;
    std::vector<Spectrum> mData;

    std::unique_ptr<DeviceBuffer> mpDeviceConstData;
    std::unique_ptr<DeviceBuffer> mpDeviceData;
};

}  // namespace CRay
