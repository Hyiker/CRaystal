#pragma once

#include "Buffer.h"
#include "Enum.h"
#include "Macros.h"
#include "Spectrum.h"

namespace CRay {

enum class MaterialType { Principled };

CRAYSTAL_ENUM_INFO(MaterialType, {
                                     {MaterialType::Principled, "Principled"},
                                 })
CRAYSTAL_ENUM_REGISTER(MaterialType)

struct CRAYSTAL_API MaterialData {
    Spectrum diffuseRefl;    ///< Kd, the diffuse reflectance of material.
    Spectrum specularRefl;   ///< Ks, the specular reflectance of material.
    Spectrum transmittance;  ///< Tr, the transmittance of material.
    Float shininess;         ///< Tr, shininess, the exponent of phong lobe.
    Float IoR;               ///< Ni, Index of Refraction.
    MaterialType type;       ///< Material type, decide which bsdf to use.
};

class CRAYSTAL_API MaterialManager {
   public:
    using Ref = std::shared_ptr<MaterialManager>;
    struct DeviceView {
        MaterialData* pData;
        uint32_t materialCount;

        CRAYSTAL_DEVICE MaterialData getMaterialData(uint32_t materialID) const;
    };

    MaterialManager(const std::vector<MaterialData>& materialData);

    DeviceView getDeviceView() const { return mView; }

   private:
    DeviceView mView;

    std::unique_ptr<DeviceBuffer> mpMaterialDataBuffer;
};

using MaterialView = MaterialManager::DeviceView;

}  // namespace CRay
