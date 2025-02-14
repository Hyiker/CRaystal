#pragma once

#include "Buffer.h"
#include "DeviceArray.h"
#include "Enum.h"
#include "Macros.h"
#include "Material/BSDF.h"
#include "Spectrum.h"

namespace CRay {

enum class MaterialType {
    Diffuse,    ///< The diffusive material.
    Principled  ///< Disney principled material.
};

CRAYSTAL_ENUM_INFO(MaterialType, {
                                     {MaterialType::Diffuse, "Diffuse"},
                                     {MaterialType::Principled, "Principled"},
                                 })

CRAYSTAL_ENUM_REGISTER(MaterialType)

enum class MaterialFlags { IsEmissive = 0x1, IsDiffuse = 0x2 };

struct CRAYSTAL_API MaterialData {
    Spectrum diffuseRefl;    ///< Kd, the diffuse reflectance of material.
    Spectrum specularRefl;   ///< Ks, the specular reflectance of material.
    Spectrum transmittance;  ///< Tr, the transmittance of material.
    Spectrum emission;       ///< Emission factor.
    Float shininess;         ///< Tr, shininess, the exponent of phong lobe.
    Float IoR;               ///< Ni, Index of Refraction.
    MaterialType type;       ///< Material type, decide which bsdf to use.
    uint32_t flags = 0u;     ///< Material flags.

    CRAYSTAL_DEVICE_HOST bool isEmissive() const {
        return flags & uint32_t(MaterialFlags::IsEmissive);
    }

    CRAYSTAL_DEVICE_HOST bool isDiffuse() const {
        return flags & uint32_t(MaterialFlags::IsDiffuse);
    }

    CRAYSTAL_HOST void finalize() {
        if (emission.maxValue() != 0.0) {
            flags |= uint32_t(MaterialFlags::IsEmissive);
        }

        if (specularRefl.maxValue() == 0.0) {
            flags |= uint32_t(MaterialFlags::IsDiffuse);
        }
    }
};

CRAYSTAL_API CRAYSTAL_DEVICE_HOST BSDF getBSDF(const MaterialData& material,
                                               const Frame& frame);

class CRAYSTAL_API MaterialManager {
   public:
    using Ref = std::shared_ptr<MaterialManager>;
    struct DeviceView {
        DeviceArray<MaterialData> materialData;
        DeviceArray<uint32_t> emissiveTriangleIndex;

        CRAYSTAL_DEVICE MaterialData getMaterialData(uint32_t materialID) const;

        CRAYSTAL_DEVICE int getEmissiveCount() const {
            return emissiveTriangleIndex.size();
        }
    };

    MaterialManager(const std::vector<MaterialData>& materialData,
                    const std::vector<uint32_t>& emissiveTriangleIndex);

    DeviceView getDeviceView() const { return mView; }

   private:
    DeviceView mView;

    std::unique_ptr<DeviceBuffer> mpMaterialDataBuffer;
    std::unique_ptr<DeviceBuffer> mpEmissiveIndexBuffer;
};

using MaterialView = MaterialManager::DeviceView;

}  // namespace CRay
