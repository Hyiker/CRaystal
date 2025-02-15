#include "Material.h"

namespace CRay {

CRAYSTAL_DEVICE_HOST BSDF getBSDF(const MaterialData& material,
                                  const Frame& frame) {
    BSDFVariant bsdfVar;

    switch (material.type) {
        case MaterialType::Diffuse: {
            bsdfVar.emplace<LambertianBRDF>(material.diffuseRefl);
        } break;
        case MaterialType::Principled: {
            bsdfVar.emplace<PrincipledBRDF>(material.diffuseRefl,
                                            material.specularRefl,
                                            material.shininess);
        } break;
    }

    return BSDF(bsdfVar, frame);
}

CRAYSTAL_DEVICE MaterialData
MaterialView::getMaterialData(uint32_t materialID) const {
    return materialData[materialID];
}

MaterialManager::MaterialManager(
    const std::vector<MaterialData>& materialData,
    const std::vector<uint32_t>& emissiveTriangleIndex) {
    mpMaterialDataBuffer = std::make_unique<DeviceBuffer>(materialData.size() *
                                                          sizeof(MaterialData));
    mpMaterialDataBuffer->copyFromHost(materialData.data());
    mView.materialData = mpMaterialDataBuffer->getDeviceArray<MaterialData>();

    mpEmissiveIndexBuffer = std::make_unique<DeviceBuffer>(
        emissiveTriangleIndex.size() * sizeof(uint32_t));
    mpEmissiveIndexBuffer->copyFromHost(emissiveTriangleIndex.data());
    mView.emissiveTriangleIndex =
        mpEmissiveIndexBuffer->getDeviceArray<uint32_t>();
}

}  // namespace CRay
