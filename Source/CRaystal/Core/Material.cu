#include "Material.h"

namespace CRay {

CRAYSTAL_DEVICE MaterialData
MaterialView::getMaterialData(uint32_t materialID) const {
    return pData[materialID];
}

MaterialManager::MaterialManager(
    const std::vector<MaterialData>& materialData) {
    mpMaterialDataBuffer = std::make_unique<DeviceBuffer>(materialData.size() *
                                                          sizeof(MaterialData));
    mpMaterialDataBuffer->copyFromHost(materialData.data());

    mView.materialCount = materialData.size();
    mView.pData = (MaterialData*)mpMaterialDataBuffer->data();
}

}  // namespace CRay
