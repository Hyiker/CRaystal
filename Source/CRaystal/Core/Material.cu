#include "Material.h"

namespace CRay {

CRAYSTAL_DEVICE MaterialData
MaterialView::getMaterialData(uint32_t materialID) const {
    return materialData[materialID];
}

CRAYSTAL_DEVICE TextureView
MaterialView::getTexture(TextureID textureID) const {
    return textureData[textureID];
}

MaterialManager::MaterialManager(
    const std::vector<MaterialData>& materialData,
    const std::vector<Image>& textureImages,
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

    createTextures(textureImages);
}

void MaterialManager::createTextures(const std::vector<Image>& textureImages) {
    std::vector<TextureView> textureViews;

    for (const auto& image : textureImages) {
        mTextures.emplace_back(image);
        textureViews.push_back(mTextures.back().getDeviceView());
    }

    mpTextureDataBuffer =
        std::make_unique<DeviceBuffer>(textureViews.size() * sizeof(Texture));
    mpTextureDataBuffer->copyFromHost(textureViews.data());
    mView.textureData = mpTextureDataBuffer->getDeviceArray<TextureView>();
}

}  // namespace CRay
