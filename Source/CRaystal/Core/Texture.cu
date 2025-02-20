#include "Math/CRayMath.h"
#include "Texture.h"

namespace CRay {

CRAYSTAL_DEVICE Float4 TextureView::fetchTexel(const UInt2& xy) const {
    uint32_t x = std::clamp<uint32_t>(xy.x, 0, desc.width - 1);
    uint32_t y = std::clamp<uint32_t>(xy.y, 0, desc.height - 1);

    uint32_t index = y * desc.width + x;

    return data[index];
}

CRAYSTAL_DEVICE Float4 TextureView::sample(const Float2& uv) const {
    Float px = uv.x * (desc.width - 1);
    Float py = uv.y * (desc.height - 1);

    uint32_t x0 = static_cast<uint32_t>(floor(px));
    uint32_t y0 = static_cast<uint32_t>(floor(py));
    uint32_t x1 = min(x0 + 1, desc.width - 1);
    uint32_t y1 = min(y0 + 1, desc.height - 1);

    Float fx = px - x0;
    Float fy = py - y0;

    Float4 c00 = fetchTexel(UInt2(x0, y0));
    Float4 c10 = fetchTexel(UInt2(x1, y0));
    Float4 c01 = fetchTexel(UInt2(x0, y1));
    Float4 c11 = fetchTexel(UInt2(x1, y1));

    Float4 c0 = lerp(c00, c10, fx);
    Float4 c1 = lerp(c01, c11, fx);
    return lerp(c0, c1, fy);
}

Texture::Texture(Float value) {
    mDeviceView.desc.width = 1u;
    mDeviceView.desc.height = 1u;

    std::vector<Float4> data(1, Float4(value));
    createDeviceData(data.data(), sizeof(Float4) * 1);
}

Texture::Texture(const Image& image) {
    mDeviceView.desc.width = image.getWidth();
    mDeviceView.desc.height = image.getHeight();

    std::vector<Float4> data(image.getArea(), Float4(0.0));
    for (int y = 0; y < image.getHeight(); y++) {
        for (int x = 0; x < image.getWidth(); x++) {
            uint32_t i = y * image.getWidth() + x;
            Float4 color(0.0);
            for (int c = 0; c < image.getChannels(); c++) {
                color[c] = image.getPixel(x, y, c);
            }
            data[i] = color;
        }
    }
    createDeviceData(data.data(), sizeof(Float4) * data.size());
}

void Texture::createDeviceData(const void* pData, uint32_t size) {
    mpDeviceTextureData = std::make_shared<DeviceBuffer>(size);
    mpDeviceTextureData->copyFromHost(pData);

    mDeviceView.data = mpDeviceTextureData->getDeviceArray<Float4>();
}

}  // namespace CRay
