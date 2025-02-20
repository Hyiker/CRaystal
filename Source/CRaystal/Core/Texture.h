#pragma once
#include <memory>
#include <span>

#include "Buffer.h"
#include "Core/DeviceArray.h"
#include "Core/Macros.h"
#include "Object.h"
#include "Utils/Image.h"
#include "Vec.h"
namespace CRay {
struct TextureDesc {
    uint32_t width = 1u;
    uint32_t height = 1u;
};
/** The rgba texture.
 */
class CRAYSTAL_API Texture {
   public:
    using Ref = std::shared_ptr<Texture>;

    struct DeviceView {
        DeviceArray<Float4> data;  ///< The texture data, row first.
        TextureDesc desc;          ///< Texture descs.

        CRAYSTAL_DEVICE Float4 fetchTexel(const UInt2& xy) const;

        CRAYSTAL_DEVICE Float4 sample(const Float2& uv) const;
    };

    /** Create a 1x1 constant value texture.
     */
    Texture(Float value);

    /** Create texture from image.
     */
    Texture(const Image& image);

    DeviceView getDeviceView() const { return mDeviceView; }

    ~Texture() = default;

   private:
    void createDeviceData(const void* pData, uint32_t size);

    DeviceView mDeviceView;

    DeviceBuffer::Ref mpDeviceTextureData;
};

using TextureView = Texture::DeviceView;

}  // namespace CRay
