#pragma once

#include <filesystem>
#include <memory>
#include <vector>

#include "Core/Enum.h"
#include "Core/Macros.h"

namespace CRay {

enum class ColorSpace {
    Linear,    ///< linear color space
    Gamma2_2,  ///< Gamma 2.2 color space, gamma corrected
};

CRAYSTAL_ENUM_INFO(ColorSpace, {{ColorSpace::Linear, "Linear"},
                                {ColorSpace::Gamma2_2, "Gamma2_2"}});
CRAYSTAL_ENUM_REGISTER(ColorSpace);

/** Image interface
 */
class CRAYSTAL_API Image {
   public:
    using Ref = std::shared_ptr<Image>;
    Image(size_t width, size_t height, int channels,
          ColorSpace colorSpace = ColorSpace::Linear)
        : mWidth(int(width)),
          mHeight(int(height)),
          mChannels(channels),
          mColorSpace(colorSpace),
          mData(channels, std::vector<float>(width * height, 0.f)) {}
    Image(const Image& other) = delete;
    Image(Image&& other) noexcept
        : mWidth(other.mWidth),
          mHeight(other.mHeight),
          mChannels(other.mChannels),
          mData(std::move(other.mData)) {}
    void resize(size_t width, size_t height);
    /** Resize the number of channels of the image, keep the original data and
     * fill the new channels with 0.
     */
    void resizeChannels(int channels) {
        this->mChannels = channels;
        mData.resize(channels,
                     std::vector<float>(size_t(mWidth * mHeight), 0.f));
    }

    // Image properties
    [[nodiscard]] auto getWidth() const { return mWidth; }
    [[nodiscard]] auto getHeight() const { return mHeight; }
    [[nodiscard]] auto getChannels() const { return mChannels; }
    [[nodiscard]] auto getArea() const { return mWidth * mHeight; }
    [[nodiscard]] const auto& getRawData() const { return mData; }

    // Pixel accessing
    [[nodiscard]] float getPixel(int x, int y, int channels) const {
        return mData[channels][x + y * mWidth];
    }
    [[nodiscard]] float getPixel(int index, int channels) const {
        return mData[channels][index];
    }
    float& getPixel(int x, int y, int channels) {
        return mData[channels][x + y * mWidth];
    }
    float& getPixel(int index, int channels) { return mData[channels][index]; }

    void setPixel(int x, int y, int channels, float value);
    void setPixel(int index, int channels, float value);

    void writeEXR(const std::filesystem::path& filename) const;
    void readEXR(const std::filesystem::path& filename);

    void writePNG(const std::filesystem::path& filename,
                  bool gammaCorrect) const;
    void readMisc(const std::filesystem::path& filename, bool toLinear);

    /** Construct an image object from a file.
     */
    static Image load(const std::filesystem::path& filename,
                      bool toLinear = true);

   private:
    int mWidth;
    int mHeight;
    int mChannels;

    ColorSpace mColorSpace = ColorSpace::Linear;  ///< color space of the image
    std::vector<std::vector<float>>
        mData;  // raw data layers splitted by channels, width x height x
                // channels, mData[channel][x + y * width]
};

}  // namespace CRay
