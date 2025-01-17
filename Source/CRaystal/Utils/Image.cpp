#include "Image.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4706)  // ignore external library header warning
#endif

#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <algorithm>
#include <array>
#include <filesystem>
#include <span>

#include "Core/Error.h"
#include "Core/Vec.h"
#include "Utils/Logger.h"

namespace CRay {

void Image::resize(size_t width, size_t height) {
    this->mWidth = int(width);
    this->mHeight = int(height);
    int area = int(width * height);
    std::for_each(mData.begin(), mData.end(),
                  [area](auto& layer) { layer.resize(area, 0.f); });
}

void Image::setPixel(int x, int y, int channels, float value) {
    setPixel(x + y * mWidth, channels, value);
}

void Image::setPixel(int index, int channels, float value) {
    CRAYSTAL_ASSERT(index < mWidth * mHeight);
    CRAYSTAL_ASSERT(channels < this->mChannels);
    mData[channels][index] = value;
}

void Image::writeEXR(const std::filesystem::path& filename) const {
    CRAYSTAL_CHECK(filename.extension() == ".exr", "Output file must be .exr");
    EXRHeader header;
    EXRImage image;
    InitEXRHeader(&header);
    InitEXRImage(&image);

    // Split RGBRGBRGB... into R, G and B layer
    CRAYSTAL_ASSERT(mChannels >= 3);
    std::array<const float*, 3> layerPointer;
    layerPointer[0] = mData[2].data();  // B
    layerPointer[1] = mData[1].data();  // G
    layerPointer[2] = mData[0].data();  // R

    image.images = reinterpret_cast<unsigned char**>(
        const_cast<float**>(layerPointer.data()));
    image.width = mWidth;
    image.height = mHeight;

    header.num_channels = 3;
    header.channels = static_cast<EXRChannelInfo*>(
        malloc(sizeof(EXRChannelInfo) * header.num_channels));
    // Must be (A)BGR order, since most of EXR viewers expect this channel
    // order.
    strncpy(header.channels[0].name, "B", 255);
    header.channels[0].name[strlen("B")] = '\0';
    strncpy(header.channels[1].name, "G", 255);
    header.channels[1].name[strlen("G")] = '\0';
    strncpy(header.channels[2].name, "R", 255);
    header.channels[2].name[strlen("R")] = '\0';

    header.pixel_types =
        static_cast<int*>(malloc(sizeof(int) * header.num_channels));
    header.requested_pixel_types =
        static_cast<int*>(malloc(sizeof(int) * header.num_channels));
    for (int i = 0; i < header.num_channels; i++) {
        header.pixel_types[i] =
            TINYEXR_PIXELTYPE_FLOAT;  // pixel type of input image
        header.requested_pixel_types[i] =
            TINYEXR_PIXELTYPE_HALF;  // pixel type of output image to be stored
                                     // in .EXR
    }

    const char* err = nullptr;  // or nullptr in C++11 or later.
    int ret =
        SaveEXRImageToFile(&image, &header, filename.string().c_str(), &err);

    free(header.channels);
    free(header.pixel_types);
    free(header.requested_pixel_types);

    if (ret != TINYEXR_SUCCESS) {
        logError("Saving EXR error: {}", err);

        FreeEXRErrorMessage(err);
        return;
    }
}

void Image::readEXR(const std::filesystem::path& filename) {
    CRAYSTAL_CHECK(filename.extension() == ".exr", "Output file must be .exr");
    logDebug("Loading EXR file from {}", filename.string());
    EXRVersion exrVersion;
    EXRImage exrImage;
    EXRHeader exrHeader;

    InitEXRHeader(&exrHeader);

    int ret = ParseEXRVersionFromFile(&exrVersion, filename.string().c_str());
    if (ret != TINYEXR_SUCCESS) {
        logError("EXR version parse error");
        return;
    }
    const char* err = nullptr;

    ret = ParseEXRHeaderFromFile(&exrHeader, &exrVersion,
                                 filename.string().c_str(), &err);
    if (ret != TINYEXR_SUCCESS) {
        if (err != nullptr) {
            logError("EXR header parse error: {}", err);
            FreeEXRErrorMessage(err);
        }
        return;
    }

    for (int i = 0; i < exrHeader.num_channels; i++) {
        exrHeader.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
    }

    InitEXRImage(&exrImage);
    ret = LoadEXRImageFromFile(&exrImage, &exrHeader, filename.string().c_str(),
                               &err);
    if (ret != TINYEXR_SUCCESS) {
        if (err != nullptr) {
            logError("EXR image load error: {}", err);
            FreeEXRErrorMessage(err);
        }
        return;
    }

    // Assuming the image is RGB(A)
    CRAYSTAL_ASSERT(exrHeader.num_channels >= 3);

    // mapping exr channels to rgb
    std::array<int, 3> channelMapping;
    for (int i = 0; i < 3; i++) {
        channelMapping[i] = -1;
        for (int j = 0; j < exrHeader.num_channels; j++) {
            if (exrHeader.channels[j].name[0] == "RGB"[i]) {
                channelMapping[i] = j;
                break;
            }
        }
        CRAYSTAL_ASSERT(channelMapping[i] != -1);
    }

    resize(exrImage.width, exrImage.height);
    resizeChannels(exrHeader.num_channels);
    int area = getArea();

    for (int c = 0; c < 3; c++) {
        int exrChannel = channelMapping[c];
        std::span<const float> layer(
            reinterpret_cast<float*>(exrImage.images[exrChannel]), area);
        std::copy(layer.begin(), layer.end(), mData[c].begin());
    }

    FreeEXRHeader(&exrHeader);
    FreeEXRImage(&exrImage);

    logDebug("Loaded EXR file from {}", filename.string());
}

void Image::writePNG(const std::filesystem::path& filename,
                     bool gammaCorrect) const {
    CRAYSTAL_CHECK(filename.extension() == ".png", "Output file must be .png");

    auto hdrToSdr = [&](float value) {
        // Do color space conversion first
        if (gammaCorrect) {
            switch (mColorSpace) {
                case ColorSpace::Linear:
                    value = std::pow(value, 1 / 2.2f);
                    break;
                case ColorSpace::Gamma2_2:
                    break;
            }
            return saturate(value);
        } else {
            return value;
        }
    };

    // From RRR... GGG... BBB... to RGBRGBRGB...
    std::vector<unsigned char> image((size_t)mWidth * mHeight * mChannels);
    for (int c = 0; c < mChannels; c++) {
        for (int i = 0; i < mWidth * mHeight; i++) {
            image[i * mChannels + c] =
                (unsigned char)(hdrToSdr(mData[c][i]) * 255.f);
        }
    }

    // Use stb_image_write to save the PNG file
    int stride = mWidth * mChannels;  // Bytes per row
    if (!stbi_write_png(filename.string().c_str(), mWidth, mHeight, mChannels,
                        image.data(), stride)) {
        logFatal("Failed to write PNG file: {}", filename.string());
    }
}

void Image::readPNG(const std::filesystem::path& filename) {
    CRAYSTAL_CHECK(filename.extension() == ".png", "Output file must be .png");

    int width, height, channels;
    unsigned char* imageData =
        stbi_load(filename.string().c_str(), &width, &height, &channels, 0);

    if (!imageData) {
        logFatal("PNG load error: {}", stbi_failure_reason());
        return;
    }

    switch (channels) {
        case 1:
            resizeChannels(1);
            break;
        case 3:
            resizeChannels(3);
            break;
        case 4:
            resizeChannels(4);
            break;
        default:
            logFatal("Unsupported color type: {}", channels);
            stbi_image_free(imageData);
            return;
    }

    resize(width, height);
    int area = getArea();
    for (int c = 0; c < mChannels; c++) {
        auto& layer = mData[c];
        for (int i = 0; i < area; i++) {
            layer[i] = float(imageData[i * mChannels + c]) / 255.f;
        }
    }

    stbi_image_free(imageData);  // Free the loaded image data
}

Image Image::load(const std::filesystem::path& filename) {
    Image image(0, 0, 3);
    if (filename.extension() == ".png") {
        image.readPNG(filename);
    } else if (filename.extension() == ".exr") {
        image.readEXR(filename);
    } else {
        logFatal("Unsupported image format: {}", filename.extension().string());
    }
    return image;
}

}  // namespace CRay
