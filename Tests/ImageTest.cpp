#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "Utils/Image.h"

using namespace CRay;
using namespace Catch::Matchers;

TEST_CASE("Image Construction", "[Image]") {
    Logger::init(Logger::Level::Disabled);

    SECTION("Default Construction") {
        Image img(100, 200, 3);  // 100x200 image with 3 channels
        REQUIRE(img.getWidth() == 100);
        REQUIRE(img.getHeight() == 200);
        REQUIRE(img.getChannels() == 3);
        REQUIRE(img.getArea() == 100 * 200);

        // Verify that all pixels are initialized to 0
        for (int c = 0; c < img.getChannels(); ++c) {
            for (int i = 0; i < img.getArea(); ++i) {
                REQUIRE_THAT(img.getPixel(i, c), WithinAbs(0.0f, 1e-5f));
            }
        }
    }
}

TEST_CASE("Image Resizing", "[Image]") {
    Image img(100, 200, 3);

    SECTION("Resize Width and Height") {
        img.resize(150, 250);
        REQUIRE(img.getWidth() == 150);
        REQUIRE(img.getHeight() == 250);
        REQUIRE(img.getArea() == 150 * 250);

        // Verify that the new pixels are initialized to 0
        for (int c = 0; c < img.getChannels(); ++c) {
            for (int i = 0; i < img.getArea(); ++i) {
                REQUIRE_THAT(img.getPixel(i, c), WithinAbs(0.0f, 1e-5f));
            }
        }
    }

    SECTION("Resize Channels") {
        img.resizeChannels(4);
        REQUIRE(img.getChannels() == 4);

        // Verify that the new channel is initialized to 0
        for (int i = 0; i < img.getArea(); ++i) {
            REQUIRE_THAT(img.getPixel(i, 3), WithinAbs(0.0f, 1e-5f));
        }
    }
}

TEST_CASE("Pixel Access", "[Image]") {
    Image img(100, 200, 3);

    SECTION("Set and Get Pixel by Index") {
        img.setPixel(50, 1, 0.5f);  // Set pixel at index 50 in channel 1
        REQUIRE_THAT(img.getPixel(50, 1), WithinAbs(0.5f, 1e-5f));
    }

    SECTION("Set and Get Pixel by Coordinates") {
        img.setPixel(10, 20, 2, 0.75f);  // Set pixel at (10, 20) in channel 2
        REQUIRE_THAT(img.getPixel(10, 20, 2), WithinAbs(0.75f, 1e-5f));
    }
}

TEST_CASE("Image File I/O", "[Image]") {
    Image img(100, 200, 3);

    SECTION("Write and Read PNG") {
        // Set some pixel values
        img.setPixel(10, 20, 0, 0.25f);
        img.setPixel(10, 20, 1, 0.5f);
        img.setPixel(10, 20, 2, 0.75f);

        // Write to PNG
        img.writePNG("test.png", false);

        // Read from PNG
        Image img2 = Image::load("test.png");

        // Verify dimensions and channels
        REQUIRE(img2.getWidth() == img.getWidth());
        REQUIRE(img2.getHeight() == img.getHeight());
        REQUIRE(img2.getChannels() == img.getChannels());

        // Verify pixel values
        REQUIRE_THAT(img2.getPixel(10, 20, 0), WithinAbs(0.25f, 1e-2f));
        REQUIRE_THAT(img2.getPixel(10, 20, 1), WithinAbs(0.5f, 1e-2f));
        REQUIRE_THAT(img2.getPixel(10, 20, 2), WithinAbs(0.75f, 1e-2f));
    }

    SECTION("Write and Read EXR") {
        // Set some pixel values
        img.setPixel(10, 20, 0, 0.25f);
        img.setPixel(10, 20, 1, 0.5f);
        img.setPixel(10, 20, 2, 0.75f);

        // Write to EXR
        img.writeEXR("test.exr");

        // Read from EXR
        Image img2 = Image::load("test.exr");

        // Verify dimensions and channels
        REQUIRE(img2.getWidth() == img.getWidth());
        REQUIRE(img2.getHeight() == img.getHeight());
        REQUIRE(img2.getChannels() == img.getChannels());

        // Verify pixel values
        REQUIRE_THAT(img2.getPixel(10, 20, 0), WithinAbs(0.25f, 1e-5f));
        REQUIRE_THAT(img2.getPixel(10, 20, 1), WithinAbs(0.5f, 1e-5f));
        REQUIRE_THAT(img2.getPixel(10, 20, 2), WithinAbs(0.75f, 1e-5f));
    }
}

TEST_CASE("Image Load from File", "[Image]") {
    SECTION("Load PNG") {
        Image img = Image::load("test.png");

        // Verify dimensions and channels
        REQUIRE(img.getWidth() == 100);
        REQUIRE(img.getHeight() == 200);
        REQUIRE(img.getChannels() == 3);

        // Verify pixel values
        REQUIRE_THAT(img.getPixel(10, 20, 0), WithinAbs(0.25f, 1e-2f));
        REQUIRE_THAT(img.getPixel(10, 20, 1), WithinAbs(0.5f, 1e-2f));
        REQUIRE_THAT(img.getPixel(10, 20, 2), WithinAbs(0.75f, 1e-2f));
    }

    SECTION("Load EXR") {
        Image img = Image::load("test.exr");

        // Verify dimensions and channels
        REQUIRE(img.getWidth() == 100);
        REQUIRE(img.getHeight() == 200);
        REQUIRE(img.getChannels() == 3);

        // Verify pixel values
        REQUIRE_THAT(img.getPixel(10, 20, 0), WithinAbs(0.25f, 1e-5f));
        REQUIRE_THAT(img.getPixel(10, 20, 1), WithinAbs(0.5f, 1e-5f));
        REQUIRE_THAT(img.getPixel(10, 20, 2), WithinAbs(0.75f, 1e-5f));
    }
}
