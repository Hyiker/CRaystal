#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "Core/Camera.h"
#include "Core/Sensor.h"

using namespace CRay;
using namespace Catch::Matchers;

TEST_CASE("CameraProxy - generateRay Function", "[CameraProxy]") {
    Camera camera;
    UInt2 sensorSize(512, 512);

    camera.calculateCameraData();

    SECTION("Generate Ray at Center of Sensor") {
        const auto& data = camera.getData();
        Float2 sensorPos(256, 256);
        Ray ray = data.generateRay(sensorSize, sensorPos);
        REQUIRE(ray.origin == Float3p32(0, 0, 0));
        REQUIRE_THAT(ray.direction.x, Catch::Matchers::WithinAbs(0.0f, 1e-5));
        REQUIRE_THAT(ray.direction.y, Catch::Matchers::WithinAbs(0.0f, 1e-5));
        REQUIRE_THAT(ray.direction.z, Catch::Matchers::WithinAbs(-1.0f, 1e-5));
    }

    SECTION("Generate Ray at Top-Left Corner of Sensor") {
        const auto& data = camera.getData();
        Float2 sensorPos(0, 0);
        Ray ray = data.generateRay(sensorSize, sensorPos);
        REQUIRE(ray.origin == Float3p32(0, 0, 0));
        REQUIRE_THAT(ray.direction.x,
                     Catch::Matchers::WithinAbs(-0.44721359f, 1e-4));
        REQUIRE_THAT(ray.direction.y,
                     Catch::Matchers::WithinAbs(0.44721359f, 1e-4));
        REQUIRE_THAT(ray.direction.z,
                     Catch::Matchers::WithinAbs(-0.774596691f, 1e-4));
    }

    SECTION("Generate Ray at Bottom-Right Corner of Sensor") {
        const auto& data = camera.getData();
        Float2 sensorPos(511, 511);
        Ray ray = data.generateRay(sensorSize, sensorPos);
        REQUIRE(ray.origin == Float3p32(0, 0, 0));
        REQUIRE_THAT(ray.direction.x,
                     Catch::Matchers::WithinAbs(0.446162969f, 1e-4));
        REQUIRE_THAT(ray.direction.y,
                     Catch::Matchers::WithinAbs(-0.446162969f, 1e-4));
        REQUIRE_THAT(ray.direction.z,
                     Catch::Matchers::WithinAbs(-0.77580744f, 1e-4));
    }
}

SCENARIO("Sensor basic operations", "[sensor]") {
    GIVEN("A sensor with specified size") {
        UInt2 size{64, 48};
        uint32_t spp = 4;
        Sensor sensor(size, spp);

        THEN("Create image should return valid image") {
            auto image = sensor.createImage();
            REQUIRE(image != nullptr);
            CHECK(image->getWidth() == size.x);
            CHECK(image->getHeight() == size.y);
        }

        WHEN("Moving sensor") {
            Sensor movedSensor = std::move(sensor);

            THEN("Moved sensor should create valid image") {
                auto image = movedSensor.createImage();
                REQUIRE(image != nullptr);
                CHECK(image->getWidth() == size.x);
                CHECK(image->getHeight() == size.y);
            }
        }
    }
}

SCENARIO("Sensor device data operations", "[sensor]") {
    GIVEN("A sensor with test data") {
        UInt2 size{2, 2};
        Sensor sensor(size, 1);

        WHEN("Updating device data") {
            sensor.updateDeviceData();

            THEN("Reading back should not throw") {
                REQUIRE_NOTHROW(sensor.readbackDeviceData());
            }
        }
    }
}

SCENARIO("Sensor edge cases", "[sensor]") {
    SECTION("Single pixel sensor") {
        UInt2 size{1, 1};
        Sensor sensor(size, 1);
        REQUIRE_NOTHROW(sensor.createImage());
    }

    SECTION("Large sensor") {
        UInt2 size{4096, 4096};
        REQUIRE_NOTHROW(Sensor(size, 1));
    }
}

SCENARIO("Sensor image creation", "[sensor]") {
    GIVEN("A sensor with known data") {
        UInt2 size{2, 2};
        Sensor sensor(size, 1);

        WHEN("Creating image") {
            auto image = sensor.createImage();

            THEN("Image properties should match sensor") {
                REQUIRE(image != nullptr);
                CHECK(image->getWidth() == size.x);
                CHECK(image->getHeight() == size.y);
            }
        }
    }
}
