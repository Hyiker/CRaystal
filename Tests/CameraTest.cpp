#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "Core/Camera.h"

using namespace CRay;
using namespace Catch::Matchers;


TEST_CASE("CameraProxy - generateRay Function", "[CameraProxy]") {
    Camera camera;

    camera.calculateCameraData();

    SECTION("Generate Ray at Center of Sensor") {
        const auto& data = camera.getData();
        Float2 sensorPos(256, 256);
        Ray ray = data.generateRay(sensorPos);
        REQUIRE(ray.origin == Float3p32(0, 0, 0));
        REQUIRE_THAT(ray.direction.x, Catch::Matchers::WithinAbs(0.0f, 1e-5));
        REQUIRE_THAT(ray.direction.y, Catch::Matchers::WithinAbs(0.0f, 1e-5));
        REQUIRE_THAT(ray.direction.z, Catch::Matchers::WithinAbs(-1.0f, 1e-5));
    }

    SECTION("Generate Ray at Top-Left Corner of Sensor") {
        const auto& data = camera.getData();
        Float2 sensorPos(0, 0);
        Ray ray = data.generateRay(sensorPos);
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
        Ray ray = data.generateRay(sensorPos);
        REQUIRE(ray.origin == Float3p32(0, 0, 0));
        REQUIRE_THAT(ray.direction.x,
                     Catch::Matchers::WithinAbs(0.446162969f, 1e-4));
        REQUIRE_THAT(ray.direction.y,
                     Catch::Matchers::WithinAbs(-0.446162969f, 1e-4));
        REQUIRE_THAT(ray.direction.z,
                     Catch::Matchers::WithinAbs(-0.77580744f, 1e-4));
    }
}
