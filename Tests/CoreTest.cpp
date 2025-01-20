#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "Core/Buffer.h"
#include "Core/Spectrum.h"

using namespace CRay;
using namespace Catch::Matchers;

TEST_CASE("Test RGBSpectrum operations") {
    Logger::init(Logger::Level::Disabled);

    SECTION("Test Constructor") {
        RGBSpectrum spectrum(1.f);
        for (int i = 0; i < spectrum.size(); ++i) {
            REQUIRE_THAT(spectrum[i], WithinAbs(1.0f, 1e-5));
        }
    }

    SECTION("Test arithmetic") {
        RGBSpectrum spec1(1.0f), spec2(2.f);
        auto result = spec1 + spec2;
        for (int i = 0; i < result.size(); ++i) {
            REQUIRE_THAT(result[i], WithinAbs(3.0f, 1e-5));
        }
    }

    SECTION("Test subtraction") {
        RGBSpectrum spec1(3.0f), spec2(1.0f);
        auto result = spec1 - spec2;
        for (int i = 0; i < result.size(); ++i) {
            REQUIRE_THAT(result[i], WithinAbs(2.0f, 1e-5));
        }
    }

    SECTION("Test multiplication with scalar") {
        RGBSpectrum spec(2.0f);
        auto result = spec * 3.0f;
        for (int i = 0; i < result.size(); ++i) {
            REQUIRE_THAT(result[i], WithinAbs(6.0f, 1e-5));
        }
    }

    SECTION("Test division with scalar") {
        RGBSpectrum spec(6.0f);
        auto result = spec / 3.0f;
        for (int i = 0; i < result.size(); ++i) {
            REQUIRE_THAT(result[i], WithinAbs(2.0f, 1e-5));
        }
    }

    SECTION("Test exp") {
        RGBSpectrum spec(1.0f);
        auto result = spec.exp();
        for (int i = 0; i < result.size(); ++i) {
            REQUIRE_THAT(result[i], WithinAbs(std::exp(1.0f), 1e-5));
        }
    }

    SECTION("Test pow") {
        RGBSpectrum spec(2.0f);
        auto result = spec.pow(3.0f);
        for (int i = 0; i < result.size(); ++i) {
            REQUIRE_THAT(result[i], WithinAbs(std::pow(2.0f, 3.0f), 1e-5));
        }
    }

    SECTION("Test sqrt") {
        RGBSpectrum spec(4.0f);
        auto result = spec.sqrt();
        for (int i = 0; i < result.size(); ++i) {
            REQUIRE_THAT(result[i], WithinAbs(std::sqrt(4.0f), 1e-5));
        }
    }
}

SCENARIO("DeviceBuffer basic operations") {
    GIVEN("A DeviceBuffer with size") {
        const int bufferSize = 1024;
        DeviceBuffer buffer(bufferSize);

        THEN("Buffer properties are correct") {
            CHECK(buffer.size() == bufferSize);
            CHECK(buffer.data() != nullptr);
        }

        WHEN("Copying data to device") {
            std::vector<float> hostData(bufferSize / sizeof(float), 1.0f);
            buffer.copyFromHost(hostData.data());

            THEN("Data can be copied back") {
                std::vector<float> resultData(bufferSize / sizeof(float), 0.0f);
                buffer.copyToHost(resultData.data());
                CHECK(resultData == hostData);
            }
        }

        WHEN("Using memset") {
            buffer.memset(0);
            std::vector<unsigned char> resultData(bufferSize, 255);
            buffer.copyToHost(resultData.data());

            THEN("Memory is set correctly") {
                CHECK(std::all_of(resultData.begin(), resultData.end(),
                                [](unsigned char v) { return v == 0; }));
            }
        }
    }

    GIVEN("A moved buffer") {
        DeviceBuffer original(1024);
        DeviceBuffer moved = std::move(original);

        THEN("Original is empty") {
            CHECK(original.size() == 0);
            CHECK(original.data() == nullptr);
        }

        AND_THEN("Moved buffer has the data") {
            CHECK(moved.size() == 1024);
            CHECK(moved.data() != nullptr);
        }
    }
}
