#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <numeric>
#include <vector>

#include "Core/Buffer.h"
#include "Core/Sampler.h"
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

TEST_CASE("Sampler basic range tests", "[sampler]") {
    UInt2 pixel{123, 456};
    Sampler sampler(pixel, 0);

    SECTION("1D samples should be in [0,1) range") {
        for (int i = 0; i < 1000; ++i) {
            float sample = sampler.nextSample1D();
            REQUIRE(sample >= 0.0f);
            REQUIRE(sample < 1.0f);
        }
    }

    SECTION("2D samples should be in [0,1) x [0,1) range") {
        for (int i = 0; i < 1000; ++i) {
            Float2 xy = sampler.nextSample2D();
            Float x = xy.x;
            Float y = xy.y;
            REQUIRE(x >= 0.0f);
            REQUIRE(x < 1.0f);
            REQUIRE(y >= 0.0f);
            REQUIRE(y < 1.0f);
        }
    }
}

TEST_CASE("Sampler different pixel seeds", "[sampler]") {
    UInt2 pixel1{0, 0};
    UInt2 pixel2{0, 1};
    Sampler sampler1(pixel1, 0);
    Sampler sampler2(pixel2, 0);

    SECTION("Different pixels should generate different sequences") {
        bool found_different = false;
        for (int i = 0; i < 100 && !found_different; ++i) {
            if (sampler1.nextSample1D() != sampler2.nextSample1D()) {
                found_different = true;
            }
        }
        REQUIRE(found_different);
    }
}

TEST_CASE("Sampler distribution tests", "[sampler]") {
    UInt2 pixel{123, 456};
    Sampler sampler(pixel, 0);

    SECTION("1D distribution test") {
        constexpr int num_bins = 10;
        constexpr int num_samples = 10000;
        std::vector<int> bins(num_bins, 0);

        for (int i = 0; i < num_samples; ++i) {
            float sample = sampler.nextSample1D();
            int bin =
                std::min(static_cast<int>(sample * num_bins), num_bins - 1);
            bins[bin]++;
        }

        float expected = static_cast<float>(num_samples) / num_bins;
        float tolerance = expected * 0.2f;

        for (int count : bins) {
            REQUIRE(count > (expected - tolerance));
            REQUIRE(count < (expected + tolerance));
        }
    }

    SECTION("2D distribution test - quadrant coverage") {
        constexpr int num_samples = 1000;
        bool found_in_q1 = false;  // (0.5-1.0, 0.5-1.0)
        bool found_in_q2 = false;  // (0.0-0.5, 0.5-1.0)
        bool found_in_q3 = false;  // (0.0-0.5, 0.0-0.5)
        bool found_in_q4 = false;  // (0.5-1.0, 0.0-0.5)

        for (int i = 0; i < num_samples; ++i) {
            auto xy = sampler.nextSample2D();
            if (xy.x >= 0.5f && xy.y >= 0.5f) found_in_q1 = true;
            if (xy.x < 0.5f && xy.y >= 0.5f) found_in_q2 = true;
            if (xy.x < 0.5f && xy.y < 0.5f) found_in_q3 = true;
            if (xy.x >= 0.5f && xy.y < 0.5f) found_in_q4 = true;
        }

        REQUIRE(found_in_q1);
        REQUIRE(found_in_q2);
        REQUIRE(found_in_q3);
        REQUIRE(found_in_q4);
    }
}

TEST_CASE("Sampler sequence consistency", "[sampler]") {
    UInt2 pixel{123, 456};

    SECTION("Same seed should produce same sequence") {
        Sampler sampler1(pixel, 0);
        Sampler sampler2(pixel, 0);

        for (int i = 0; i < 100; ++i) {
            REQUIRE(sampler1.nextSample1D() == sampler2.nextSample1D());
        }
    }

    SECTION("Different sample indices should produce different sequences") {
        Sampler sampler1(pixel, 0);
        Sampler sampler2(pixel, 1);

        bool found_different = false;
        for (int i = 0; i < 100 && !found_different; ++i) {
            if (sampler1.nextSample1D() != sampler2.nextSample1D()) {
                found_different = true;
            }
        }
        REQUIRE(found_different);
    }
}

TEST_CASE("Sampler statistical tests", "[sampler]") {
    UInt2 pixel{123, 456};
    Sampler sampler(pixel, 0);

    SECTION("Mean should be approximately 0.5") {
        constexpr int num_samples = 10000;
        double sum = 0.0;

        for (int i = 0; i < num_samples; ++i) {
            sum += sampler.nextSample1D();
        }

        double mean = sum / num_samples;
        REQUIRE_THAT(mean, WithinAbs(0.5, 0.05));
    }

    SECTION("Variance should be approximately 1/12") {
        constexpr int num_samples = 10000;
        std::vector<double> samples;
        samples.reserve(num_samples);

        for (int i = 0; i < num_samples; ++i) {
            samples.push_back(sampler.nextSample1D());
        }

        double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
        double mean = sum / num_samples;

        double variance = 0.0;
        for (double sample : samples) {
            variance += (sample - mean) * (sample - mean);
        }
        variance /= (num_samples - 1);

        REQUIRE_THAT(variance, WithinAbs(1.0 / 12.0, 0.01));
    }
}
