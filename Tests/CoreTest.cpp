#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "Core/Spectrum.h"

using namespace CRay;
using namespace Catch::Matchers;

TEST_CASE("Test RGBSpectrum operations") {
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
