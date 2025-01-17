#include <fmt/format.h>

#include "CRaystal.h"

using namespace CRay;
int main(int argc, char const *argv[]) {
    Logger::init();
    Spectrum::initialize();
    logInfo("Hello world from CRayli");
    return 0;
}
