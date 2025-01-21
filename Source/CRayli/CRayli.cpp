#include "CRaystal.h"
#include "Core/Walkthrough.h"

using namespace CRay;
int main(int argc, char const *argv[]) {
    Logger::init();
    Spectrum::initialize();

    crayRenderSample();
    return 0;
}
