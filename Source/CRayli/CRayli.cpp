#include "CRaystal.h"
#include "Utils/Importer.h"
#include "Core/Walkthrough.h"

using namespace CRay;
int main(int argc, char const *argv[]) {
    Logger::init();
    Spectrum::initialize();

    Importer importer;
    importer.import(argv[1]);
    return 0;
}
