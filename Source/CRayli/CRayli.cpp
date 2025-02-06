#include <CLI11.hpp>

#include "CRaystal.h"
#include "Core/Walkthrough.h"
#include "Utils/Importer.h"

using namespace CRay;
int main(int argc, char const *argv[]) {
    Logger::init();
    Spectrum::initialize();

    std::string modelPath;
    int spp = 16;

    CLI::App app{"CRaystal renderer"};
    app.add_option("modelXML", modelPath, "Path to model XML file")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("--spp", spp, "Samples per pixel")->default_val(16);
    CLI11_PARSE(app, argc, argv);

    Importer importer;
    auto pScene = importer.import(modelPath);

    crayRenderSample(pScene, spp);
    return 0;
}
