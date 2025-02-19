#include <CLI11.hpp>

#include "CRaystal.h"

using namespace CRay;

void crayRenderSample(const PathTraceIntegrator::Ref& pIntegrator,
                      Scene::Ref pScene, int spp,
                      const std::filesystem::path& outPath) {
    pIntegrator->dispatch(*pScene, spp);

    auto pImage = pScene->getCamera()->getSensor()->createImage();
    pImage->writeEXR(outPath.string());
}

int main(int argc, char const* argv[]) {
    Logger::init();
    Spectrum::initialize();

    std::string modelPath;
    int spp = 16;
    bool noMIS;

    CLI::App app{"CRaystal renderer"};
    app.add_option("modelXML", modelPath, "Path to model XML file")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("--spp", spp, "Samples per pixel")->default_val(16);
    app.add_flag("--no-mis", noMIS, "Disable multiple importance sampling");
    CLI11_PARSE(app, argc, argv);

    Importer importer;
    auto pScene = importer.import(modelPath);
    PathTraceIntegrator::Configs integratorConf;
    integratorConf.useMIS = !noMIS;
    auto pIntegrator = std::make_shared<PathTraceIntegrator>(integratorConf);

    std::filesystem::path outPath = "craystal.exr";

    crayRenderSample(pIntegrator, pScene, spp, outPath);
    return 0;
}
