#pragma once
#include <filesystem>
#include <unordered_map>

#include "Core/Macros.h"
#include "Scene/Scene.h"

namespace CRay {

class CRAYSTAL_API Importer {
   public:
    using EmissiveDict = std::unordered_map<std::string, Spectrum>;

    Importer() = default;

    /** Create scene from XML scene configuration.
     */
    Scene::Ref import(const std::filesystem::path& path);

   private:
    std::filesystem::path resolveResourcePath(
        const std::filesystem::path& relPath) const;

    SceneData createSceneData(const std::filesystem::path& objPath,
                              const EmissiveDict& emissiveDict);

    std::filesystem::path mBasePath;
};

}  // namespace CRay
