#pragma once
#include <filesystem>

#include "Core/Macros.h"
#include "Scene/Scene.h"

namespace CRay {

class CRAYSTAL_API Importer {
   public:
    Importer() = default;

    /** Create scene from XML scene configuration.
     */
    Scene::Ref import(const std::filesystem::path& path);

   private:
};

}  // namespace CRay
