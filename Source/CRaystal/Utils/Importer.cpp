#include "Importer.h"

#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <assimp/Importer.hpp>
#include <numeric>
#include <pugixml.hpp>
#include <sstream>

#include "Core/Error.h"
#include "Logger.h"

namespace fs = std::filesystem;
namespace CRay {

static Float3 parseFloat3(const pugi::xml_node& node) {
    return Float3(node.attribute("x").as_float(),
                  node.attribute("y").as_float(),
                  node.attribute("z").as_float());
}

static Camera::Ref createCamera(const pugi::xml_node& cameraNode) {
    auto camera = std::make_shared<Camera>();

    float fovY = cameraNode.attribute("fovy").as_float();
    camera->setFovY(fovY);

    pugi::xml_node eyeNode = cameraNode.child("eye");
    if (eyeNode) {
        camera->setWorldPosition(parseFloat3(eyeNode));
    }

    pugi::xml_node lookatNode = cameraNode.child("lookat");
    if (lookatNode) {
        camera->setTarget(parseFloat3(lookatNode));
    }

    pugi::xml_node upNode = cameraNode.child("up");
    if (upNode) {
        camera->setUp(parseFloat3(upNode));
    }

    UInt2 sensorSize(cameraNode.attribute("width").as_uint(),
                     cameraNode.attribute("height").as_uint());
    auto sensor = std::make_shared<Sensor>(sensorSize);

    camera->setSensor(sensor);
    camera->calculateCameraData();

    return camera;
}

static Spectrum toIllumSpectrum(const std::string& str) {
    std::string token;
    Float3 numbers;
    int idx = 0;

    std::istringstream iss(str);
    while (std::getline(iss, token, ',') && idx < 3) {
        std::istringstream convert(token);
        convert >> numbers[idx];
        idx++;
    }
    return Spectrum::fromRGB(numbers, true);
}

using EmissiveDict = Importer::EmissiveDict;

static EmissiveDict createEmissiveDict(const pugi::xml_node& rootNode) {
    std::unordered_map<std::string, Spectrum> result;
    for (const auto& lightNode : rootNode.children("light")) {
        std::string mtlName = lightNode.attribute("mtlname").as_string();
        std::string radianceStr = lightNode.attribute("radiance").as_string();
        result[mtlName] = toIllumSpectrum(radianceStr);
    }
    return result;
}

std::filesystem::path Importer::resolveResourcePath(
    const std::filesystem::path& relPath) const {
    return mBasePath / relPath;
}

SceneData Importer::createSceneData(const std::filesystem::path& path,
                                    const EmissiveDict& emissiveDict) {
    SceneData data;

    Assimp::Importer importer;

    const aiScene* scene = importer.ReadFile(
        path.string(), aiProcess_Triangulate | aiProcess_GenNormals |
                           aiProcess_CalcTangentSpace |
                           aiProcess_JoinIdenticalVertices);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE ||
        !scene->mRootNode) {
        CRAYSTAL_THROW("Assimp error: {}", importer.GetErrorString());
    }

    data.materials.resize(scene->mNumMaterials);
    for (unsigned int i = 0; i < scene->mNumMaterials; i++) {
        const aiMaterial* material = scene->mMaterials[i];
        auto& matData = data.materials[i];

        aiColor3D color(0.f, 0.f, 0.f);

        if (material->Get(AI_MATKEY_COLOR_DIFFUSE, color) == AI_SUCCESS) {
            matData.diffuseRefl =
                Spectrum::fromRGB(Float3(color.r, color.g, color.b));
        }

        aiString diffusePath;
        if (material->GetTexture(aiTextureType_DIFFUSE, 0, &diffusePath) ==
            AI_SUCCESS) {
            matData.flags |= uint32_t(MaterialFlags::hasDiffuseTexture);
            matData.diffuseHandle = data.textureImages.size();
            data.textureImages.push_back(
                Image::load(resolveResourcePath(diffusePath.C_Str())));
        }

        if (material->Get(AI_MATKEY_COLOR_SPECULAR, color) == AI_SUCCESS) {
            matData.specularRefl = Spectrum(Float3(color.r, color.g, color.b));
        }

        float transparency = 0.0f;
        if (material->Get(AI_MATKEY_OPACITY, transparency) == AI_SUCCESS) {
            matData.transmittance = Spectrum(1.0f - transparency);
        }

        float shininess = 0.0f;
        if (material->Get(AI_MATKEY_SHININESS, shininess) == AI_SUCCESS) {
            matData.shininess = shininess;
        }

        float ior = 1.0f;
        if (material->Get(AI_MATKEY_REFRACTI, ior) == AI_SUCCESS) {
            matData.IoR = ior;
        }

        // Determine material type based on properties
        matData.type = MaterialType::Phong;
        std::string mtlName = material->GetName().C_Str();
        if (auto it = emissiveDict.find(mtlName); it != emissiveDict.end()) {
            matData.emission = it->second;
        }

        matData.finalize();

        logDebug(
            "Loaded material: {}, transparency: {}, shininess: {}, IoR: {}, "
            "radiance: {}",
            material->GetName().C_Str(), transparency, matData.shininess,
            matData.IoR, matData.emission[0]);
    }

    uint32_t primitiveID = 0u;
    data.meshes.resize(scene->mNumMeshes);
    for (uint32_t i = 0; i < scene->mNumMeshes; i++) {
        const aiMesh* mesh = scene->mMeshes[i];
        auto& meshData = data.meshes[i];

        meshData.position.resize(mesh->mNumVertices);
        meshData.normal.resize(mesh->mNumVertices);
        meshData.texCrd.resize(mesh->mNumVertices);
        meshData.materialID = mesh->mMaterialIndex;

        for (uint32_t j = 0; j < mesh->mNumVertices; j++) {
            meshData.position[j] =
                Float3(mesh->mVertices[j].x, mesh->mVertices[j].y,
                       mesh->mVertices[j].z);

            if (mesh->HasNormals()) {
                meshData.normal[j] =
                    Float3(mesh->mNormals[j].x, mesh->mNormals[j].y,
                           mesh->mNormals[j].z);
            }

            if (mesh->HasTextureCoords(0)) {
                meshData.texCrd[j] = Float2(mesh->mTextureCoords[0][j].x,
                                            mesh->mTextureCoords[0][j].y);
            }
        }

        for (uint32_t j = 0; j < mesh->mNumFaces; j++) {
            const aiFace& face = mesh->mFaces[j];
            for (uint32_t k = 0; k < face.mNumIndices; k++) {
                meshData.index.push_back(face.mIndices[k]);
            }
            if (data.materials[meshData.materialID].isEmissive()) {
                uint32_t offset = data.emissiveIndex.size();
                data.emissiveIndex.resize(offset + face.mNumIndices / 3);
                std::iota(data.emissiveIndex.begin() + offset,
                          data.emissiveIndex.end(), primitiveID);
            }
            primitiveID += face.mNumIndices / 3;
        }
    }

    return data;
}

Scene::Ref Importer::import(const std::filesystem::path& path) {
    mBasePath = path.parent_path();

    logInfo("Importing from {}", path.string());

    CRAYSTAL_CHECK(path.extension() == ".xml",
                   "Imported file must be xml file.");

    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(path.c_str());
    if (!result) CRAYSTAL_THROW("Invalid XML file {}.", path.string());

    pugi::xml_node cameraNode = doc.child("camera");
    CRAYSTAL_CHECK(!cameraNode.empty(), "No camera node found.");

    // Parse camera
    auto pCamera = createCamera(cameraNode);

    // Parse light
    auto emissiveDict = createEmissiveDict(doc);

    // Parse scene
    auto objPath = path;
    objPath.replace_extension(".obj");
    auto sceneData = createSceneData(objPath, emissiveDict);

    auto pScene = std::make_shared<Scene>(std::move(sceneData));
    pScene->setCamera(pCamera);

    logInfo("Meshes: {}.", sceneData.meshes.size());
    logInfo("Emissive triangles: {}.", sceneData.emissiveIndex.size());

    return pScene;
}

}  // namespace CRay
