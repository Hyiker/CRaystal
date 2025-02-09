#include "Importer.h"

#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <assimp/Importer.hpp>
#include <pugixml.hpp>

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

static SceneData createSceneData(const std::filesystem::path& path) {
    SceneData data;

    Assimp::Importer importer;

    const aiScene* scene = importer.ReadFile(
        path.string(), aiProcess_Triangulate | aiProcess_GenNormals |
                           aiProcess_CalcTangentSpace |
                           aiProcess_JoinIdenticalVertices |
                           aiProcess_RemoveRedundantMaterials);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE ||
        !scene->mRootNode) {
        CRAYSTAL_THROW("Assimp error: {}", importer.GetErrorString());
    }

    data.meshes.resize(scene->mNumMeshes);
    for (unsigned int i = 0; i < scene->mNumMeshes; i++) {
        const aiMesh* mesh = scene->mMeshes[i];
        auto& meshData = data.meshes[i];

        meshData.position.resize(mesh->mNumVertices);
        meshData.normal.resize(mesh->mNumVertices);
        meshData.texCrd.resize(mesh->mNumVertices);
        meshData.materialID = mesh->mMaterialIndex;

        for (unsigned int j = 0; j < mesh->mNumVertices; j++) {
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

        for (unsigned int j = 0; j < mesh->mNumFaces; j++) {
            const aiFace& face = mesh->mFaces[j];
            for (unsigned int k = 0; k < face.mNumIndices; k++) {
                meshData.index.push_back(face.mIndices[k]);
            }
        }
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
        matData.type = MaterialType::Principled;
        logDebug(
            "Loaded material: {}, transparency: {}, shininess: {}, IoR: {}",
            material->GetName().C_Str(), transparency, matData.shininess,
            matData.IoR);
    }

    return data;
}

Scene::Ref Importer::import(const std::filesystem::path& path) {
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

    // Parse scene
    auto objPath = path;
    objPath.replace_extension(".obj");
    auto sceneData = createSceneData(objPath);

    auto pScene = std::make_shared<Scene>(std::move(sceneData));
    pScene->setCamera(pCamera);

    logInfo("{} Meshes imported.", sceneData.meshes.size());

    return pScene;
}

}  // namespace CRay
