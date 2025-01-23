#include "CRaystal.h"
#include "Walkthrough.h"
namespace CRay {

__global__ void renderKernel(const SceneView* pScene,
                             const CameraProxy* pCamera, SensorData* pSensor) {
    UInt2 xy(blockIdx.x * blockDim.x + threadIdx.x,
             blockIdx.y * blockDim.y + threadIdx.y);

    UInt2 sensorSize = pSensor->size;
    if (xy.x >= sensorSize.x || xy.y >= sensorSize.y) {
        return;
    }

    Float2 pixel = Float2(xy) + Float2(0.5);

    auto ray = pCamera->generateRay(sensorSize, pixel);
    Spectrum color;

    RayHit rayHit;
    rayHit.ray = ray;
    if (pScene->intersect(rayHit)) {
        const Intersection it = pScene->createIntersection(rayHit);

        color = Spectrum(it.faceNormal);
    }

    pSensor->addSample(color, pixel);
}

void crayRenderSample() {
    // Build scene
    Scene scene;
    Sphere sphere;

    sphere.center = Float3(0);
    sphere.radius = 1.f;

    scene.addSphere(sphere);
    scene.finalize();
    scene.updateDeviceData();

    UInt2 size(512, 512);

    Sensor sensor(size, 1);
    sensor.updateDeviceData();

    Camera cam;
    cam.setWorldPosition(Float3(0, 0, 5));
    cam.setTarget(Float3(0, 0, 0));
    cam.calculateCameraData();
    cam.updateDeviceData();

    renderKernel<<<dim3(size.x, size.y, 1), dim3(16, 16, 1)>>>(
        scene.getDeviceView(), cam.getDeviceView(), sensor.getDeviceView());

    sensor.readbackDeviceData();
    auto pImage = sensor.createImage();
    pImage->writeEXR("walkthrough.exr");
}

}  // namespace CRay
