// Resident Filament renderer for ContactBench policy cameras (render version 4).
//
// Loads a glb scene once (one node per body, named in body_names.txt), then serves requests over stdin/stdout:
//   request  (little-endian): int32 magic 0x46494C31 ("FIL1"), nbodies x 7 float32 body poses (px py pz qx qy qz qw,
//            world), then per camera 17 float32: a column-major 4x4 camera-to-world matrix (GL axes: looks down -Z,
//            +Y up) and the vertical field of view in degrees
//   response: per camera W*H*3 uint8 RGB, rows top to bottom
// An int32 magic of 0 ends the session.
//
// usage: filament_server scene.glb body_names.txt ibl.ktx sh.txt WxH ncams [ibl_lux sun_lux]
#include <filament/Camera.h>
#include <filament/Engine.h>
#include <filament/IndirectLight.h>
#include <filament/LightManager.h>
#include <filament/RenderTarget.h>
#include <filament/Renderer.h>
#include <filament/Scene.h>
#include <filament/Texture.h>
#include <filament/TransformManager.h>
#include <filament/View.h>
#include <filament/Viewport.h>
#include <gltfio/AssetLoader.h>
#include <gltfio/FilamentAsset.h>
#include <gltfio/MaterialProvider.h>
#include <gltfio/ResourceLoader.h>
#include <gltfio/TextureProvider.h>
#include <gltfio/materials/uberarchive.h>
#include <image/Ktx1Bundle.h>
#include <ktxreader/Ktx1Reader.h>
#include <math/mat4.h>
#include <math/quat.h>
#include <utils/EntityManager.h>

#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace filament;
using namespace filament::math;
using namespace filament::gltfio;
using utils::Entity;

static std::vector<uint8_t> readFile(const char* path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static bool readAll(void* dst, size_t n) { return std::fread(dst, 1, n, stdin) == n; }

int main(int argc, char** argv) {
    if (argc < 7) { std::cerr << "usage: filament_server scene.glb names.txt ibl.ktx sh.txt WxH ncams [ibl_lux sun_lux]\n"; return 1; }
    int W = 0, H = 0; std::sscanf(argv[5], "%dx%d", &W, &H);
    int ncams = std::atoi(argv[6]);
    float iblLux = argc > 7 ? std::atof(argv[7]) : 18000.f, sunLux = argc > 8 ? std::atof(argv[8]) : 75000.f;

    // the pixel stream gets a private copy of stdout; Filament's own logging (which prints to stdout) goes to stderr,
    // so no text can ever land inside the binary responses
    int dataFd = dup(STDOUT_FILENO);
    dup2(STDERR_FILENO, STDOUT_FILENO);
    FILE* data = fdopen(dataFd, "wb");

    Engine* engine = Engine::create(Engine::Backend::VULKAN);
    Renderer* renderer = engine->createRenderer();
    Scene* scene = engine->createScene();

    auto glb = readFile(argv[1]);
    MaterialProvider* materials = createUbershaderProvider(engine, UBERARCHIVE_DEFAULT_DATA, UBERARCHIVE_DEFAULT_SIZE);
    AssetConfiguration acfg{}; acfg.engine = engine; acfg.materials = materials;
    AssetLoader* loader = AssetLoader::create(acfg);
    FilamentAsset* asset = loader->createAsset(glb.data(), (uint32_t) glb.size());
    if (!asset) { std::cerr << "[filament] glb load failed\n"; return 2; }
    ResourceConfiguration rcfg{}; rcfg.engine = engine; rcfg.normalizeSkinningWeights = true;
    ResourceLoader resources(rcfg);
    resources.addTextureProvider("image/png", createStbProvider(engine));
    resources.loadResources(asset);
    asset->releaseSourceData();
    scene->addEntities(asset->getEntities(), asset->getEntityCount());

    std::vector<Entity> bodies;
    { std::ifstream f(argv[2]); std::string s; while (std::getline(f, s)) bodies.push_back(asset->getFirstEntityByName(s.c_str())); }
    int nb = (int) bodies.size(), found = 0;
    for (auto e : bodies) found += !e.isNull();
    std::fprintf(stderr, "[filament] %d/%d body nodes, %dx%d, %d cameras\n", found, nb, W, H, ncams);
    auto& tm = engine->getTransformManager();

    auto ibl = readFile(argv[3]);
    auto* bundle = new image::Ktx1Bundle(ibl.data(), (uint32_t) ibl.size());
    Texture* iblTex = ktxreader::Ktx1Reader::createTexture(engine, bundle, false);
    float3 sh[9]; { std::ifstream f(argv[4]); std::string line; int i = 0;
        while (std::getline(f, line) && i < 9) { auto a = line.find('('), b = line.find(')');
            if (a == std::string::npos) continue; std::stringstream ss(line.substr(a + 1, b - a - 1)); char c;
            ss >> sh[i].x >> c >> sh[i].y >> c >> sh[i].z; i++; } }
    scene->setIndirectLight(IndirectLight::Builder().reflections(iblTex).irradiance(3, sh).intensity(iblLux).build(*engine));
    Entity sun = utils::EntityManager::get().create();
    LightManager::Builder(LightManager::Type::SUN).color({1, 1, 1}).intensity(sunLux)
        .direction(normalize(float3{-0.5f, 0.6f, -0.62f})).castShadows(true).build(*engine, sun);
    scene->addEntity(sun);

    std::vector<View*> views; std::vector<RenderTarget*> rts; std::vector<Camera*> cams;
    std::vector<std::vector<uint8_t>> rgba(ncams, std::vector<uint8_t>(size_t(W) * H * 4));
    std::vector<uint8_t> rgb(size_t(W) * H * 3);
    for (int c = 0; c < ncams; c++) {
        Texture* color = Texture::Builder().width(W).height(H).levels(1)
            .usage(Texture::Usage::COLOR_ATTACHMENT | Texture::Usage::SAMPLEABLE | Texture::Usage::BLIT_SRC)
            .format(Texture::InternalFormat::RGBA8).build(*engine);
        Texture* depth = Texture::Builder().width(W).height(H).levels(1)
            .usage(Texture::Usage::DEPTH_ATTACHMENT).format(Texture::InternalFormat::DEPTH24).build(*engine);
        RenderTarget* rt = RenderTarget::Builder().texture(RenderTarget::AttachmentPoint::COLOR, color)
            .texture(RenderTarget::AttachmentPoint::DEPTH, depth).build(*engine);
        Camera* cam = engine->createCamera(utils::EntityManager::get().create());
        cam->setExposure(16.f, 1.f / 125.f, 100.f);
        View* v = engine->createView();
        v->setScene(scene); v->setCamera(cam); v->setViewport({0, 0, (uint32_t) W, (uint32_t) H});
        v->setRenderTarget(rt);
        v->setAmbientOcclusionOptions({.radius = 0.15f, .intensity = 1.0f, .enabled = true});
        v->setMultiSampleAntiAliasingOptions({.enabled = true, .sampleCount = 4});
        v->setAntiAliasing(View::AntiAliasing::FXAA);
        v->setShadowType(View::ShadowType::PCSS);
        views.push_back(v); rts.push_back(rt); cams.push_back(cam);
    }
    std::fprintf(stderr, "[filament] ready\n");
    std::fflush(stderr);

    std::vector<float> poses(size_t(nb) * 7), camf(size_t(ncams) * 17);
    while (true) {
        int32_t magic = 0;
        if (!readAll(&magic, 4) || magic == 0) break;
        if (magic != 0x46494C31) { std::cerr << "[filament] bad magic\n"; return 3; }
        if (!readAll(poses.data(), poses.size() * 4) || !readAll(camf.data(), camf.size() * 4)) break;
        for (int b = 0; b < nb; b++) {
            if (bodies[b].isNull()) continue;
            const float* p = &poses[b * 7];
            tm.setTransform(tm.getInstance(bodies[b]),
                            mat4f::translation(float3{p[0], p[1], p[2]}) * mat4f(quatf(p[6], p[3], p[4], p[5])));
        }
        for (int c = 0; c < ncams; c++) {
            const float* m = &camf[c * 17];
            mat4f M; std::memcpy(&M, m, 16 * 4);
            cams[c]->setProjection(m[16], double(W) / H, 0.01, 10.0, Camera::Fov::VERTICAL);
            cams[c]->setModelMatrix(M);
            renderer->renderStandaloneView(views[c]);
            renderer->readPixels(rts[c], 0, 0, W, H,
                backend::PixelBufferDescriptor(rgba[c].data(), rgba[c].size(),
                    backend::PixelDataFormat::RGBA, backend::PixelDataType::UBYTE));
        }
        engine->flushAndWait();
        for (int c = 0; c < ncams; c++) {
            for (size_t i = 0, n = size_t(W) * H; i < n; i++) std::memcpy(&rgb[i * 3], &rgba[c][i * 4], 3);
            std::fwrite(rgb.data(), 1, rgb.size(), data);
        }
        std::fflush(data);
    }
    Engine::destroy(&engine);
    return 0;
}
