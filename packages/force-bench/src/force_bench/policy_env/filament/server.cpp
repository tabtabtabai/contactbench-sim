// Resident Filament renderer for ContactBench policy cameras (render version 4).
//
// Loads a glb scene once (one node per body, named in body_names.txt), then serves requests over stdin/stdout:
//   request  (little-endian): int32 magic 0x46494C31 ("FIL1"), nbodies x 7 float32 body poses (px py pz qx qy qz qw,
//            world), then per camera 17 float32: a column-major 4x4 camera-to-world matrix (GL axes: looks down -Z,
//            +Y up) and the vertical field of view in degrees
//   response: per camera W*H*3 uint8 RGB, rows top to bottom
// An int32 magic of 0 ends the session.
//
// usage: filament_server scene.glb body_names.txt (ibl.ktx sh.txt | - -) WxH ncams [ibl_lux sun_lux] [key=value ...]
#include <filament/Camera.h>
#include <filament/Engine.h>
#include <filament/ColorGrading.h>
#include <filament/IndexBuffer.h>
#include <filament/IndirectLight.h>
#include <filament/RenderableManager.h>
#include <filament/VertexBuffer.h>
#include <geometry/SurfaceOrientation.h>
#include <filament/Skybox.h>
#include <filament/ToneMapper.h>
#include <map>
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
#include <math/half.h>
#include <math/mat4.h>
#include <math/vec4.h>
#include <math/quat.h>
#include <utils/EntityManager.h>

#include <cmath>
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
    // optional key=value look options after the positional arguments:
    //   sun_dir=x,y,z (travel direction)  sun_color=r,g,b  sun_radius_deg=a  ev100=e  tonemap=aces|aceslegacy|filmic|
    //   linear|pbrneutral|agx  skybox=0|1 (draw the IBL as the background)  ao_radius=m
    std::map<std::string, std::string> opt;
    for (int i = 9; i < argc; i++) { std::string s = argv[i]; auto k = s.find('='); if (k != std::string::npos) opt[s.substr(0, k)] = s.substr(k + 1); }
    auto f3 = [&](const char* k, float3 d) { if (!opt.count(k)) return d; float3 v; std::sscanf(opt[k].c_str(), "%f,%f,%f", &v.x, &v.y, &v.z); return v; };
    auto f1 = [&](const char* k, float d) { return opt.count(k) ? (float) std::atof(opt[k].c_str()) : d; };
    float3 sunDir = f3("sun_dir", {-0.5f, 0.6f, -0.62f}), sunColor = f3("sun_color", {1, 1, 1});
    float sunRadius = f1("sun_radius_deg", 0.545f), ev100 = f1("ev100", 14.966f), aoRadius = f1("ao_radius", 0.15f), aoIntensity = f1("ao_intensity", 1.0f);
    bool ssct = f1("ssct", 0) > 0.5f;
    std::string tonemap = opt.count("tonemap") ? opt["tonemap"] : "";
    bool skybox = f1("skybox", 0) > 0.5f, ssr = f1("ssr", 0) > 0.5f;
    // refl_ktx: a separate (e.g. dimmer) reflections cubemap, keeping the irradiance from sh.txt; skybox_ktx: the
    // background cubemap (default: the reflections map)
    std::string reflKtx = opt.count("refl_ktx") ? opt["refl_ktx"] : "", skyKtx = opt.count("skybox_ktx") ? opt["skybox_ktx"] : "";

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

    // the environment: ibl_color=r,g,b builds a uniform dome in-process (a flat cubemap and its SH: Isaac's DomeLight has no
    // texture); otherwise argv[3]/argv[4] are a cmgen KTX cubemap and its sh.txt
    Texture* iblTex = nullptr;
    float3 sh[9] = {};
    if (opt.count("ibl_color")) {
        float3 c = f3("ibl_color", {1, 1, 1});
        const uint32_t S = 16;
        auto* px = new std::vector<math::half4>(size_t(S) * S * 6, math::half4{math::half(c.x), math::half(c.y), math::half(c.z), math::half(1.f)});
        iblTex = Texture::Builder().width(S).height(S).levels(1).sampler(Texture::Sampler::SAMPLER_CUBEMAP)
            .format(Texture::InternalFormat::RGBA16F).build(*engine);
        iblTex->setImage(*engine, 0, 0, 0, 0, S, S, 6, Texture::PixelBufferDescriptor(px->data(), px->size() * sizeof(math::half4),
            Texture::Format::RGBA, Texture::Type::HALF, [](void*, size_t, void* u) { delete (std::vector<math::half4>*) u; }, px));
        sh[0] = c;                                     // L00 of a constant environment (cmgen's pre-scaled base) = its colour
    } else {
        auto ibl = readFile(argv[3]);
        auto* bundle = new image::Ktx1Bundle(ibl.data(), (uint32_t) ibl.size());
        iblTex = ktxreader::Ktx1Reader::createTexture(engine, bundle, false);
        std::ifstream f(argv[4]); std::string line; int i = 0;
        while (std::getline(f, line) && i < 9) { auto a = line.find('('), b = line.find(')');
            if (a == std::string::npos) continue; std::stringstream ss(line.substr(a + 1, b - a - 1)); char ch;
            ss >> sh[i].x >> ch >> sh[i].y >> ch >> sh[i].z; i++; }
    }
    static std::vector<std::vector<uint8_t>> keep;     // the bytes stay alive for the bundles
    auto loadKtx = [&](const std::string& path) {
        keep.push_back(readFile(path.c_str()));
        return ktxreader::Ktx1Reader::createTexture(engine, new image::Ktx1Bundle(keep.back().data(), (uint32_t) keep.back().size()), false);
    };
    Texture* reflTex = reflKtx.empty() ? iblTex : loadKtx(reflKtx);
    Texture* skyTex = skyKtx.empty() ? iblTex : loadKtx(skyKtx);
    scene->setIndirectLight(IndirectLight::Builder().reflections(reflTex).irradiance(3, sh).intensity(iblLux).build(*engine));
    if (skybox) scene->setSkybox(Skybox::Builder().environment(skyTex).intensity(iblLux).build(*engine));
    Entity sun = utils::EntityManager::get().create();
    LightManager::Builder(LightManager::Type::SUN).color(sunColor).intensity(sunLux).sunAngularRadius(sunRadius)
        .direction(normalize(sunDir)).castShadows(true).build(*engine, sun);
    ColorGrading* grading = nullptr;
    if (!tonemap.empty()) {
        ToneMapper* tm = nullptr;
        if (tonemap == "aces") tm = new ACESToneMapper();
        else if (tonemap == "aceslegacy") tm = new ACESLegacyToneMapper();
        else if (tonemap == "filmic") tm = new FilmicToneMapper();
        else if (tonemap == "linear") tm = new LinearToneMapper();
        else if (tonemap == "pbrneutral") tm = new PBRNeutralToneMapper();
        else if (tonemap == "agx") tm = new AgxToneMapper();
        if (tm) grading = ColorGrading::Builder().toneMapper(tm).build(*engine);
    }
    std::fprintf(stderr, "[filament] sun (%.3f,%.3f,%.3f) %.0f lux r=%.2f deg, ibl %.0f, ev100 %.2f, tonemap %s, skybox %d\n",
                 sunDir.x, sunDir.y, sunDir.z, sunLux, sunRadius, iblLux, ev100, tonemap.empty() ? "default" : tonemap.c_str(), skybox);
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
        // EV100 = log2(N^2 / t) at ISO 100; Filament clamps the shutter time, so hold t = 1/125 s and set the aperture
        cam->setExposure(std::sqrt(std::pow(2.f, ev100) / 125.f), 1.f / 125.f, 100.f);
        View* v = engine->createView();
        v->setScene(scene); v->setCamera(cam); v->setViewport({0, 0, (uint32_t) W, (uint32_t) H});
        v->setRenderTarget(rt);
        {
            View::AmbientOcclusionOptions ao{};
            ao.radius = aoRadius; ao.intensity = aoIntensity; ao.enabled = true;
            if (ssct) { ao.ssct.enabled = true; ao.ssct.lightDirection = normalize(sunDir); ao.ssct.shadowDistance = 0.3f;
                        ao.ssct.lightConeRad = 0.07f; ao.ssct.intensity = 0.8f; }
            v->setAmbientOcclusionOptions(ao);
        }
        if (grading) v->setColorGrading(grading);
        if (ssr) v->setScreenSpaceReflectionsOptions({.thickness = 0.05f, .bias = 0.01f, .maxDistance = 3.0f, .stride = 2.0f, .enabled = true});
        v->setMultiSampleAntiAliasingOptions({.enabled = true, .sampleCount = 4});
        v->setAntiAliasing(View::AntiAliasing::FXAA);
        v->setShadowType(View::ShadowType::PCSS);
        views.push_back(v); rts.push_back(rt); cams.push_back(cam);
    }
    std::fprintf(stderr, "[filament] ready\n");
    std::fflush(stderr);

    // deform=<mesh file>: one node (deform_node=<name>, default "deform/towel") whose vertex positions and normals
    // arrive with every request (after the cameras: nv x 6 float32, position then normal).  The file holds the static
    // topology: int32 nv, int32 ntri, nv x 2 float32 uv, ntri x 3 uint32 indices.  The node's own glTF material is
    // reused; its static primitive is removed from the scene.
    int dnv = 0, dnt = 0;
    std::vector<float> duv, dpn, dpos, dnrm; std::vector<uint32_t> didx; std::vector<short4> dq;
    std::vector<uint32_t> dcol;                       // white vertex colours: gltfio's ubershader multiplies by COLOR
    VertexBuffer* dvb = nullptr;
    if (opt.count("deform")) {
        auto bytes = readFile(opt["deform"].c_str());
        std::memcpy(&dnv, bytes.data(), 4); std::memcpy(&dnt, bytes.data() + 4, 4);
        duv.resize(size_t(dnv) * 2); std::memcpy(duv.data(), bytes.data() + 8, duv.size() * 4);
        didx.resize(size_t(dnt) * 3); std::memcpy(didx.data(), bytes.data() + 8 + duv.size() * 4, didx.size() * 4);
        dpn.resize(size_t(dnv) * 6); dpos.resize(size_t(dnv) * 3); dnrm.resize(size_t(dnv) * 3); dq.resize(dnv);
        std::string nodeName = opt.count("deform_node") ? opt["deform_node"] : "deform/towel";
        Entity src = asset->getFirstEntityByName(nodeName.c_str());
        auto& rm = engine->getRenderableManager();
        if (src.isNull() || !rm.hasComponent(src)) { std::cerr << "[filament] no renderable node " << nodeName << "\n"; return 4; }
        MaterialInstance* mi = rm.getMaterialInstanceAt(rm.getInstance(src), 0);
        scene->remove(src);
        dcol.assign(dnv, 0xFFFFFFFFu);
        dvb = VertexBuffer::Builder().vertexCount(dnv).bufferCount(4)
            .attribute(VertexAttribute::POSITION, 0, VertexBuffer::AttributeType::FLOAT3)
            .attribute(VertexAttribute::TANGENTS, 1, VertexBuffer::AttributeType::SHORT4)
            .normalized(VertexAttribute::TANGENTS)
            .attribute(VertexAttribute::UV0, 2, VertexBuffer::AttributeType::FLOAT2)
            .attribute(VertexAttribute::COLOR, 3, VertexBuffer::AttributeType::UBYTE4).normalized(VertexAttribute::COLOR)
            .build(*engine);
        dvb->setBufferAt(*engine, 2, VertexBuffer::BufferDescriptor(duv.data(), duv.size() * 4, nullptr));
        dvb->setBufferAt(*engine, 3, VertexBuffer::BufferDescriptor(dcol.data(), dcol.size() * 4, nullptr));
        IndexBuffer* dib = IndexBuffer::Builder().indexCount(dnt * 3).bufferType(IndexBuffer::IndexType::UINT).build(*engine);
        dib->setBuffer(*engine, IndexBuffer::BufferDescriptor(didx.data(), didx.size() * 4, nullptr));
        Entity cloth = utils::EntityManager::get().create();
        RenderableManager::Builder(1).boundingBox({{-20, -20, -20}, {20, 20, 20}}).material(0, mi)
            .geometry(0, RenderableManager::PrimitiveType::TRIANGLES, dvb, dib)
            .castShadows(true).receiveShadows(true).culling(false).build(*engine, cloth);
        scene->addEntity(cloth);
        std::fprintf(stderr, "[filament] deform node %s: %d vertices, %d triangles\n", nodeName.c_str(), dnv, dnt);
    }
    std::vector<float> poses(size_t(nb) * 7), camf(size_t(ncams) * 17);
    while (true) {
        int32_t magic = 0;
        if (!readAll(&magic, 4) || magic == 0) break;
        if (magic != 0x46494C31) { std::cerr << "[filament] bad magic\n"; return 3; }
        if (!readAll(poses.data(), poses.size() * 4) || !readAll(camf.data(), camf.size() * 4)) break;
        if (dvb) {
            if (!readAll(dpn.data(), dpn.size() * 4)) break;
            for (int i = 0; i < dnv; i++) {
                std::memcpy(&dpos[i * 3], &dpn[i * 6], 12); std::memcpy(&dnrm[i * 3], &dpn[i * 6 + 3], 12);
            }
            auto* so = geometry::SurfaceOrientation::Builder().vertexCount(dnv)
                .normals((const float3*) dnrm.data()).uvs((const float2*) duv.data())
                .positions((const float3*) dpos.data()).triangleCount(dnt).triangles((const uint3*) didx.data()).build();
            so->getQuats(dq.data(), dnv);
            delete so;
            dvb->setBufferAt(*engine, 0, VertexBuffer::BufferDescriptor(dpos.data(), dpos.size() * 4, nullptr));
            dvb->setBufferAt(*engine, 1, VertexBuffer::BufferDescriptor(dq.data(), dq.size() * sizeof(short4), nullptr));
        }
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
