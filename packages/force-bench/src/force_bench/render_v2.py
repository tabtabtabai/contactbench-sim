"""Render version 2 (clockwork-5 fork): materials, textures and lighting for the ContactBench scenes.

``TB_RENDER_VERSION=2`` turns it on for a process; the default, 1, is upstream's look and nothing here runs.  It
changes only how the scene looks: camera poses, resolutions and fields of view are untouched, and the physics is
not changed (the backdrop walls are visual-only shapes appended after every other shape, so no shape index moves and
nothing new collides).

What it does, within what Newton's tiled-camera raytracer supports (per-shape colors, 2-D textures through a
triplanar projection on any shape, any number of directional/point/spot lights with hard shadow rays, a
hemispheric ambient term, Lambert shading only):

* MATERIALS by shape label (``MATERIALS``): an oak door and frame, a walnut table, brushed brass for the doorknob
  and handle, satin and dark steel for the rosettes, gears and bolts, an opal (frosted) glass bulb, a black bakelite
  socket, a dark anodised dial with a white scale and a red arrow, zinc-plated steel for the bolt and nut.  Each is a
  procedural texture (``_texture``) projected triplanar in the shape's own frame, at a physical scale (cycles per
  metre), so the CAD meshes need no UVs.
* A FLOOR AND WALLS: the ground plane gets large stone tiles, and three plaster walls stand well behind the scene
  (``add_backdrop``) in place of the flat backdrop gradient.
* LIGHTS: a soft key light (``KEY_SAMPLES`` shadow-casting directional lights jittered within ``KEY_CONE_DEG``, so
  shadow edges get a ``KEY_SAMPLES``-step penumbra), a fill and a rim light, each with its own intensity.
* SHADOW ACNE: upstream offsets shadow rays 0.1 mm along the normal, which leaves speckles near the terminator of
  curved, finely tessellated parts (the bulb). Version 2 offsets them by ``SHADOW_BIAS_M`` along the normal and the
  light direction.

The last two need a per-light intensity and a shadow bias, which the raytracer does not expose, so version 2
replaces Newton's ``lighting.create_compute_lighting_function`` with a copy that reads each light's intensity from
the length of its direction vector and uses the larger bias (``_patch_lighting``; Newton 1.5.0).  The renderer has
no specular term and no transparency: metal and glass are carried by their textures and colors only (see
PHOTOREAL.md in the clockwork-5 notes for what a physically based path would take).
"""

from __future__ import annotations

import os
import re

import numpy as np
import warp as wp

KEY_SAMPLES = 6
KEY_CONE_DEG = 7.0
KEY_INTENSITY = 0.80
FILL_INTENSITY = 0.30
RIM_INTENSITY = 0.22
SHADOW_BIAS_M = 1.5e-3
SHADOW_MIN_VISIBILITY = 0.32
KEY_DIR = np.array([-0.5, 0.6, -0.62])     # the v1 key's travel direction, kept so shadows fall the same way
FILL_DIR = np.array([0.72, -0.36, -0.58])
RIM_DIR = np.array([0.15, -0.85, -0.45])   # from behind the scene, toward the cameras
TEX_PX = 512


def version() -> int:
    try:
        return int(os.environ.get("TB_RENDER_VERSION", "1"))
    except ValueError:
        return 1


def enabled() -> bool:
    return version() >= 2


# ── materials ────────────────────────────────────────────────────────────────────────────────────────────────────
# (label regex, material); the first match wins.  Labels are the model's shape labels (``force-bench`` kit parts are
# "kit/<asset>/<part>"); unmatched shapes keep their color.
MATERIALS = [
    (r"^ground_plane$", "floor_tile"),
    (r"^table$", "walnut"),
    (r"^render_v2/wall", "plaster"),
    (r"^kit/door/door(_frame_\d)?$", "oak"),
    (r"^kit/doorknob_module/(doorknob_\d|door_handle_\d)", "brass"),
    (r"^kit/doorknob_module/doorknob_module_(top|bottom)$", "satin_steel"),
    (r"^kit/doorknob_module/doorknob_module_(gear|latch|spring)", "dark_steel"),
    (r"^kit/lock_module/", "dark_steel"),
    (r"^kit/bulb/", "opal_glass"),
    (r"^kit/socket/", "bakelite"),
    (r"^kit/dial/dial_(bottom|top)$", "anodised_black"),
    (r"^kit/dial/dial_grid$", "enamel_white"),
    (r"^kit/dial/dial_arrow$", "enamel_red"),
    (r"^kit/dial/dial_knob", "satin_steel"),
    (r"^kit/dial/(pad_\d|dial_spring)", "dark_steel"),
    (r"^bolt$", "zinc"),
    (r"^nut_shape$", "zinc"),
]

# material -> (base color, texture kind, cycles per metre)
SPEC = {
    "floor_tile": ((0.56, 0.54, 0.50), "tile", 1.6),
    "walnut": ((0.40, 0.27, 0.17), "wood_fine", 2.5),
    "plaster": ((0.80, 0.78, 0.74), "plaster", 0.8),
    "oak": ((0.70, 0.52, 0.33), "wood", 3.0),
    "brass": ((0.80, 0.63, 0.32), "brushed", 18.0),
    "satin_steel": ((0.72, 0.73, 0.75), "brushed", 18.0),
    "dark_steel": ((0.38, 0.39, 0.41), "brushed", 22.0),
    "zinc": ((0.78, 0.79, 0.82), "brushed", 30.0),
    "opal_glass": ((0.95, 0.94, 0.91), "opal", 14.0),
    "bakelite": ((0.13, 0.12, 0.11), "plaster", 20.0),
    "anodised_black": ((0.16, 0.17, 0.19), "brushed", 25.0),
    "enamel_white": ((0.93, 0.93, 0.91), None, 1.0),
    "enamel_red": ((0.84, 0.17, 0.12), None, 1.0),
}


def _fbm(shape, rng, octaves=5):
    """Fractal value noise in [0, 1] on a tileable grid."""
    h, w = shape
    out = np.zeros(shape, np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        n = 4 * 2 ** o
        g = rng.random((n, n)).astype(np.float32)
        ys, xs = np.linspace(0, n, h, endpoint=False), np.linspace(0, n, w, endpoint=False)
        y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
        fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
        fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
        a = g[y0 % n][:, x0 % n]
        b = g[y0 % n][:, (x0 + 1) % n]
        c = g[(y0 + 1) % n][:, x0 % n]
        d = g[(y0 + 1) % n][:, (x0 + 1) % n]
        out += amp * (a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy)
        tot += amp
        amp *= 0.5
    return out / tot


def _texture(kind: str, seed: int = 0) -> np.ndarray:
    """A tileable grayscale-ish RGBA modulation texture (multiplies the material's base color)."""
    rng = np.random.default_rng(seed)
    n = TEX_PX
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / n
    if kind in ("wood", "wood_fine"):
        warp_ = _fbm((n, n), rng, 4)
        rings = 10 if kind == "wood" else 16
        g = np.sin(2 * np.pi * (rings * x + 1.8 * warp_)) * 0.5 + 0.5
        g = g ** 2.2
        fibre = _fbm((n, n), rng, 6)
        v = 0.78 + 0.22 * (0.65 * g + 0.35 * fibre)
        v *= 1.0 - 0.06 * (np.sin(2 * np.pi * 90 * x + 3 * fibre) > 0.96)
        rgb = np.stack([v, v * 0.97, v * 0.93], -1)
    elif kind == "brushed":
        streak = np.repeat(rng.random((n, 1)).astype(np.float32), n, axis=1)
        streak = np.clip(0.5 * streak + 0.5 * np.roll(streak, 1, 0), 0, 1)
        v = 0.86 + 0.14 * (0.6 * streak + 0.4 * _fbm((n, n), rng, 3))
        rgb = np.stack([v, v, v], -1)
    elif kind == "opal":
        v = 0.94 + 0.06 * _fbm((n, n), rng, 4)
        rgb = np.stack([v, v, v * 0.99], -1)
    elif kind == "tile":
        noise = _fbm((n, n), rng, 6)
        v = 0.80 + 0.20 * noise
        grout = (np.minimum(np.abs(x - np.round(x * 2) / 2), np.abs(y - np.round(y * 2) / 2)) < 0.006)
        v = np.where(grout, 0.55, v)
        rgb = np.stack([v, v * 0.98, v * 0.95], -1)
    elif kind == "plaster":
        v = 0.90 + 0.10 * _fbm((n, n), rng, 6)
        rgb = np.stack([v, v, v], -1)
    else:
        rgb = np.ones((n, n, 3), np.float32)
    rgba = np.concatenate([np.clip(rgb, 0, 1), np.ones((n, n, 1), np.float32)], -1)
    return (rgba * 255 + 0.5).astype(np.uint8)


def _match(label: str):
    for pat, mat in MATERIALS:
        if re.search(pat, label):
            return mat
    return None


def apply_colors(model) -> None:
    """Material base colors on the model (shared by every camera); textured shapes then modulate them."""
    if not enabled():
        return
    colors = model.shape_color.numpy().copy()
    for i, lbl in enumerate(model.shape_label):
        mat = _match(str(lbl or ""))
        if mat is not None:
            colors[i] = SPEC[mat][0]
    model.shape_color.assign(colors.astype(np.float32))


# ── the backdrop ─────────────────────────────────────────────────────────────────────────────────────────────────
def add_backdrop(rig) -> None:
    """Three visual-only plaster walls well behind the scene (appended last; no collision, no mass)."""
    if not enabled():
        return
    import newton
    b = rig.builder
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.has_shape_collision = False
    cfg.has_particle_collision = False
    half_t, h = 0.02, 0.9
    for label, p, hx, hy in (("render_v2/wall_back", (0.0, 1.6, h), 2.4, half_t),
                             ("render_v2/wall_left", (-1.8, 0.2, h), half_t, 1.6),
                             ("render_v2/wall_right", (1.8, 0.2, h), half_t, 1.6)):
        b.add_shape_box(body=-1, xform=wp.transform(wp.vec3(*p), wp.quat_identity()), hx=hx, hy=hy, hz=h,
                        cfg=cfg, label=label)


# ── a camera sensor ──────────────────────────────────────────────────────────────────────────────────────────────
def configure_sensor(cam, model) -> None:
    """Version-2 textures and lights on one SensorTiledCamera (call before its first render)."""
    if not enabled():
        return
    _patch_lighting()
    from newton._src.sensors.warp_raytrace.types import TextureData, TextureProjectionMode
    cfg = cam.default_render_config
    cfg.enable_textures = True
    cfg.enable_shadows = True
    cfg.texture_projection_mode = TextureProjectionMode.TRIPLANAR
    ctx = getattr(cam, "_SensorTiledCamera__render_context", None)
    if ctx is None:
        return
    dev = ctx.device
    ids = ctx.shape_texture_ids.numpy().copy() if ctx.shape_texture_ids is not None else \
        -np.ones(len(model.shape_label), np.int32)
    if (ids >= 0).any():          # keep any texture a mesh brought (none do in these scenes)
        return _lights(ctx, dev)
    datas, by_mat = [], {}
    for i, lbl in enumerate(model.shape_label):
        mat = _match(str(lbl or ""))
        if mat is None or SPEC[mat][1] is None:
            continue
        if mat not in by_mat:
            d = TextureData()
            d.texture = wp.Texture2D(_texture(SPEC[mat][1], seed=len(by_mat)), filter_mode=wp.TextureFilterMode.LINEAR,
                                     address_mode=wp.TextureAddressMode.WRAP, normalized_coords=True, dtype=wp.uint8,
                                     num_channels=4, device=dev)
            r = float(SPEC[mat][2])
            d.repeat = wp.vec2f(r, r)
            by_mat[mat] = len(datas)
            datas.append(d)
        ids[i] = by_mat[mat]
    if datas:
        # the device array only holds the textures' handles: keep the Python objects (and so the GPU textures) alive
        # for the context's lifetime, as upstream does with its own list
        ctx._render_v2_textures = datas
        ctx.texture_data = wp.array(datas, dtype=TextureData, device=dev)
        ctx.shape_texture_ids = wp.array(ids.astype(np.int32), dtype=wp.int32, device=dev)
    apply_colors(model)
    _lights(ctx, dev)


def _lights(ctx, dev) -> None:
    """Soft key (jittered samples), fill and rim; each light's intensity is the length of its direction vector."""
    key = KEY_DIR / np.linalg.norm(KEY_DIR)
    a = np.cross(key, [0.0, 0.0, 1.0])
    a /= np.linalg.norm(a)
    b = np.cross(key, a)
    dirs, shadow = [], []
    for k in range(KEY_SAMPLES):
        ang = 2 * np.pi * k / KEY_SAMPLES
        r = np.tan(np.radians(KEY_CONE_DEG)) * (0.55 + 0.45 * (k % 2))
        d = key + r * (np.cos(ang) * a + np.sin(ang) * b)
        dirs.append(d / np.linalg.norm(d) * (KEY_INTENSITY / KEY_SAMPLES))
        shadow.append(True)
    for vec, inten in ((FILL_DIR, FILL_INTENSITY), (RIM_DIR, RIM_INTENSITY)):
        dirs.append(vec / np.linalg.norm(vec) * inten)
        shadow.append(False)
    n = len(dirs)
    ctx.lights_active = wp.array([True] * n, dtype=wp.bool, device=dev)
    ctx.lights_type = wp.array([1] * n, dtype=wp.int32, device=dev)
    ctx.lights_cast_shadow = wp.array(shadow, dtype=wp.bool, device=dev)
    ctx.lights_position = wp.array([wp.vec3f(0.0)] * n, dtype=wp.vec3f, device=dev)
    ctx.lights_orientation = wp.array([wp.vec3f(*map(float, d)) for d in dirs], dtype=wp.vec3f, device=dev)


_PATCHED = False


def _patch_lighting() -> None:
    """Newton's lighting function with a per-light intensity (|direction| for directional lights) and a larger shadow
    bias.  A copy of newton 1.5.0's ``create_compute_lighting_function``; the edits are marked."""
    global _PATCHED
    if _PATCHED:
        return
    from newton._src.core import MAXVAL
    from newton._src.sensors.warp_raytrace import lighting, raytrace

    def create_compute_lighting_function(config, state):
        raytrace_first_hit = raytrace.create_first_hit_function(config, state)

        @wp.func
        def compute_lighting(
            world_index: wp.int32, bvh_shapes_size: wp.int32, bvh_shapes_id: wp.uint64,
            bvh_shapes_group_roots: wp.array[wp.int32], bvh_particles_size: wp.int32, bvh_particles_id: wp.uint64,
            bvh_particles_group_roots: wp.array[wp.int32], shape_enabled: wp.array[wp.uint32],
            shape_types: wp.array[wp.int32], shape_sizes: wp.array[wp.vec3f],
            shape_transforms: wp.array[wp.transformf], shape_source_ptr: wp.array[wp.uint64],
            light_active: wp.bool, light_type: wp.int32, light_cast_shadow: wp.bool, light_position: wp.vec3f,
            light_orientation: wp.vec3f, particles_position: wp.array[wp.vec3f],
            particles_radius: wp.array[wp.float32], topology_particle_mask: wp.array[wp.bool],
            triangle_mesh_id: wp.uint64, triangle_mesh_group_roots: wp.array[wp.int32], normal: wp.vec3f,
            hit_point: wp.vec3f,
        ) -> wp.float32:
            if not light_active:
                return wp.float32(0.0)
            L = wp.vec3f(0.0, 0.0, 0.0)
            dist_to_light = wp.float32(MAXVAL)
            attenuation = wp.float32(1.0)
            intensity = wp.float32(1.0)
            if light_type == 1:
                L = wp.normalize(-light_orientation)
                intensity = wp.length(light_orientation)            # EDIT: per-light intensity
            else:
                to_light = light_position - hit_point
                dist_to_light = wp.length(to_light)
                L = wp.normalize(to_light)
                attenuation = 1.0 / (1.0 + 0.02 * dist_to_light * dist_to_light)
                if light_type == 0:
                    spot_dir = wp.normalize(light_orientation)
                    cos_theta = wp.dot(-L, spot_dir)
                    spot_factor = wp.min(1.0, wp.max(0.0, (cos_theta - 0.85) / (0.95 - 0.85)))
                    attenuation = attenuation * spot_factor
            ndotl = wp.max(0.0, wp.dot(normal, L))
            if ndotl == 0.0:
                return wp.float32(0.0)
            visible = wp.float32(1.0)
            if wp.static(config.enable_shadows) and light_cast_shadow:
                # EDIT: a bias along the normal AND toward the light (upstream: 1e-4 m along the normal only)
                shadow_origin = hit_point + normal * wp.static(SHADOW_BIAS_M) + L * wp.static(0.5 * SHADOW_BIAS_M)
                max_t = wp.max(wp.float32(1.0e-4), wp.float32(dist_to_light - 1.0e-3))
                if light_type == 1:
                    max_t = wp.float32(1.0e8)
                shadow_hit = raytrace_first_hit(
                    bvh_shapes_size, bvh_shapes_id, bvh_shapes_group_roots, bvh_particles_size, bvh_particles_id,
                    bvh_particles_group_roots, world_index, shape_enabled, shape_types, shape_sizes,
                    shape_transforms, shape_source_ptr, particles_position, particles_radius,
                    topology_particle_mask, triangle_mesh_id, triangle_mesh_group_roots, shadow_origin, L, max_t)
                if shadow_hit:
                    visible = wp.static(SHADOW_MIN_VISIBILITY)
            return ndotl * attenuation * visible * intensity

        return compute_lighting

    lighting.create_compute_lighting_function = create_compute_lighting_function
    _PATCHED = True
