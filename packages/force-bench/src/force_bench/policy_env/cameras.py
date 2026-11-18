"""Policy cameras: a third-person stereo pair (LEFT, RIGHT) and a wrist camera, rendered on request.

Rendering uses Newton's tiled camera as ``recorder.RolloutRecorder`` does, but separately: its own resolution and
fields of view, no supersampling, and only when a policy asks.  The calibration follows the matrix scene-observation
interface (``matrix.scene-observation.v1``): ``K`` is pinhole in pixels (derived from the renderer's own rays, so the
FOV convention cannot be misread), ``T_base_camera`` is camera-to-robot-root (``fr3/fr3_link0``) in OpenCV axes
(+x right, +y down, +z forward).

LEFT/RIGHT share the task's own scene camera (the one its rollout video uses), split into a parallel pair with
``baseline_m`` between them.  The wrist camera rides on the hand, behind and beside the fingers, looking at the
fingertip centre.  The sim model collapses the URDF's fixed links (link8, hand, hand_tcp) into ``fr3_link7``, so the
camera is placed on link7 through the link7 -> hand transform of the uncollapsed IK model.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from force_bench.recorder import BG_BOTTOM, BG_TOP, _resolve_frame_kernel, lookat_transform

GL_TO_CV = np.diag([1.0, -1.0, -1.0])
WRIST_IN_HAND = np.array([0.075, 0.0, 0.065])      # hand frame: z toward the fingers, y the finger travel; the camera
#                                                     sits beside the hand, across the fingers, so neither finger hides
#                                                     the part between them
WRIST_LOOK_IN_HAND = np.array([0.0, 0.0, 0.105])   # the pads' centre


def _mat_of(tf) -> np.ndarray:
    """4x4 of a warp transform (p, q xyzw)."""
    p = np.array([tf[0], tf[1], tf[2]], float)
    x, y, z, w = (float(tf[i]) for i in range(3, 7))
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, p
    return M


class PolicyCameras:
    """LEFT/RIGHT on the scene camera, then the robot's own cameras: the rig's ``camera_mounts`` (an embodiment's
    [(name, body label, T_body_camera in GL axes, vertical FOV deg)], e.g. the G1's head camera), or by default the
    FR3's wrist camera."""

    def __init__(self, rig, *, scene_pos, scene_target, width=640, height=480, scene_fov_deg=45.0,
                 wrist_fov_deg=80.0, baseline_m=0.12):
        from newton.sensors import SensorTiledCamera
        self.rig, self.model = rig, rig.model
        self.width, self.height = int(width), int(height)
        mounts = getattr(rig, "camera_mounts", None)
        if mounts is None:
            link7 = rig.body_index("fr3/fr3_link7")
            l7_hand = (np.linalg.inv(_mat_of(rig.ik.body_pose("fr3/fr3_link7")))
                       @ _mat_of(rig.ik.body_pose("fr3/fr3_hand")))
            self.mounts = [("wrist", link7, l7_hand @ _hand_wrist_cam(), wrist_fov_deg)]
        else:
            self.mounts = [(n, rig.body_index(b), np.asarray(T, float), fov) for n, b, T, fov in mounts]
        self.NAMES = ("left", "right", *[m[0] for m in self.mounts])
        n = len(self.NAMES)
        self.cam = SensorTiledCamera(model=self.model)
        self.cam.default_render_config.enable_shadows = True
        self.cam.utils.create_default_light(enable_shadows=True)
        fovs = np.radians([scene_fov_deg, scene_fov_deg, *[m[3] for m in self.mounts]])
        self.rays = self.cam.utils.compute_camera_rays_pinhole(self.width, self.height, camera_fovs=fovs)
        self.color = self.cam.utils.create_color_image_output(self.width, self.height, n)
        self.sidx = self.cam.utils.create_shape_index_image_output(self.width, self.height, n)
        self.out = wp.zeros((self.height, self.width, 3), dtype=wp.uint8)
        self.K = [self._k_from_rays(i) for i in range(n)]
        M = _mat_of(lookat_transform(scene_pos, scene_target))      # camera-to-world, GL axes
        right = M[:3, 0]
        self.scene_world = []
        for s in (-0.5, 0.5):
            Mi = M.copy()
            Mi[:3, 3] = M[:3, 3] + s * baseline_m * right
            self.scene_world.append(Mi)
        self.root = _mat_of(rig.robot_base)                          # the robot's root in the world
        self.root_inv = np.linalg.inv(self.root)
        self._cache = None

    def _k_from_rays(self, i) -> list:
        """K from the renderer's own per-pixel rays ((camera, h, w, [origin, direction], 3), GL camera axes, rays
        through pixel centres): no reading of the FOV convention (the configured angle is the VERTICAL one)."""
        d = self.rays.numpy()[i][..., 1, :]
        tx, ty = d[..., 0] / -d[..., 2], d[..., 1] / -d[..., 2]      # tangents at depth 1 (GL looks along -z)
        fx = (self.width - 1) / (tx[0, -1] - tx[0, 0])
        fy = (self.height - 1) / (ty[0, 0] - ty[-1, 0])               # GL +y up, OpenCV +v down
        cx, cy = -tx[0, 0] * fx, ty[0, 0] * fy
        return [[float(fx), 0.0, float(cx)], [0.0, float(fy), float(cy)], [0.0, 0.0, 1.0]]

    def _robot_worlds(self, bq) -> list:
        return [_mat_of(bq[b]) @ T for _, b, T, _ in self.mounts]

    def capture(self, t_s: float) -> dict:
        """The three views and their calibration at the current state (cached for the frame: a policy request and
        the episode's camera videos share one render)."""
        if self._cache is not None and self._cache[0] == t_s:
            return self._cache[1]
        out = self._render(t_s)
        self._cache = (t_s, out)
        return out

    def _render(self, t_s: float) -> dict:
        state = self.rig.s0
        bq = state.body_q.numpy()
        worlds = [*self.scene_world, *self._robot_worlds(bq)]
        tfs = [wp.transformf(wp.vec3f(*M[:3, 3]), wp.quatf(*_quat_xyzw(M[:3, :3]))) for M in worlds]
        cam_tf = wp.array([[tf] for tf in tfs], dtype=wp.transformf)
        self.model.bvh_refit_shapes(state)
        self.cam.update(state, cam_tf, self.rays, color_image=self.color, shape_index_image=self.sidx,
                        clear_data=type(self.cam).GRAY_CLEAR_DATA)
        images, calib = {}, {}
        for ci, name in enumerate(self.NAMES):
            wp.launch(_resolve_frame_kernel, dim=(self.height, self.width),
                      inputs=[self.color, self.sidx, self.out, 1, self.height, wp.vec3f(*BG_TOP),
                              wp.vec3f(*BG_BOTTOM), ci, 0])
            images[name] = self.out.numpy().copy()
            T = self.root_inv @ worlds[ci]
            T[:3, :3] = T[:3, :3] @ GL_TO_CV
            calib[name] = {"frame": "robot_root", "convention": "opencv", "K": self.K[ci], "T_base_camera":
                           T.tolist(), "capture_time_s": float(t_s), "resolution": [self.width, self.height]}
        return {"images": images, "calibration": calib}


def _hand_wrist_cam() -> np.ndarray:
    """The FR3 wrist camera in the hand frame (GL axes): beside the hand, looking at the pad centre."""
    p, tgt = WRIST_IN_HAND, WRIST_LOOK_IN_HAND
    fwd = (tgt - p) / np.linalg.norm(tgt - p)
    x = np.cross(fwd, np.array([1.0, 0.0, 0.0]))
    x /= np.linalg.norm(x)
    y = np.cross(x, fwd)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, -fwd, p
    return M


def _quat_xyzw(R) -> tuple:
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return (x, y, z, w)
