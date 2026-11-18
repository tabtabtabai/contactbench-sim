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
WRIST_IN_HAND = np.array([0.0, -0.055, 0.020])     # hand frame: z toward the fingers, y along the finger travel
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
    NAMES = ("left", "right", "wrist")

    def __init__(self, rig, *, scene_pos, scene_target, width=640, height=480, scene_fov_deg=45.0,
                 wrist_fov_deg=80.0, baseline_m=0.12):
        from newton.sensors import SensorTiledCamera
        self.rig, self.model = rig, rig.model
        self.width, self.height = int(width), int(height)
        self.cam = SensorTiledCamera(model=self.model)
        self.cam.default_render_config.enable_shadows = True
        self.cam.utils.create_default_light(enable_shadows=True)
        fovs = np.radians([scene_fov_deg, scene_fov_deg, wrist_fov_deg])
        self.rays = self.cam.utils.compute_camera_rays_pinhole(self.width, self.height, camera_fovs=fovs)
        self.color = self.cam.utils.create_color_image_output(self.width, self.height, 3)
        self.sidx = self.cam.utils.create_shape_index_image_output(self.width, self.height, 3)
        self.out = wp.zeros((self.height, self.width, 3), dtype=wp.uint8)
        self.K = [self._k_from_rays(i) for i in range(3)]
        # the stereo pair: the scene camera's pose, shifted -+ baseline/2 along its own right axis
        M = _mat_of(lookat_transform(scene_pos, scene_target))      # camera-to-world, GL axes
        right = M[:3, 0]
        self.scene_world = []
        for s in (-0.5, 0.5):
            Mi = M.copy()
            Mi[:3, 3] = M[:3, 3] + s * baseline_m * right
            self.scene_world.append(Mi)
        self.link7 = rig.body_index("fr3/fr3_link7")
        self.l7_hand = (np.linalg.inv(_mat_of(rig.ik.body_pose("fr3/fr3_link7")))
                        @ _mat_of(rig.ik.body_pose("fr3/fr3_hand")))
        self.root = _mat_of(rig.robot_base)                          # fr3_link0 in the world
        self.root_inv = np.linalg.inv(self.root)

    def _k_from_rays(self, i) -> list:
        r = self.rays.numpy()
        r = r.reshape(-1, self.height, self.width, 3)[i] if r.ndim != 4 or r.shape[0] != 3 else r[i]
        d = r / -r[..., 2:3]                                         # GL: forward is -z; normalise to z = 1 depth
        fx = (self.width - 1) / (d[0, -1, 0] - d[0, 0, 0])
        fy = (self.height - 1) / abs(d[-1, 0, 1] - d[0, 0, 1])
        cx = -d[0, 0, 0] * fx
        cy = d[0, 0, 1] * fy                                         # GL +y up -> OpenCV +y down
        return [[float(fx), 0.0, float(cx)], [0.0, float(fy), float(cy)], [0.0, 0.0, 1.0]]

    def _wrist_world(self, bq) -> np.ndarray:
        H = _mat_of(bq[self.link7]) @ self.l7_hand
        p = H[:3, :3] @ WRIST_IN_HAND + H[:3, 3]
        tgt = H[:3, :3] @ WRIST_LOOK_IN_HAND + H[:3, 3]
        fwd = (tgt - p) / np.linalg.norm(tgt - p)
        up_hint = -H[:3, 1]                                          # image up = away from the camera's side
        x = np.cross(fwd, up_hint)
        x /= np.linalg.norm(x)
        y = np.cross(x, fwd)
        M = np.eye(4)
        M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, -fwd, p      # GL: looks along -z
        return M

    def capture(self, t_s: float) -> dict:
        state = self.rig.s0
        bq = state.body_q.numpy()
        worlds = [*self.scene_world, self._wrist_world(bq)]
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
