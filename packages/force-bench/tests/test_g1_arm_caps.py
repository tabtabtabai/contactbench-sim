"""Every G1 arm joint has the G1's own torque limit (the exported model once carried none: unlimited)."""
import pytest

pytest.importorskip("newton")
from force_bench.g1.robot import ARM_EFFORT_NM, arm_cap  # noqa: E402

ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")


@pytest.mark.parametrize("side", ["left", "right"])
def test_every_arm_joint_has_its_urdf_limit(side):
    for j in ARM:
        cap = arm_cap(f"{side}_{j}_joint")
        assert cap is not None and 0.0 < cap < float("inf"), j
    assert arm_cap(f"{side}_wrist_pitch_joint") == 5.0 and arm_cap(f"{side}_wrist_yaw_joint") == 5.0
    assert arm_cap(f"{side}_wrist_roll_joint") == 25.0 and arm_cap(f"{side}_elbow_joint") == 25.0


def test_fingers_and_legs_are_not_arm_joints():
    assert arm_cap("right_dex1_finger_joint_1") is None and arm_cap("left_hip_pitch_joint") is None
    assert len(ARM_EFFORT_NM) == 7
