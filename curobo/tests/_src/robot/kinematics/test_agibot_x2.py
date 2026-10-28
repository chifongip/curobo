# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

"""Validate the packaged Agibot X2 dual-arm model against its URDF."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pytest
import torch
import trimesh
from yourdfpy import URDF

from curobo._src.robot.builder.debugger_robot import RobotDebugger
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.state.state_joint import JointState
from curobo._src.util_file import load_yaml
from curobo.content import get_assets_path, get_robot_configs_path
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.types import GoalToolPose

CONFIG_PATH = Path(get_robot_configs_path()) / "agibot_x2.yml"
ARM_JOINTS = [
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in (
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_yaw",
        "wrist_pitch",
        "wrist_roll",
    )
]
TOOL_FRAMES = ["left_wrist_roll_link", "right_wrist_roll_link"]
WAIST_COLLISION_LINKS = {"waist_yaw_link", "waist_pitch_link"}


@pytest.fixture(scope="module")
def x2_data() -> dict[str, Any]:
    """Load the installed robot configuration."""
    return load_yaml(str(CONFIG_PATH))["robot_cfg"]["kinematics"]


@pytest.fixture(scope="module")
def x2_debugger() -> RobotDebugger:
    """Load the X2 model and its self-collision checker on CUDA."""
    return RobotDebugger(str(CONFIG_PATH))


def test_packaged_assets_and_collision_geometry(x2_data: dict[str, Any]) -> None:
    """Resolve every referenced mesh and cover every URDF collision link."""
    assets = Path(get_assets_path())
    assert not Path(x2_data["urdf_path"]).is_absolute()
    assert not Path(x2_data["asset_root_path"]).is_absolute()
    urdf_path = assets / x2_data["urdf_path"]
    root = ElementTree.parse(urdf_path).getroot()
    for mesh in root.findall(".//mesh"):
        assert (urdf_path.parent / mesh.attrib["filename"]).is_file()
    assert (urdf_path.parent / "LICENSE").is_file()

    collision_links = {
        link.attrib["name"] for link in root.findall("link") if link.find("collision") is not None
    }
    spheres = x2_data["collision_spheres"]
    assert set(x2_data["collision_link_names"]) == collision_links | WAIST_COLLISION_LINKS
    assert set(spheres) == set(x2_data["collision_link_names"])
    assert sum(map(len, spheres.values())) < 1000
    for link_spheres in spheres.values():
        values = torch.tensor(
            [sphere["center"] + [sphere["radius"]] for sphere in link_spheres],
            dtype=torch.float32,
        )
        assert values.shape[0] > 0
        assert torch.isfinite(values).all()
        assert (values[:, 3] > 0).all()


def test_collision_spheres_cover_source_surfaces(x2_data: dict[str, Any]) -> None:
    """Cover source surfaces and limit outward extent along sampled directions."""
    root = ElementTree.parse(Path(get_assets_path()) / x2_data["urdf_path"]).getroot()
    asset_root = Path(get_assets_path()) / x2_data["asset_root_path"]
    directions = torch.cartesian_prod(
        *[torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32, device="cuda")] * 3
    )
    directions = directions[directions.any(dim=1)]
    directions = directions / torch.linalg.vector_norm(directions, dim=1, keepdim=True)
    for link in root.findall("link"):
        link_name = link.attrib["name"]
        collision = link.find("collision")
        if collision is None and link_name in WAIST_COLLISION_LINKS:
            collision = link.find("visual")
        if collision is None:
            continue
        geometry = collision.find("geometry/mesh")
        mesh = trimesh.load_mesh(asset_root / geometry.attrib["filename"])
        mesh.apply_scale([float(x) for x in geometry.attrib.get("scale", "1 1 1").split()])
        # This URDF expresses collision meshes directly in the owning link frame.
        origin = collision.find("origin")
        assert origin is None or all(
            float(value) == 0.0
            for key in ("xyz", "rpy")
            for value in origin.attrib.get(key, "0 0 0").split()
        )
        samples, _ = trimesh.sample.sample_surface(mesh, 10000, seed=2026)
        points = torch.cat(
            [
                torch.tensor(mesh.vertices.copy(), dtype=torch.float32, device="cuda"),
                torch.tensor(samples, dtype=torch.float32, device="cuda"),
            ]
        )
        spheres = torch.tensor(
            [s["center"] + [s["radius"]] for s in x2_data["collision_spheres"][link_name]],
            dtype=torch.float32,
            device="cuda",
        )
        gaps = torch.cat(
            [
                (
                    torch.cdist(chunk, spheres[:, :3], compute_mode="donot_use_mm_for_euclid_dist")
                    - spheres[:, 3]
                ).amin(dim=1)
                for chunk in points.split(4096)
            ]
        ).clamp_min(0.0)
        padding = x2_data["collision_sphere_buffer"]
        if isinstance(padding, dict):
            padding = padding[link_name]
        assert 0.0 < padding <= 0.003, link_name
        assert gaps.max() < padding, link_name
        mesh_extent = (points @ directions.T).amax(dim=0)
        sphere_extent = (
            spheres[:, :3] @ directions.T + spheres[:, 3:4] + padding
        ).amax(dim=0)
        assert (sphere_extent - mesh_extent).amax() < 0.025, link_name


def test_joint_order_locks_and_limits(
    x2_data: dict[str, Any],
    x2_debugger: RobotDebugger,
) -> None:
    """Keep both arms active and preserve URDF limits and all body locks."""
    model = x2_debugger.robot_model
    assert model.dof == 14
    assert model.joint_names == ARM_JOINTS
    assert model.tool_frames == TOOL_FRAMES
    root = ElementTree.parse(Path(get_assets_path()) / x2_data["urdf_path"]).getroot()
    joints = {j.attrib["name"]: j for j in root.findall("joint")}
    movable = {name for name, joint in joints.items() if joint.attrib["type"] != "fixed"}
    assert set(x2_data["lock_joints"]) == movable - set(ARM_JOINTS)
    assert len(x2_data["lock_joints"]) == 17
    assert all(value == 0.0 for value in x2_data["lock_joints"].values())
    limits = model.get_joint_limits()
    for index, name in enumerate(ARM_JOINTS):
        limit = joints[name].find("limit").attrib
        expected = torch.tensor(
            [float(limit["lower"]), float(limit["upper"])],
            device=limits.position.device,
            dtype=torch.float32,
        )
        torch.testing.assert_close(limits.position[:, index], expected)
        assert limits.velocity[1, index].item() == pytest.approx(float(limit["velocity"]))
    assert torch.all(model.default_joint_position >= limits.position[0])
    assert torch.all(model.default_joint_position <= limits.position[1])


@pytest.mark.parametrize("moving_side", ["left", "right"])
def test_forward_kinematics_matches_urdf(
    moving_side: str,
    x2_data: dict[str, Any],
) -> None:
    """Match independent URDF transforms for moving arms and the locked body."""
    frames = TOOL_FRAMES + [
        "pelvis",
        "torso_link",
        "head_pitch_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ]
    model = Kinematics(KinematicsCfg.from_robot_yaml_file("agibot_x2.yml", frames))
    q = model.default_joint_position.view(1, -1).clone()
    q[0, ARM_JOINTS.index(f"{moving_side}_shoulder_pitch_joint")] -= 0.15
    q[0, ARM_JOINTS.index(f"{moving_side}_elbow_joint")] -= 0.2
    state = model.compute_kinematics(JointState.from_position(q, joint_names=model.joint_names))
    reference = URDF.load(
        str(Path(get_assets_path()) / x2_data["urdf_path"]),
        load_meshes=False,
    )
    reference.update_cfg(
        {
            **x2_data["lock_joints"],
            **dict(zip(model.joint_names, q[0].cpu().tolist())),
        }
    )
    for frame in frames:
        expected = torch.as_tensor(
            reference.get_transform(frame, x2_data["base_link"]).copy(),
            device=q.device,
            dtype=torch.float32,
        )
        actual = state.tool_poses[frame].get_matrix()[0]
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)


def test_default_posture_is_collision_free(x2_debugger: RobotDebugger) -> None:
    """Keep the posture used for planner warmup clear of self-collisions."""
    result = x2_debugger.check_default_joint_configuration_collision()
    assert not result["has_collision"], result["distances"]


def test_crossed_wrists_are_in_collision(x2_debugger: RobotDebugger) -> None:
    """Detect contact between the two arms when both wrists meet at the center."""
    q = x2_debugger.robot_model.default_joint_position.clone()
    q[ARM_JOINTS.index("left_shoulder_yaw_joint")] = -1.57
    q[ARM_JOINTS.index("right_shoulder_yaw_joint")] = 1.57
    result = x2_debugger.check_collision_at_config(q)
    assert result["has_collision"]
    assert ("left_wrist_roll_link", "right_wrist_roll_link") in result["colliding_pairs"]


def test_arm_to_arm_and_body_collision_pairs(x2_debugger: RobotDebugger) -> None:
    """Keep opposite arms and nonadjacent body links in the collision matrix."""
    cfg = x2_debugger.robot_config
    params = cfg.kinematics_config
    pairs = params.link_sphere_idx_map[cfg.self_collision_config.collision_pairs.int()]
    checked = {frozenset(pair) for pair in torch.unique(pairs, dim=0).cpu().tolist()}
    link_indices = params.link_name_to_idx_map
    for left in ARM_JOINTS[:7]:
        left_index = link_indices[left.removesuffix("_joint") + "_link"]
        for right in ARM_JOINTS[7:]:
            right_index = link_indices[right.removesuffix("_joint") + "_link"]
            assert frozenset((left_index, right_index)) in checked
    for side in ("left", "right"):
        for arm in ("elbow", "wrist_yaw", "wrist_roll"):
            for body in (
                "pelvis",
                "torso_link",
                "head_pitch_link",
                "left_hip_roll_link",
                "right_hip_roll_link",
                "waist_yaw_link",
                "waist_pitch_link",
            ):
                assert (
                    frozenset((link_indices[f"{side}_{arm}_link"], link_indices[body])) in checked
                )


def test_dual_arm_inverse_kinematics(
    x2_debugger: RobotDebugger,
    x2_data: dict[str, Any],
) -> None:
    """Solve simultaneous wrist goals with the packaged collision model enabled."""
    model = x2_debugger.robot_model
    q = model.default_joint_position.view(1, -1).clone()
    q[:, [0, 7]] -= 0.1
    q[:, [3, 10]] += 0.1
    # Kinematics reuses output buffers; preserve goals before evaluating the solution.
    target = model.compute_kinematics(
        JointState.from_position(q, joint_names=model.joint_names)
    ).tool_poses.clone()
    solver = InverseKinematics(
        InverseKinematicsCfg.create(
            robot="agibot_x2.yml",
            num_seeds=8,
            use_cuda_graph=False,
        )
    )
    result = solver.solve_pose(
        GoalToolPose.from_poses({frame: target[frame] for frame in TOOL_FRAMES}, num_goalset=1),
        current_state=model.default_joint_state.unsqueeze(0),
    )
    assert result.success.all()
    assert result.js_solution.position.shape[-1] == 31
    for name, value in x2_data["lock_joints"].items():
        index = result.js_solution.joint_names.index(name)
        assert torch.all(result.js_solution.position[..., index] == value)
    achieved = model.compute_kinematics(result.js_solution.reorder(model.joint_names))
    for frame in TOOL_FRAMES:
        actual = achieved.tool_poses[frame]
        expected = target[frame]
        assert torch.linalg.vector_norm(actual.position - expected.position) < 0.005
        alignment = (actual.quaternion * expected.quaternion).sum(dim=-1).abs()
        assert torch.all(2.0 * torch.acos(alignment.clamp(max=1.0)) < 0.05)
