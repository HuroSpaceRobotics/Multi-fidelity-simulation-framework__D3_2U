from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_conjugate, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def pose_tracking_reward_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std_pos: float = 1.0,
    std_orient: float = 1.0,
    pos_weight: float = 1.0,
    orient_weight: float = 1.0,
) -> torch.Tensor:
    """Gaussian-kernel pose tracking reward, bounded in [0, pos_weight + orient_weight]

    exp(-error^2 / std^2) behaves like an L2 error near the target and is smoother and
    easier to learn from than a log reward
    """
    cmd = env.command_manager.get_command("target_pose")
    asset: Articulation = env.scene[asset_cfg.name]

    pos_error = torch.norm(cmd["position"] - asset.data.root_pos_w, dim=-1)

    # Orientation error: 2 * acos(|q . q_t|)
    quat_diff = quat_mul(asset.data.root_quat_w, quat_conjugate(cmd["orientation"]))
    rot_error = 2.0 * torch.atan2(torch.norm(quat_diff[:, 1:], dim=-1), torch.abs(quat_diff[:, 0]))

    pos_reward = torch.exp(-torch.square(pos_error) / (std_pos**2))
    rot_reward = torch.exp(-torch.square(rot_error) / (std_orient**2))

    return pos_weight * pos_reward + orient_weight * rot_reward


def linear_velocity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Quadratic penalty on the linear velocity"""
    v = env.scene[asset_cfg.name].data.root_lin_vel_w
    return -5.0 * torch.square(v.norm(dim=-1))


def angular_velocity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Quadratic penalty on the angular velocity relative to the LVLH frame

    Drives the spacecraft to stop tumbling, using the inertia of the booms
    """
    asset: Articulation = env.scene[asset_cfg.name]
    omega_body_w = asset.data.root_ang_vel_w

    # The target is to rotate with the LVLH frame (N, 3)
    omega_lvlh_w = env.command_manager.get_command("target_pose")["angular_velocity"]

    omega_error = omega_body_w - omega_lvlh_w
    return -torch.sum(torch.square(omega_error), dim=-1) * 1000


def action_rate_l2_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalizes abrupt changes in the boom command, reducing jitter: |a_t - a_{t-1}|^2"""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=-1)


def boom_velocity_physical_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalizes the actual velocity of the boom joints"""
    asset: Articulation = env.scene[asset_cfg.name]
    action_term = env.action_manager.get_term("boom_extension")
    joint_vel = asset.data.joint_vel[:, action_term.joint_indices]
    return torch.sum(torch.square(joint_vel), dim=-1)