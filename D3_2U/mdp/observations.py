from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_conjugate, quat_mul, quat_rotate

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector from the world frame to the body frame"""
    return quat_rotate(quat_conjugate(q), v)


def get_boom_states(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Normalized state of the master boom joints

    Returns a (batch, 8) tensor [ext_1..ext_4, vel_1..vel_4] with extension in [0, 1]
    and velocity in [-1, 1]
    """
    asset = env.scene[asset_cfg.name]
    names = asset.joint_names

    NUM_BOOMS = 4
    MAX_LEN = 3.7  # m
    MAX_VELOCITY = 0.05  # m/s

    exts = torch.zeros((env.num_envs, NUM_BOOMS), device=env.device)
    vels = torch.zeros_like(exts)

    # read raw positions & velocities
    for b in range(NUM_BOOMS):
        idx = names.index(f"d3_boom_{b + 1}_joint")
        exts[:, b] = asset.data.joint_pos[:, idx]
        vels[:, b] = asset.data.joint_vel[:, idx]

    # normalize & clamp
    norm_exts = torch.clamp(exts / MAX_LEN, 0.0, 1.0)
    norm_vels = torch.clamp(vels / MAX_VELOCITY, -1.0, 1.0)
    return torch.cat([norm_exts, norm_vels], dim=-1).to(dtype=torch.float32)


@torch.no_grad()
def get_lvlh_rate_error_body(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    omega_scale: float = 0.01,  # rad/s
    clip: float = 5.0,  # clamp after scaling to avoid outliers
) -> torch.Tensor:
    """Angular-rate error between the body and the LVLH frame, expressed in the body frame

    Returns an (N, 3) tensor: (omega_body_b - omega_lvlh_b) / omega_scale
    """
    asset: Articulation = env.scene[asset_cfg.name]
    q_wb = asset.data.root_quat_w
    omega_body_b = asset.data.root_ang_vel_b

    # LVLH rate in the world frame, from the command manager
    omega_lvlh_w = env.command_manager.get_command("target_pose")["angular_velocity"]
    omega_lvlh_b = quat_rotate_inverse(q_wb, omega_lvlh_w)

    omega_err_b = omega_body_b - omega_lvlh_b
    return torch.clamp(omega_err_b / omega_scale, -clip, clip).to(torch.float32)


def get_body_linear_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Linear velocity in the body frame, clamped to [-1, 1]"""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.clamp(asset.data.root_lin_vel_b, -1.0, 1.0).to(dtype=torch.float32)


def get_position_error_body(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position error (target - current) expressed in the body frame

    Lets the agent correct the orbital drift (in-track)
    """
    asset: Articulation = env.scene[asset_cfg.name]

    pos_target = env.command_manager.get_command("target_pose")["position"]
    error_w = pos_target - asset.data.root_pos_w

    error_b = quat_rotate_inverse(asset.data.root_quat_w, error_w)

    # The drift can span meters to kilometers: scale so that ~10 m maps to 1.0,
    # and clip so that a 100 km error does not blow up the network
    scale = 0.1
    clip_val = 5.0
    return torch.clamp(error_b * scale, -clip_val, clip_val).to(dtype=torch.float32)


def get_attitude_error_body(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 5.0,
) -> torch.Tensor:
    """Attitude error in the body frame: vector part (x, y, z) of the error quaternion"""
    asset: Articulation = env.scene[asset_cfg.name]
    q_body = asset.data.root_quat_w

    # The target pose command holds the LVLH orientation in the world frame
    q_target = env.command_manager.get_command("target_pose")["orientation"]

    # How much the body must rotate, about its own axes, to match the target
    q_err = quat_mul(quat_conjugate(q_body), q_target)

    # Canonicalize (q == -q) so that w >= 0 and the observation has no discontinuities
    q_err = torch.where(q_err[:, 0:1] < 0, -q_err, q_err)

    # For small errors the vector part is ~half the error angle in radians; the scale
    # makes a 10 deg error (~0.17 rad) visible to the network
    return (q_err[:, 1:] * scale).to(dtype=torch.float32)