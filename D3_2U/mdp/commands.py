from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import compute_pose_error
from scipy.spatial.transform import Rotation

from .events import DragGlobal, OrbitDataManager, create_raycaster_marker
from .observations import get_boom_states

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class OrbitDataCommand(CommandTerm):
    """Exposes the pre-computed orbit data (position, LVLH attitude, velocity, atmosphere, ...)."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.orbit_manager = OrbitDataManager(
            orbit_data_path=cfg.orbit_data_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            sim_dt=env.cfg.sim.dt,
            episode_length_s=env.cfg.episode_length_s,
            decimation=env.cfg.decimation,
            time_scale=env.cfg.time_scale,
        )
        self.orbit_manager.reset()

    @property
    def command(self) -> dict[str, torch.Tensor]:
        n = self._env.num_envs
        get = self.orbit_manager.get_data
        return {
            "position": get("position", num_envs=n),
            "orientation": get("orientation", num_envs=n),
            "velocity": get("velocity", num_envs=n),
            "atm_ang_velocity": get("atm_ang_velocity", num_envs=n),
            "air_density": get("air_density", num_envs=n),
            "time": self.orbit_manager.get_current_time(),
            "j2_perturbation_acc": get("j2_perturbation_acc", num_envs=n),
        }

    def reset_orbit_data(self, env_ids: torch.Tensor, reset_time: float | None = None):
        """Jump the orbit to ``reset_time``. Only done when all environments reset together."""
        if len(env_ids) != self._env.num_envs:
            return
        self.orbit_manager.reset(start_time=reset_time)

        # Keep the target pose consistent with the new orbit state
        target_pose = self._env.command_manager._terms.get("target_pose")
        if target_pose is not None:
            target_pose._update_command()
        else:
            print("Warning: target_pose command term not found in command manager")

    def _update_command(self):
        self.orbit_manager.step()

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _update_metrics(self):
        pass

    def close(self):
        self.orbit_manager.close()
        super().close()


def _create_frame_marker(prim_path: str, scale: float) -> VisualizationMarkers:
    """Coordinate-frame marker used to visualize poses."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(scale, scale, scale),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)


def quaternion_to_euler_deg(quat: torch.Tensor) -> torch.Tensor:
    """Quaternion (w, x, y, z) to extrinsic xyz Euler angles [deg]; degenerate quaternions map to identity."""
    quat_np = quat.detach().cpu().numpy()
    norms = np.linalg.norm(quat_np, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        normed = quat_np / norms
    quat_np = np.where(norms > 1e-6, normed, np.array([1.0, 0.0, 0.0, 0.0]))
    euler_rad = Rotation.from_quat(quat_np[:, [1, 2, 3, 0]]).as_euler("xyz", degrees=False)
    return torch.from_numpy(euler_rad).to(quat.device).float() * 180.0 / torch.pi


class TargetPoseCommand(CommandTerm):
    """Target pose: the LVLH frame, taken from the orbit data command.

    With ``cfg.log_metrics`` it also computes tracking metrics (pose error, boom state,
    drag, ...) every step and logs those of environment 0 to Weights & Biases.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._position_command = torch.zeros((env.num_envs, 3), device=env.device)
        self._orientation_command = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
        self._angular_velocity_command = torch.zeros((env.num_envs, 3), device=env.device)

        self._orbit_data_term_name = cfg.orbit_data_term_name
        # The orbit command is registered after this term, so it is linked lazily
        self._orbit_command_initialized = False

        self._target_marker = _create_frame_marker("/Visuals/TargetPoseMarker", scale=0.2)
        self._body_marker = _create_frame_marker("/Visuals/BodyPoseMarker", scale=0.15)
        env._raycaster_marker = create_raycaster_marker()

        self._episode_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._wandb_step = 0
        self._wandb = self._init_wandb() if cfg.log_metrics else None

    def _init_wandb(self):
        try:
            import wandb

            if wandb.run is None:
                wandb.init(
                    project=self.cfg.wandb_project,
                    name="inference",
                    config={
                        "num_envs": self.num_envs,
                        "episode_length": self._env.max_episode_length,
                        "sim_dt": self._env.step_dt,
                    },
                )
            return wandb
        except Exception as e:
            print(f"W&B logging disabled: {e}")
            return None

    @property
    def command(self) -> dict[str, torch.Tensor]:
        if not self._orbit_command_initialized:
            self._try_init_orbit_command()

        return {
            "position": self._position_command,
            "orientation": self._orientation_command,
            "angular_velocity": self._angular_velocity_command,
        }

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        self._episode_steps[env_ids] = 0
        return super().reset(env_ids)

    def _try_init_orbit_command(self):
        try:
            manager = getattr(self._env, "command_manager", None)
            if manager is not None and self._orbit_data_term_name in manager._terms:
                self._orbit_command_initialized = True
                self._update_commands()
        except Exception as e:
            print(f"Warning: Unable to initialize orbit command: {e}")

    def _get_orbit_command(self):
        manager = getattr(self._env, "command_manager", None)
        if manager is None:
            return None
        try:
            return manager.get_command(self._orbit_data_term_name)
        except KeyError as e:
            print(f"Warning: Failed to get orbit command: {e}")
            return None

    def _update_command(self):
        if not self._orbit_command_initialized:
            self._try_init_orbit_command()
            return
        self._update_commands()
        self._update_body_marker()
        self._update_target_marker()

    def _update_commands(self):
        orbit_command = self._get_orbit_command()
        if not orbit_command:
            print("No orbit command found in TargetPoseCommand")
            return

        self._orientation_command = orbit_command["orientation"].clone()
        self._position_command = self._env.scene.env_origins.clone()

        # Angular velocity of the LVLH frame in the world frame: omega = (r x v) / |r|^2
        r_w = orbit_command["position"].to(dtype=torch.float32)
        v_w = orbit_command["velocity"].to(dtype=torch.float32)
        h_w = torch.cross(r_w, v_w, dim=-1)
        self._angular_velocity_command = h_w / torch.sum(r_w**2, dim=-1, keepdim=True)

        if self.cfg.use_orbit_position:
            self._position_command = orbit_command["position"]

    def _update_target_marker(self):
        marker_indices = torch.zeros(self._env.num_envs, dtype=torch.long, device=self._env.device)
        self._target_marker.visualize(self._position_command, self._orientation_command, marker_indices=marker_indices)

    def _update_body_marker(self):
        if not self.cfg.visualize_target:
            return
        robot = self._env.scene["robot"]
        marker_indices = torch.zeros(self._env.num_envs, dtype=torch.long, device=self._env.device)
        self._body_marker.visualize(robot.data.root_pos_w, robot.data.root_quat_w, marker_indices=marker_indices)

    def _resample_command(self, env_ids: torch.Tensor):
        pass

    def _store_xyz(self, prefix: str, values: torch.Tensor, suffix: str = ""):
        for i, axis in enumerate("xyz"):
            self.metrics[f"{prefix}{axis}{suffix}"] = values[:, i]

    def _update_metrics(self):
        if not self.cfg.log_metrics:
            return

        robot = self._env.scene["robot"]
        m = self.metrics

        # Pose error w.r.t. the target
        pos = robot.data.root_pos_w[:, :3]
        quat = robot.data.root_quat_w
        target_pos = self._position_command
        target_quat = self._orientation_command
        pos_error, orient_error = compute_pose_error(pos, quat, target_pos, target_quat)
        m["position_error"] = torch.norm(pos_error, dim=-1)
        m["orientation_error"] = torch.norm(orient_error, dim=-1)
        m["orientation_error_deg"] = torch.rad2deg(m["orientation_error"])
        self._store_xyz("position/current_", pos)
        self._store_xyz("position/target_", target_pos)
        self._store_xyz("position/error_", pos_error)

        # Position in the LVLH frame: along-track (x), cross-track (y), radial (z)
        drift = (pos - self._env.scene.env_origins[:, :3]).float()
        lvlh_pos = math_utils.quat_apply(math_utils.quat_conjugate(target_quat.float()), drift)
        self._store_xyz("lvlh/pos_", lvlh_pos)

        # Attitude error as Euler angles (per body axis) and as a quaternion
        q_err_euler = math_utils.quat_mul(math_utils.quat_conjugate(quat), target_quat)
        q_err_euler = torch.where(q_err_euler[:, 0:1] < 0, -q_err_euler, q_err_euler)
        euler_err = quaternion_to_euler_deg(q_err_euler)
        euler_err = torch.where(euler_err > 180.0, euler_err - 360.0, euler_err)
        self._store_xyz("orientation/error_", euler_err.abs(), "_deg")
        self._store_xyz("orientation/current_", quaternion_to_euler_deg(quat), "_deg")
        self._store_xyz("orientation/target_", quaternion_to_euler_deg(target_quat), "_deg")

        e_quat = math_utils.quat_mul(math_utils.quat_conjugate(target_quat), quat)
        e_quat = torch.where(e_quat[:, 0:1] < 0, -e_quat, e_quat)  # w >= 0 avoids sign jumps
        for i in range(4):
            m[f"error_quaternion/q{i}"] = e_quat[:, i]

        # Velocities
        lin_vel = robot.data.root_lin_vel_b
        ang_vel = robot.data.root_ang_vel_b
        m["root_velocity_norm"] = torch.norm(torch.cat([lin_vel, ang_vel], dim=-1), dim=-1)
        self._store_xyz("lin_vel_", lin_vel)
        self._store_xyz("angular_velocity_", robot.data.root_com_ang_vel_b)

        # Booms: measured state and commanded targets, normalized to [0, 1]
        boom_states = get_boom_states(self._env)
        extension, velocity = boom_states[:, :4], boom_states[:, 4:]
        action_term = self._env.action_manager._terms["boom_extension"]
        target = (action_term.target_boom_extensions / action_term.max_len).clamp(0.0, 1.0)
        for i in range(4):
            m[f"boom/ext_{i + 1}"] = extension[:, i]
            m[f"boom/vel_{i + 1}"] = velocity[:, i]
            m[f"boom/target_ext_{i + 1}"] = target[:, i]
        m["boom/extension_mean"] = extension.mean(dim=1)
        m["boom/velocity_mean"] = velocity.abs().mean(dim=1)
        m["boom/velocity_max"] = velocity.abs().max(dim=1).values

        if DragGlobal.drag_forces is not None and DragGlobal.drag_torques is not None:
            self._store_xyz("drag/force_", DragGlobal.drag_forces)
            self._store_xyz("drag/torque_", DragGlobal.drag_torques)

        self._episode_steps += 1
        m["sim_time"] = self._episode_steps * self._env.step_dt
        self._log_wandb()

    def _log_wandb(self):
        if self._wandb is None or self._wandb.run is None:
            return
        self._wandb_step += 1
        log = {key: value[0].item() for key, value in self.metrics.items()}
        # Population statistics of the pose error over all environments
        for key in ("position_error", "orientation_error", "orientation_error_deg", "root_velocity_norm"):
            log[f"{key}_mean"] = self.metrics[key].mean().item()
        log["position_error_max"] = self.metrics["position_error"].max().item()
        try:
            self._wandb.log(log, step=self._wandb_step)
        except Exception as e:
            print(f"W&B logging error, disabling: {e}")
            self._wandb = None


@configclass
class OrbitDataCommandCfg(CommandTermCfg):
    class_type: type[CommandTerm] = OrbitDataCommand
    orbit_data_path: str = "orbit_data/orbit_data.h5"
    resampling_time_range: tuple[float, float] = (-1.0, -1.0)


@configclass
class TargetPoseCommandCfg(CommandTermCfg):
    class_type: type[CommandTerm] = TargetPoseCommand
    orbit_data_term_name: str = "orbit_data"
    use_orbit_position: bool = False
    resampling_time_range: tuple[float, float] = (-1.0, -1.0)
    visualize_target: bool = True
    log_metrics: bool = False  # compute tracking metrics and log them to W&B (for inference runs)
    wandb_project: str = "D3_2U_inference"