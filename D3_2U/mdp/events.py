from __future__ import annotations

from doctest import debug
import glob
import math
import os
from typing import TYPE_CHECKING, Optional

import h5py
import numpy as np
import torch
import torch.nn as nn

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

NUM_BOOMS = 4
NUM_LINKS_PER_BOOM = 34
MAX_BOOM_LENGTH = 3.7  # m


##
# Reset
##


def reset_cubesat_body(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the CubeSat to the LVLH frame plus a random pose and velocity offset

    Based on "reset_root_state_uniform". The angular velocity is the LVLH rotation rate
    plus a random perturbation

    Args:
        env: The environment instance
        env_ids: Environment indices to reset
        pose_range: (min, max) per axis for 'x', 'y', 'z', 'roll', 'pitch', 'yaw'
        velocity_range: (min, max) per axis for 'x', 'y', 'z', 'roll', 'pitch', 'yaw'
        asset_cfg: Asset configuration
    """
    asset: Articulation = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()

    keys = ["x", "y", "z", "roll", "pitch", "yaw"]

    # Sample positions and orientations
    ranges = torch.tensor([pose_range.get(key, (0.0, 0.0)) for key in keys], device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]

    # Reset the orbit data for the new episode
    orbit_cmd = env.command_manager._terms["orbit_data"]
    orbit_cmd.reset_orbit_data(env_ids) #, reset_time=0.0)

    orbit_data = env.command_manager.get_command("orbit_data")
    target_orientation = orbit_data.get("orientation")[env_ids].to(dtype=torch.float32)
    orbit_position = orbit_data.get("position")[env_ids].to(dtype=torch.float32)
    orbit_vel = orbit_data.get("velocity")[env_ids].to(dtype=torch.float32)

    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(orientations_delta, target_orientation)

    # Sample velocities
    ranges = torch.tensor([velocity_range.get(key, (0.0, 0.0)) for key in keys], device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    lin_vels = rand_samples[:, :3]

    # Angular velocity in the world frame from orbit mechanics: omega = (r x v) / |r|^2
    r2 = (orbit_position**2).sum(dim=-1, keepdim=True)
    h_w = torch.cross(orbit_position, orbit_vel, dim=-1)
    omega_world = h_w / r2

    omega_noise = math_utils.quat_apply(orientations, rand_samples[:, 3:6])  # noise in the world frame
    ang_vels = omega_world + omega_noise

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.cat([lin_vels.contiguous(), ang_vels.contiguous()], dim=-1), env_ids=env_ids)
    asset.write_data_to_sim()


def reset_booms_visual_noPM(
    env,
    env_ids: torch.Tensor,
    extension_range=(0.0, 3.7),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the master boom joints (d3_boom_i_joint) to a random extension

    - Zero joint velocities
    - Write the joint state in one operation, so there is no impulse
    - Align the actuator position targets with the state, so the PD error is zero
    - Synchronize the boom action buffers so that RL sees consistent values at t=0
    """
    asset = env.scene[asset_cfg.name]
    joint_names = asset.joint_names
    device = env.device

    B = env_ids.shape[0]  # number of environments being reset
    D = len(joint_names)  # total DOFs of the articulation

    # Joint indices of the masters (robust to the USD ordering)
    master_idx = []
    for b in range(1, NUM_BOOMS + 1):
        name = f"d3_boom_{b}_joint"
        try:
            master_idx.append(joint_names.index(name))
        except ValueError:
            raise KeyError(f"Missing master joint {name}")
    master_idx = torch.as_tensor(master_idx, device=device, dtype=torch.long)

    exts = math_utils.sample_uniform(extension_range[0], extension_range[1], (B, NUM_BOOMS), device=device)

    pos = torch.zeros((B, D), device=device)
    vel = torch.zeros_like(pos)  # zero velocity to avoid adding momentum
    pos[:, master_idx] = exts

    # Respect the joint limits
    lim_pos = asset.data.soft_joint_pos_limits[env_ids]
    lim_vel = asset.data.soft_joint_vel_limits[env_ids]
    pos = torch.max(torch.min(pos, lim_pos[..., 1]), lim_pos[..., 0])
    vel = torch.clamp(vel, -lim_vel, lim_vel)

    # Teleport the joints without impulses
    asset.write_joint_state_to_sim(pos, vel, env_ids=env_ids)

    # Align the actuator targets (masters only) so that the PD error is zero
    target = torch.zeros((B, D), device=device)
    target[:, master_idx] = pos[:, master_idx]
    asset.set_joint_position_target(target, env_ids=env_ids)

    # Sync the action term so the commanded extension matches the teleported state
    act = env.action_manager._terms.get("boom_extension")
    if act is not None:
        act._current_extension[env_ids] = pos[:, master_idx]
        act._raw_actions[env_ids] = 0.0  # 0 means "do not move"

    asset.reset(env_ids)


def reset_booms(
    env,
    env_ids: torch.Tensor,
    extension_range=(0.0, 3.7),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the master boom joints (d3_boom_i_joint) and their mimic point-mass joints
    (d3_boom_i_pointmass_joint)

    - Randomize the master extension in extension_range, and set each mimic to 0.5 * master
    - Zero the joint velocities
    - Write the joint state in one operation, so there is no impulse
    - Align the actuator position targets with the state, so the PD error is zero
    - Synchronize the boom action buffers so that RL sees consistent values at t=0
    """
    asset = env.scene[asset_cfg.name]
    joint_names = asset.joint_names
    device = env.device

    B = env_ids.shape[0]  # number of environments being reset
    D = len(joint_names)  # total DOFs of the articulation

    # Joint indices of the masters and the mimics (robust to the USD ordering)
    master_idx = []
    mimic_idx = []
    for b in range(1, NUM_BOOMS + 1):
        master_name = f"d3_boom_{b}_joint"
        mimic_name = f"d3_boom_{b}_pointmass_joint"
        try:
            master_idx.append(joint_names.index(master_name))
        except ValueError:
            raise KeyError(f"Missing master joint {master_name}")
        try:
            mimic_idx.append(joint_names.index(mimic_name))
        except ValueError:
            raise KeyError(f"Missing mimic joint {mimic_name}")
    master_idx = torch.as_tensor(master_idx, device=device, dtype=torch.long)
    mimic_idx = torch.as_tensor(mimic_idx, device=device, dtype=torch.long)

    exts = math_utils.sample_uniform(extension_range[0], extension_range[1], (B, NUM_BOOMS), device=device)

    pos = torch.zeros((B, D), device=device)
    vel = torch.zeros_like(pos)  # zero velocity to avoid adding momentum
    pos[:, master_idx] = exts
    pos[:, mimic_idx] = 0.5 * exts  # mimics track half of the extension

    # Respect the joint limits
    lim_pos = asset.data.soft_joint_pos_limits[env_ids]
    lim_vel = asset.data.soft_joint_vel_limits[env_ids]
    pos = torch.max(torch.min(pos, lim_pos[..., 1]), lim_pos[..., 0])
    vel = torch.clamp(vel, -lim_vel, lim_vel)

    # Teleport the joints without impulses
    asset.write_joint_state_to_sim(pos, vel, env_ids=env_ids)

    # Align the actuator targets (masters only) so that the PD error is zero
    target = torch.zeros((B, D), device=device)
    target[:, master_idx] = pos[:, master_idx]
    asset.set_joint_position_target(target, env_ids=env_ids)

    # Sync the action term so the commanded extension matches the teleported state
    act = env.action_manager._terms.get("boom_extension")
    if act is not None:
        act._current_extension[env_ids] = pos[:, master_idx]
        act._raw_actions[env_ids] = 0.0  

    asset.reset(env_ids)

##
# Orbit data
##


class OrbitDataManager:
    """Time-synchronized access to pre-computed orbit data (HDF5) during the simulation

    A window of the file is cached on the device. The window is loaded in ``reset()``
    and sized to cover a full episode, so ``step()`` never needs to reload it
    """

    # Dataset in the HDF5 file for each cached quantity
    DATASETS = {
        "position": "orbit/position",
        "orientation": "orbit/orientation",
        "velocity": "orbit/velocity",
        "atm_ang_velocity": "atmosphere/ang_velocity",
        "air_density": "atmosphere/air_density",
        "j2_perturbation_acc": "orbit/j2_perturbation_acc",
    }

    def __init__(
        self,
        orbit_data_path: str,
        device: str,
        sim_dt: float,
        episode_length_s: float = 10.0,
        decimation: float = 0.0,
        time_scale: float = 1.0,
        orbit_data_dt: Optional[float] = None,
    ):
        """
        Args:
            orbit_data_path: Path to the HDF5 file with the orbit data
            device: PyTorch device
            sim_dt: Simulation timestep [s]
            episode_length_s: Episode length [s]
            decimation: Simulation steps per policy step
            time_scale: Scale between simulated and orbit time
            orbit_data_dt: Time between orbit data points (calculated from the data if None)
        """
        self.device = device
        self.sim_dt = sim_dt
        self.orbit_data_path = orbit_data_path
        self.episode_length_s = episode_length_s
        self.decimation = decimation
        self.time_scale = time_scale
        self.policy_dt = sim_dt * decimation * time_scale

        self.file = h5py.File(orbit_data_path, "r")
        self.total_timesteps = len(self.file["time_steps"])
        self.time_steps = torch.tensor(self.file["time_steps"][:], device=device)

        if orbit_data_dt is None:
            if self.total_timesteps > 1:
                orbit_data_dt = float(self.time_steps[1] - self.time_steps[0])
            else:
                orbit_data_dt = sim_dt
        self.orbit_data_dt = orbit_data_dt

        # The cache must cover a full episode: orbit points advanced in one episode plus a safety margin
        num_policy_steps = math.ceil(episode_length_s / self.policy_dt)
        orbit_points_needed = math.ceil(num_policy_steps * self.policy_dt / orbit_data_dt)
        safety_buffer = max(100, int(orbit_points_needed * 0.5))
        self.cache_size = orbit_points_needed + safety_buffer

        self.current_time_idx = 0
        self.accumulated_time = 0.0
        self.current_sim_time = float(self.time_steps[0])

        self._load_cache(0)

    def _load_cache(self, start_idx: int):
        """Load the cache window [start_idx, start_idx + cache_size) from the file"""
        self.cache_start_idx = max(0, start_idx)
        self.cache_end_idx = min(self.total_timesteps, self.cache_start_idx + self.cache_size)

        actual_cache_size = self.cache_end_idx - self.cache_start_idx
        if actual_cache_size < self.cache_size:
            print(f"WARNING: could not load the full cache (requested {self.cache_size}, got {actual_cache_size}), "
                  "the start time is close to the end of the orbit data")

        self.cached_data = {}
        try:
            for key, path in self.DATASETS.items():
                data_slice = self.file[path][self.cache_start_idx:self.cache_end_idx] if path in self.file else []
                if len(data_slice) == 0:
                    self._initialize_default_data(key)
                    continue
                tensor = torch.tensor(data_slice, device=self.device)
                if key == "air_density":
                    tensor = self._fill_nan_density(tensor)
                    if tensor.dim() == 1:
                        tensor = tensor.unsqueeze(-1)
                self.cached_data[key] = tensor
        except Exception as e:
            print(f"ERROR loading the orbit data cache: {e}")
            for key in ["position", "velocity", "atm_ang_velocity", "air_density", "j2_perturbation_acc"]:
                self.cached_data[key] = torch.zeros((1, 3), device=self.device)
            self.cached_data["orientation"] = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)

    def _initialize_default_data(self, key: str):
        print(f"Initializing default orbit data for '{key}'")
        if key == "orientation":
            self.cached_data[key] = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)
        elif key == "air_density":
            self.cached_data[key] = torch.zeros((1, 1), device=self.device)
        else:
            self.cached_data[key] = torch.zeros((1, 3), device=self.device)

    def reset(self, start_time=None):
        """Reset the orbit to ``start_time`` [s], or to a random time if None

        This is the only place where the cache is normally reloaded
        """
        if start_time is not None:
            self.current_time_idx = torch.argmin(torch.abs(self.time_steps - start_time)).item()
            self.current_time_idx = min(max(self.current_time_idx, 0), self.total_timesteps - 1)
        else:
            # A random start must leave at least one full cache window ahead
            max_safe_start = self.total_timesteps - self.cache_size
            if max_safe_start < 10:
                print(f"WARNING: orbit data may be too short (total {self.total_timesteps} points, "
                      f"cache size {self.cache_size})")
                max_safe_start = max(1, self.total_timesteps // 2)
            self.current_time_idx = torch.randint(0, max_safe_start, (1,), device=self.device).item()

        self.current_sim_time = float(self.time_steps[self.current_time_idx])
        self.accumulated_time = 0.0

        self._load_cache(self.current_time_idx)

        cache_coverage = self.cache_end_idx - self.current_time_idx
        orbit_points_needed = math.ceil(self.episode_length_s / self.orbit_data_dt)
        if cache_coverage < orbit_points_needed:
            print(f"CRITICAL ERROR: insufficient cache coverage: {cache_coverage} points "
                  f"({cache_coverage * self.orbit_data_dt:.2f}s) for an episode that needs {orbit_points_needed} "
                  f"({orbit_points_needed * self.orbit_data_dt:.2f}s). The episode will run out of cached data")

    def step(self):
        """Advance the orbit data by one policy step"""
        self.accumulated_time += self.policy_dt
        steps_to_advance = int(self.accumulated_time / self.orbit_data_dt)
        if steps_to_advance == 0:
            return

        self.current_time_idx += steps_to_advance
        self.accumulated_time -= steps_to_advance * self.orbit_data_dt

        if self.current_time_idx < self.total_timesteps:
            self.current_sim_time = float(self.time_steps[self.current_time_idx])

            # Should never happen if reset() was called: it means the cache was sized wrongly
            if not self.cache_start_idx <= self.current_time_idx < self.cache_end_idx:
                print(f"ERROR: cache miss during step() at index {self.current_time_idx} "
                      f"(cache [{self.cache_start_idx}, {self.cache_end_idx}))")
                self._load_cache(self.current_time_idx)

    def _fill_nan_density(self, data: torch.Tensor) -> torch.Tensor:
        """Replace NaNs in a (T, 1) density tensor by interpolating over time

        With fewer than 2 valid points a small floor is used instead
        """
        arr = data.squeeze(-1).cpu().numpy()
        idx = np.arange(len(arr))
        valid = ~np.isnan(arr)

        if valid.sum() >= 2:
            filled = np.interp(idx, idx[valid], arr[valid])
        else:
            filled = np.nan_to_num(arr, nan=1e-12)
            print("Not enough valid air density points, using a floor value of 1e-12 kg/m³")

        return torch.from_numpy(filled).to(self.device).unsqueeze(-1)

    def get_data(self, data_key, num_envs=1):
        """Current value of an orbit quantity, repeated for ``num_envs`` environments"""
        if data_key not in self.cached_data:
            print(f"Unknown data key: {data_key}. Available keys: {list(self.cached_data.keys())}")
            return self._get_fallback_data(data_key, num_envs)

        cache_idx = self.current_time_idx - self.cache_start_idx
        cache_idx = min(max(0, cache_idx), len(self.cached_data[data_key]) - 1)

        data = self.cached_data[data_key][cache_idx]
        if num_envs > 1:
            return data.unsqueeze(0).repeat(num_envs, 1)
        return data.unsqueeze(0)

    def _get_fallback_data(self, data_key, num_envs):
        if data_key == "orientation":
            return torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(num_envs, 1)
        if data_key == "air_density":
            return torch.zeros((num_envs, 1), device=self.device)
        return torch.zeros((num_envs, 3), device=self.device)

    def get_current_time(self):
        """Current simulation time [s]"""
        return self.current_sim_time

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None


##
# Raycaster
##


def create_raycaster_marker() -> VisualizationMarkers:
    """Visualization marker for the raycaster"""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/RaycasterMarker",
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),  # slightly smaller than the target marker
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)


def update_raycaster_orientation(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    orbit_data_term_name: str = "orbit_data",
    offset_raycaster: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    raycaster_name: str = "drag_sensor",
):
    """Update the position and orientation of the RayCaster

    - The position is offset from the CubeSat along the orbital +Y axis
    - The orientation aligns +Z with the orbital Y axis, using look-at logic with world Z as up
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    raycaster = env.scene[raycaster_name]
    orbit_data = env.command_manager.get_command(orbit_data_term_name)

    orbit_quat = orbit_data["orientation"][env_ids].to(dtype=torch.float32)
    orbit_quat = orbit_quat / torch.norm(orbit_quat, dim=1, keepdim=True)
    cubesat_pos = env.scene[asset_cfg.name].data.root_pos_w[env_ids].to(dtype=torch.float32)

    # Position
    offset_orbit = torch.tensor([0.0, offset_raycaster, 0.0], device=env.device).repeat(len(env_ids), 1)
    raycaster_pos = cubesat_pos + math_utils.quat_apply(orbit_quat, offset_orbit)

    # Orientation
    orbital_y_local = torch.tensor([0.0, 1.0, 0.0], device=env.device).repeat(len(env_ids), 1)
    forward = torch.nn.functional.normalize(math_utils.quat_apply(orbit_quat, orbital_y_local), dim=1)

    up = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(len(env_ids), 1)
    right = torch.nn.functional.normalize(torch.cross(up, forward, dim=1), dim=1)
    up_proj = torch.cross(forward, right, dim=1)

    rot_matrix = torch.stack([right, up_proj, forward], dim=-1)  # columns are [right, up, forward]
    raycaster_quat = torch.nn.functional.normalize(math_utils.quat_from_matrix(rot_matrix), dim=1)

    raycaster._data.pos_w[env_ids] = raycaster_pos
    raycaster._data.quat_w[env_ids] = raycaster_quat

    marker_indices = torch.zeros(len(env_ids), dtype=torch.long, device=env.device)
    env._raycaster_marker.visualize(raycaster_pos, raycaster_quat, marker_indices=marker_indices)


##
# Ray-casting drag
##


def calculate_drag_force(
    area: float,
    air_density: torch.Tensor,
    orbit_velocity: torch.Tensor,
    orbit_position: torch.Tensor,
    atmosphere_ang_velocity: torch.Tensor,
    drag_coefficient: float = 2.2,
) -> torch.Tensor:
    """Drag force magnitude on a surface of the given area

    Args:
        area: Cross-sectional area exposed to drag [m²]
        air_density: Air density [kg/m³], shape [num_envs, 1]
        orbit_velocity: Orbital velocity [km/s], shape [num_envs, 3]
        orbit_position: Orbital position [km], shape [num_envs, 3]
        atmosphere_ang_velocity: Atmosphere angular velocity [rad/s], shape [num_envs, 3]
        drag_coefficient: Drag coefficient

    Returns:
        Drag force magnitudes [N], shape [num_envs]
    """
    # Atmosphere velocity at the spacecraft position
    atmosphere_velocity = torch.cross(atmosphere_ang_velocity, orbit_position, dim=1)

    # Relative velocity magnitude, converted from km/s to m/s
    relative_velocity = torch.norm(orbit_velocity - atmosphere_velocity, dim=1) * 1000.0

    # 0.5 * rho * v^2 * Cd * A
    return 0.5 * air_density.squeeze(-1) * (relative_velocity**2) * drag_coefficient * area


def create_mesh_to_body_mapping(robot, raycaster):
    """Map the mesh indices of the raycaster to the body (link) indices of the robot

    The link is found from the mesh prim path, e.g. ".../d3_boom_1_p1_link/visuals/..."
    """
    mapping = {}
    for mesh_idx, mesh_path in enumerate(raycaster.cfg.mesh_prim_paths):
        for component in mesh_path.split("/"):
            if component.endswith("_link"):
                for link_idx, link_name in enumerate(robot.body_names):
                    if component == link_name:
                        mapping[mesh_idx] = link_idx
                        break
    return mapping


def calculate_atmospheric_drag_forces(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    drag_coefficient: float = 2.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    drag_sensor_name: str = "drag_sensor",
    orbit_data_term_name: str = "orbit_data",
    return_net_at_root: bool = False,
    return_net_at_root_local: bool = False,
):
    """Net drag force and torque at the root, in the world frame, from the ray-casting sensor

    Every ray that hits the spacecraft contributes the drag of its cell area, in the direction
    opposite to the orbital velocity (-Y of the LVLH frame). The torque is taken about the root COM.
    Nothing is applied to the simulation.

    Returns:
        Tuple (forces [N_valid, 3], torques [N_valid, 3], valid env ids [N_valid]), or
        (None, None, None) if no environment has a valid hit
    """
    debug = False
    
    # --- Environments
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if len(env_ids) == 0:
        return None, None, None
    
    # --- Get Orbit Data
    orbit_command = env.command_manager.get_command(orbit_data_term_name)
    if not orbit_command:
        print(f"Failed to get orbit data command with name {orbit_data_term_name}")
        return None, None, None
    
    # Get the orbit orientation (target pose orientation)
    orbit_orientation = orbit_command.get("orientation")[env_ids]
    
    
    # --- Raycaster
    raycaster = env.scene[drag_sensor_name]
    if raycaster is None:
        print(f"Warning: Raycaster sensor with name {drag_sensor_name} not found in the scene.")
        return None, None, None
    if not hasattr(raycaster, "_data") or raycaster._data.ray_hits_w is None:
        if debug:
            print("No valid sensor data available")
        return None, None, None

    # Get sensor data
    device = env.device
    
    # Get raycaster data for selected environments
    raycaster_ray_hits = raycaster._data.ray_hits_w[env_ids]
    raycaster_hit_mesh_indices = raycaster._data.hit_mesh_indices[env_ids]
    
    # Find environments with valid hits
    hit_counts = torch.sum(raycaster_hit_mesh_indices != -1, dim=1)
    valid_env_mask = hit_counts > 0
    
    if debug:
            print(f"Processing {len(env_ids)} environments with {hit_counts.tolist()} hits")
    
    if not torch.any(valid_env_mask):
        if debug:
            print("No environments have valid hits, skipping")
        return None, None, None
    
    # Filter to only environments with valid hits
    valid_env_indices = torch.nonzero(valid_env_mask, as_tuple=True)[0]
    valid_env_ids = env_ids[valid_env_indices]
    
    
    # Get the robot articulation
    robot = env.scene[asset_cfg.name]
    
    # Get the mapping between mesh IDs and links
    if not hasattr(env, "_mesh_id_to_body_map"):
        print("Initializing mesh ID to body mapping")
        env._mesh_id_to_body_map = create_mesh_to_body_mapping(robot, raycaster)
            

    # Drag force of a single ray cell (the orbit data is the same for all environments)
    drag_per_ray = calculate_drag_force(
        area=raycaster.cfg.pattern_cfg.resolution**2,
        air_density=orbit_command.get("air_density"),
        orbit_velocity=orbit_command.get("velocity"),
        orbit_position=orbit_command.get("position"),
        atmosphere_ang_velocity=orbit_command.get("atm_ang_velocity"),
        drag_coefficient=drag_coefficient,
    )[: len(env_ids)]
    valid_drag = drag_per_ray[valid_env_indices]

    # Ensure drag_per_ray has the right shape
    if drag_per_ray.shape[0] != len(env_ids):
        new_drag = torch.zeros(len(env_ids), device=device)
        copy_size = min(drag_per_ray.shape[0], len(env_ids))
        new_drag[:copy_size] = drag_per_ray[:copy_size]
        drag_per_ray = new_drag
    
    # Filter drag values for valid environments
    valid_drag = drag_per_ray[valid_env_indices]
    
    # Create force vectors in LVLH frame and transform to world frame
    num_valid_envs = len(valid_env_ids)
    
    # Define negative Y direction in LVLH frame (force is opposite to velocity)
    neg_y_lvlh = torch.zeros((num_valid_envs, 3), device=device, dtype=torch.float)
    neg_y_lvlh[:, 1] = -1.0  # Negative Y-axis in LVLH frame
    
    orbit_orientation = orbit_orientation[env_ids].to(dtype=torch.float64)
    neg_y_lvlh = neg_y_lvlh.to(dtype=torch.float64)
    
    # Transform to world frame using orbit orientation quaternions
    force_dir_world = math_utils.quat_apply(
        orbit_orientation[valid_env_indices], 
        neg_y_lvlh
    )
    
    # Normalize the direction
    force_dir_norm = torch.norm(force_dir_world, dim=1, keepdim=True)
    valid_norm_mask = force_dir_norm > 1e-6
    force_dir_world = torch.where(
        valid_norm_mask, 
        force_dir_world / force_dir_norm, 
        neg_y_lvlh  # Fallback in case of normalization issues
    )
    
    
    # Apply drag magnitude to get force vectors
    force_vectors = force_dir_world.unsqueeze(1) * valid_drag.view(num_valid_envs, 1, 1)
    
    if debug and len(valid_env_ids) > 0:
        print(f"Drag force direction (world frame): {force_dir_world[0]}")
        print(f"Drag force magnitude: {valid_drag[0]}")
        print(f"Drag force vector: {force_vectors[0, 0]}")
    
    # Initialize forces and torques for all bodies in valid environments
    local_forces = torch.zeros((num_valid_envs, robot.num_bodies, 3), device=device, dtype=torch.float)
    local_torques = torch.zeros((num_valid_envs, robot.num_bodies, 3), device=device, dtype=torch.float)
        

    # Return the net forces and torques at the root (cubesat)
    if return_net_at_root:
        net_forces_at_root = torch.zeros((num_valid_envs, 3), device=device)
        net_torques_at_root = torch.zeros((num_valid_envs, 3), device=device)
        # root_positions = robot.data.root_pos_w[valid_env_ids][:, :3]
        root_com_positions = robot.data.root_com_pos_w[valid_env_ids].to(dtype=torch.float32)
    

    # Get valid hit data
        valid_ray_hits = raycaster_ray_hits[valid_env_indices]
        valid_hit_indices = raycaster_hit_mesh_indices[valid_env_indices]
        # print("Number of hits in valid environments: ", torch.sum(valid_hit_indices != -1).item(), " total area of ", torch.sum((valid_hit_indices != -1).sum(dim=1) * (raycaster.cfg.pattern_cfg.resolution ** 2)).item(), " m^2")
        
        # Get body positions and quaternions for valid environments
        body_com_positions = robot.data.body_com_pos_w[valid_env_ids].to(dtype=torch.float)
        body_quats = robot.data.body_quat_w[valid_env_ids]
        body_quats_inv = math_utils.quat_inv(body_quats)
        
        # Process meshes with valid hits across all environments simultaneously
        all_mesh_ids = torch.unique(valid_hit_indices[valid_hit_indices != -1])
        
        if debug:
            print(f"Processing {len(all_mesh_ids)} unique mesh IDs")
        # print(f"Processing {len(all_mesh_ids)} unique mesh IDs")
    
        
        for mesh_id in all_mesh_ids:
            mesh_id_int = mesh_id.item()
            if mesh_id_int not in env._mesh_id_to_body_map:
                if debug:
                    print(f"Mesh ID {mesh_id_int} not in body map, skipping")
                continue
            
            body_id = env._mesh_id_to_body_map[mesh_id_int]
            
            # Create a mask for rays that hit this mesh ID across all valid environments
            mesh_hit_mask = valid_hit_indices == mesh_id
            
            # For each environment, process the hits for this mesh
            for env_idx in range(num_valid_envs):
                # Get the rays that hit this mesh in this environment
                env_mesh_mask = mesh_hit_mask[env_idx]
                if not torch.any(env_mesh_mask):
                    continue
                    
                ray_indices = torch.nonzero(env_mesh_mask, as_tuple=True)[0]

                # Get hit points for this mesh in this environment
                hit_points = valid_ray_hits[env_idx, ray_indices]
                # print("Num hits ", len(hit_points), " for mesh ", mesh_id_int, " in env ", valid_env_ids[env_idx].item())
                
                # Use the same force vector from our calculated direction for all hits
                hit_forces = force_vectors[env_idx].expand(len(ray_indices), 3).to(dtype=torch.float)
                
                # Get total force on this body
                net_force = hit_forces.sum(dim=0) if hit_forces.size(0) > 0 else torch.zeros(3, device=device)
                
                if return_net_at_root:
                    net_forces_at_root[env_idx] += net_force
                    
                    # Torque contribution: sum of r × F for each hit
                    root_com = root_com_positions[env_idx]
                    r_vectors = hit_points - root_com  # lever arms relative to root COM
                    torques = torch.cross(r_vectors, hit_forces, dim=1)
                    net_torque = torques.sum(dim=0)
                    net_torques_at_root[env_idx] += net_torque
                    # print("Net force ", net_force, " - ", net_torque)     
                else:
                    # Center of mass for the body in this environment
                    body_com = body_com_positions[env_idx, body_id]
                    
                    # Calculate relative positions and torques
                    r_vectors = hit_points - body_com
                    torques = torch.cross(r_vectors, hit_forces, dim=1)
                    
                    # Sum forces and torques for this mesh in this environment
                    net_torque = torques.sum(dim=0) if torques.size(0) > 0 else torch.zeros(3, device=device)
                    # print("Body force ", net_force, " vs ", net_torque)
                
                    # Skip tiny forces
                    if torch.norm(net_force) <= 1e-16:
                        continue
                    
                    # Transform to local frame
                    local_force = math_utils.quat_apply(body_quats_inv[env_idx, body_id], net_force)
                    local_torque = math_utils.quat_apply(body_quats_inv[env_idx, body_id], net_torque)
                    
                    # Add to the accumulated forces and torques
                    local_forces[env_idx, body_id] += local_force
                    local_torques[env_idx, body_id] += local_torque
                
        # Return net or local based on flag
        if return_net_at_root:
            # print("Net forces ", net_forces_at_root, " vs ", net_forces_at_root[env_ids])
            
            # Test:
            test = False
            if test:
                # Convert world forces to local frame
                root_quat = robot.data.root_quat_w[valid_env_ids]
                root_quat_inv = math_utils.quat_inv(root_quat)
                
                local_forces = math_utils.quat_apply(root_quat_inv, net_forces_at_root)
                local_torques = math_utils.quat_apply(root_quat_inv, net_torques_at_root)
                
                # Create tensors for set_external_force_and_torque
                forces = torch.zeros((len(valid_env_ids), robot.num_bodies, 3), device=env.device)
                torques = torch.zeros((len(valid_env_ids), robot.num_bodies, 3), device=env.device)
                
                # Set forces and torques for root body (index 0)
                forces[:, 0] = local_forces
                torques[:, 0] = local_torques
                
                print("\nConverted to local frame:")
                for i in range(min(3, len(valid_env_ids))):  # Print up to 3 environments
                    env_idx = valid_env_ids[i].item()
                    print(f"Env {env_idx}:")
                    print(f"  Local force: {local_forces[i]} N - world {net_forces_at_root}" )
                    print(f"  Local torque: {local_torques[i]} N·m - world {net_torques_at_root}")
                        
                # Apply forces
                robot.set_external_force_and_torque(
                    forces=forces,
                    torques=torques,
                    body_ids=None,  # Apply to all bodies (will be zeroed except root)
                    env_ids=valid_env_ids
                )
                
                # Write to simulation
                robot.write_data_to_sim()
                
                print(f"root state {robot.data.root_state_w}")
                print(f"Body 0 state {robot.data.body_state_w[0]}")
                print(f"Body 1 state {robot.data.body_state_w[1]}")
    
                print(f"root com state {robot.data.root_com_state_w}")    
                print(f"body 0 root state {robot.data.body_com_state_w[0]}")     
                print(f"body 1 root state {robot.data.body_com_state_w[1]}")        
       
                print("-----------------")
                print("Successfully applied drag forces to root body!")
                
                
            if return_net_at_root_local:            # Convert net forces and torques to local frame at root
                root_quat = robot.data.root_quat_w[valid_env_ids]
                root_quat_inv = math_utils.quat_inv(root_quat)
                
                local_forces_at_root = math_utils.quat_apply(root_quat_inv, net_forces_at_root)
                local_torques_at_root = math_utils.quat_apply(root_quat_inv, net_torques_at_root)
                
                if debug:
                    print(f"Local forces at root for first environment: {local_forces_at_root[0]}")
                    print(f"Local torques at root for first environment: {local_torques_at_root[0]}")
                    
                print("Net Local ", local_forces_at_root[0], " -- ", local_torques_at_root[0])
                # print("Net World ", net_forces_at_root[0], " -- ", net_torques_at_root[0])
                    
                return local_forces_at_root, local_torques_at_root, valid_env_ids
            else:
                # print("Returning net forces and torques at root in world frame")
                # print(f"Net forces at root (world frame) for first environment: {net_forces_at_root[0]}")
                # print(f"Net torques at root (world frame) for first environment: {net_torques_at_root[0]}")
                return net_forces_at_root, net_torques_at_root, valid_env_ids
        
        # Find environments with significant forces to apply
        force_magnitudes = torch.sum(torch.norm(local_forces, dim=2), dim=1)
        significant_forces_mask = force_magnitudes > 1e-16
        
        if debug:
            print(f"Drag forces in first body: {local_forces[0,0]}")
            print(f"Drag force magnitudes: {force_magnitudes}")
        
        # Return only environments with significant forces
        if torch.any(significant_forces_mask):
            final_env_indices = torch.nonzero(significant_forces_mask, as_tuple=True)[0]
            final_env_ids = valid_env_ids[final_env_indices]
            
            return local_forces[final_env_indices], local_torques[final_env_indices], final_env_ids
        
        return None, None, None

##
# Gravity gradient
##


class OptimizedOrbitLink:
    """GPU-batched orbital perturbations (tidal + J2 forces and gravity-gradient torque)

    LVLH axes: X radial (outward), Y along-track (velocity), Z orbit normal
    """

    def __init__(self, num_envs: int, device: str, body_indices: list[int], angular_velocity: float):
        self.num_envs = num_envs
        self.device = device
        self.body_indices = torch.tensor(body_indices, device=device, dtype=torch.long)
        self.num_bodies = len(body_indices)

        # Mean motion
        self.n0 = torch.tensor(angular_velocity, device=device)
        n0_sq = self.n0**2

        # Clohessy-Wiltshire accelerations from the state [x, y, z, vx, vy, vz]
        #   ax = 3n^2 x + 2n vy   radial: tide + Coriolis
        #   ay = -2n vx           along-track: Coriolis
        #   az = -n^2 z           cross-track: harmonic oscillator
        self.cw = torch.zeros((num_envs, 3, 6), device=device)
        self.cw[:, 0, 0] = 3.0 * n0_sq
        self.cw[:, 0, 4] = 2.0 * self.n0
        self.cw[:, 1, 3] = -2.0 * self.n0
        self.cw[:, 2, 2] = -1.0 * n0_sq

        self.forces = torch.zeros((num_envs, self.num_bodies, 3), device=device)
        self.torques = torch.zeros((num_envs, self.num_bodies, 3), device=device)

        # The gravity-gradient torque tries to align the principal inertia axis with the radial vector (X)
        self.radial_vec_lvlh = torch.tensor([1.0, 0.0, 0.0], device=device)
        self.radial_vec_lvlh_batch = self.radial_vec_lvlh.view(1, 1, 3, 1).expand(num_envs, self.num_bodies, -1, -1)

    def initialize_from_articulation(self, articulation: Articulation):
        self.articulation = articulation
        body_indices_cpu = self.body_indices.cpu()

        self.mass = articulation.data.default_mass[:, body_indices_cpu].to(self.device).unsqueeze(-1)

        # Inertia tensors [N, B, 3, 3] from the flattened default inertia
        inertia_flat = articulation.data.default_inertia[:, body_indices_cpu].to(self.device)
        self.inertia = torch.zeros((self.num_envs, self.num_bodies, 3, 3), device=self.device)
        self.inertia[:, :, 0, 0] = inertia_flat[:, :, 0]  # xx
        self.inertia[:, :, 1, 1] = inertia_flat[:, :, 4]  # yy
        self.inertia[:, :, 2, 2] = inertia_flat[:, :, 8]  # zz
        self.inertia[:, :, 0, 1] = self.inertia[:, :, 1, 0] = inertia_flat[:, :, 1]
        self.inertia[:, :, 0, 2] = self.inertia[:, :, 2, 0] = inertia_flat[:, :, 2]
        self.inertia[:, :, 1, 2] = self.inertia[:, :, 2, 1] = inertia_flat[:, :, 5]

    def _batch_quat_to_matrix(self, q: torch.Tensor) -> torch.Tensor:
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        R = torch.zeros(q.shape[0], 3, 3, device=self.device)
        R[:, 0, 0] = 1 - 2 * (yy + zz); R[:, 0, 1] = 2 * (xy - wz); R[:, 0, 2] = 2 * (xz + wy)
        R[:, 1, 0] = 2 * (xy + wz); R[:, 1, 1] = 1 - 2 * (xx + zz); R[:, 1, 2] = 2 * (yz - wx)
        R[:, 2, 0] = 2 * (xz - wy); R[:, 2, 1] = 2 * (yz + wx); R[:, 2, 2] = 1 - 2 * (xx + yy)
        return R

    def compute_forces(self, env_ids: torch.Tensor, orbit_data: dict, orbit_quats: torch.Tensor) -> torch.Tensor:
        """Tidal (CW) + J2 forces, expressed in the body frames"""
        body_indices_cpu = self.body_indices.cpu()
        num_bodies = self.num_bodies

        # Body states in the world frame. The simulation origin is the LVLH origin, so these
        # are already relative to the orbit reference
        pos_w = self.articulation.data.body_pos_w[env_ids][:, body_indices_cpu].to(self.device)  # [N, B, 3]
        vel_w = self.articulation.data.body_lin_vel_w[env_ids][:, body_indices_cpu].to(self.device)

        # R_WL: LVLH -> world (orbit_quats is the LVLH orientation w.r.t. the world)
        R_WL = self._batch_quat_to_matrix(orbit_quats)
        R_WL_expanded = R_WL.unsqueeze(1).expand(-1, num_bodies, -1, -1)

        # R_WB: body -> world
        q_WB = self.articulation.data.body_quat_w[env_ids][:, body_indices_cpu].to(self.device)
        R_WB = self._batch_quat_to_matrix(q_WB.reshape(-1, 4)).view(len(env_ids), num_bodies, 3, 3)

        # State in the LVLH frame [x, y, z, vx, vy, vz]
        R_LW = R_WL_expanded.transpose(-2, -1)
        pos_lvlh = torch.matmul(R_LW, pos_w.unsqueeze(-1))
        vel_lvlh = torch.matmul(R_LW, vel_w.unsqueeze(-1))
        state_lvlh = torch.cat([pos_lvlh, vel_lvlh], dim=2)  # [N, B, 6, 1]

        cw_matrix = self.cw[env_ids].unsqueeze(1).expand(-1, num_bodies, -1, -1)
        accel_lvlh = torch.matmul(cw_matrix, state_lvlh.to(dtype=cw_matrix.dtype))

        # LVLH -> world
        accel_w = torch.matmul(R_WL_expanded.to(dtype=accel_lvlh.dtype), accel_lvlh)

        # J2 perturbation, already in the world frame
        if "j2_perturbation_acc" in orbit_data:
            j2_accel_w = orbit_data["j2_perturbation_acc"][env_ids].unsqueeze(1).expand(-1, num_bodies, -1).unsqueeze(-1)
            accel_w = accel_w + j2_accel_w.to(dtype=accel_w.dtype)

        forces_w = accel_w * self.mass[env_ids].unsqueeze(-1).to(dtype=accel_w.dtype)

        # World -> body, to apply the forces locally
        forces_body = torch.matmul(R_WB.transpose(-2, -1).to(dtype=forces_w.dtype), forces_w).squeeze(-1)

        self.forces[env_ids] = forces_body.to(dtype=self.forces.dtype)
        return self.forces[env_ids]

    def compute_torques(self, env_ids: torch.Tensor, orbit_quats: torch.Tensor) -> torch.Tensor:
        """Gravity-gradient torque in the body frames: T = 3 n^2 (r x I r), with r the unit radial vector"""
        body_indices_cpu = self.body_indices.cpu()

        q_WB = self.articulation.data.body_quat_w[env_ids][:, body_indices_cpu].to(self.device)
        R_WB = self._batch_quat_to_matrix(q_WB.reshape(-1, 4)).view(len(env_ids), self.num_bodies, 3, 3)

        R_WL = self._batch_quat_to_matrix(orbit_quats)
        R_WL = R_WL.unsqueeze(1).expand(-1, self.num_bodies, -1, -1)

        # LVLH -> body: R_LB = R_WB^T * R_WL
        R_LB = torch.matmul(R_WB.transpose(-2, -1), R_WL.to(dtype=R_WB.dtype))

        # The radial vector in the body frame. The torque is the same for r and -r
        radial_body = torch.matmul(R_LB, self.radial_vec_lvlh_batch.to(dtype=R_LB.dtype)).squeeze(-1)

        I_body = self.inertia[env_ids].reshape(-1, 3, 3).to(dtype=radial_body.dtype)
        r_body = radial_body.reshape(-1, 3, 1)
        I_dot_r = torch.bmm(I_body, r_body).view(len(env_ids), self.num_bodies, 3)

        cross_prod = torch.linalg.cross(radial_body, I_dot_r, dim=-1)
        tau_body = 3.0 * (self.n0.to(dtype=cross_prod.dtype) ** 2) * cross_prod

        self.torques[env_ids] = tau_body.to(dtype=self.torques.dtype)
        return self.torques[env_ids]


def initialize_optimized_orbit_links(
    env,
    asset: Articulation,
    orbit_angular_velocity: float,
    target_bodies: Optional[list[str]] = None,
    min_mass_threshold: float = 0.0,
):
    """Create the batched orbit link for the target bodies (or all bodies above a mass threshold)"""
    if target_bodies is not None:
        target_bodies_set = set(target_bodies)
        body_indices = [idx for idx, name in enumerate(asset.data.body_names) if name in target_bodies_set]
    else:
        body_indices = [
            idx for idx in range(len(asset.data.body_names))
            if asset.data.default_mass[0, idx].item() >= min_mass_threshold
        ]

    if not body_indices:
        print("No bodies found matching the criteria")
        return

    env._optimized_orbit_link = OptimizedOrbitLink(
        num_envs=env.num_envs,
        device=env.device,
        body_indices=body_indices,
        angular_velocity=orbit_angular_velocity,
    )
    env._optimized_orbit_link.initialize_from_articulation(asset)
    print(f"Initialized optimized orbit link for {len(body_indices)} bodies")


def calculate_gravity_gradient_optimized(
    env,
    env_ids: torch.Tensor,
    asset_name: str,
    target_bodies: Optional[list[str]] = None,
    min_mass_threshold: float = 0.0,
):
    """Gravity-gradient and tidal forces and torques per target body, in the body frames

    Returns:
        Tuple (forces [N, B, 3], torques [N, B, 3], body indices), or (None, None, None)
    """
    asset: Articulation = env.scene[asset_name]

    if "orbit_data" in env.command_manager._terms:
        orbit_data = env.command_manager.get_command("orbit_data")
        orbit_quats = orbit_data["orientation"][env_ids]
    else:
        # No orbit data: identity LVLH orientation
        orbit_quats = torch.zeros((len(env_ids), 4), device=env.device)
        orbit_quats[:, 0] = 1.0
        orbit_data = {}

    if not hasattr(env, "_optimized_orbit_link"):
        n0 = 0.00113  # default LEO mean motion [rad/s]
        if "atm_ang_velocity" in orbit_data:
            n0 = torch.norm(orbit_data["atm_ang_velocity"][env_ids[0]]).item()
        initialize_optimized_orbit_links(env, asset, n0, target_bodies, min_mass_threshold)

    if not hasattr(env, "_optimized_orbit_link"):
        return None, None, None
    orbit_link = env._optimized_orbit_link

    forces = orbit_link.compute_forces(env_ids, orbit_data, orbit_quats)
    torques = orbit_link.compute_torques(env_ids, orbit_quats)
    return forces, torques, orbit_link.body_indices.cpu().numpy().tolist()


##
# Perturbations
##


class DragGlobal:
    """Last computed drag wrench, shared with the logging code"""

    drag_forces = None
    drag_torques = None
    step = None
    n_affected = None


def apply_perturbations_raycaster(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    apply_drag: bool = True,
    apply_gg: bool = True,
    drag_coefficient: float = 2.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    drag_sensor_name: str = "drag_sensor",
    orbit_data_term_name: str = "orbit_data",
    target_bodies: Optional[list[str]] = None,
    min_mass_threshold: float = 0.0,
):
    """Apply the atmospheric drag (raycasting) and the gravity gradient in a single write to the simulation

    Args:
        env: The environment instance
        env_ids: Environment indices to apply the forces to
        apply_drag: Whether to apply the atmospheric drag
        apply_gg: Whether to apply the gravity gradient
        drag_coefficient: Drag coefficient
        asset_cfg: Asset configuration of the robot
        drag_sensor_name: Name of the raycaster drag sensor
        orbit_data_term_name: Name of the orbit data command term
        target_bodies: Names of the bodies the gravity gradient is applied to
        min_mass_threshold: Minimum body mass for the gravity gradient (used if target_bodies is None)
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    asset: Articulation = env.scene[asset_cfg.name]

    # env id -> (forces, torques) for all bodies
    accumulated_forces_by_env = {}

    if apply_drag:
        drag_forces, drag_torques, affected_drag_env_ids = calculate_atmospheric_drag_forces(
            env=env,
            env_ids=env_ids,
            drag_coefficient=drag_coefficient,
            asset_cfg=asset_cfg,
            drag_sensor_name=drag_sensor_name,
            orbit_data_term_name=orbit_data_term_name,
            return_net_at_root=False,
        )

        if affected_drag_env_ids is not None and len(affected_drag_env_ids) > 0:
            for i, env_id in enumerate(affected_drag_env_ids):
                # The net drag is applied to the root body (body 0)
                env_forces = torch.zeros((asset.num_bodies, 3), device=env.device)
                env_torques = torch.zeros((asset.num_bodies, 3), device=env.device)
                env_forces[0] = drag_forces[i].clone()
                env_torques[0] = drag_torques[i].clone()
                accumulated_forces_by_env[env_id.item()] = (env_forces, env_torques)

        if drag_forces is not None and drag_torques is not None:
            DragGlobal.drag_forces = drag_forces.detach().clone()
            DragGlobal.drag_torques = drag_torques.detach().clone()
            DragGlobal.step = int(env.common_step_counter)
            DragGlobal.n_affected = 0 if affected_drag_env_ids is None else int(len(affected_drag_env_ids))

    if apply_gg:
        gg_forces, gg_torques, body_indices = calculate_gravity_gradient_optimized(
            env=env,
            env_ids=env_ids,
            asset_name=asset_cfg.name,
            target_bodies=target_bodies,
            min_mass_threshold=min_mass_threshold,
        )

        if body_indices:
            for i, env_id in enumerate(env_ids):
                env_id_item = env_id.item()
                if env_id_item not in accumulated_forces_by_env:
                    accumulated_forces_by_env[env_id_item] = (
                        torch.zeros((asset.num_bodies, 3), device=env.device),
                        torch.zeros((asset.num_bodies, 3), device=env.device),
                    )
                forces, torques = accumulated_forces_by_env[env_id_item]

                for j, body_idx in enumerate(body_indices):
                    forces[body_idx] += gg_forces[i, j]
                    torques[body_idx] += gg_torques[i, j]

    if accumulated_forces_by_env:
        env_ids_with_forces = list(accumulated_forces_by_env.keys())
        asset.set_external_force_and_torque(
            forces=torch.stack([accumulated_forces_by_env[e][0] for e in env_ids_with_forces]),
            torques=torch.stack([accumulated_forces_by_env[e][1] for e in env_ids_with_forces]),
            body_ids=None,  # all bodies
            env_ids=torch.tensor(env_ids_with_forces, device=env.device),
        )
        asset.write_data_to_sim()


def apply_perturbations_model(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    apply_drag: bool = True,
    apply_gg: bool = True,
    drag_coefficient: float = 2.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    orbit_data_term_name: str = "orbit_data",
    target_bodies: Optional[list[str]] = None,
    min_mass_threshold: float = 0.0,
):
    """Like apply_perturbations, but with a model of the drag instead of the raycasting sensor

    The drag comes from the neural-network surrogate and are applied to the root body

    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    asset: Articulation = env.scene[asset_cfg.name]

    accumulated_forces = torch.zeros((len(env_ids), asset.num_bodies, 3), device=env.device)
    accumulated_torques = torch.zeros((len(env_ids), asset.num_bodies, 3), device=env.device)

    if apply_drag and env.cfg.use_surrogate_model:
        drag_surrogate = env.cfg.get_drag_surrogate()
        if drag_surrogate is not None:
            drag_forces, drag_torques, _ = calculate_atmospheric_drag_forces_surrogate(
                env=env,
                env_ids=env_ids,
                surrogate_model=drag_surrogate,
                asset_cfg=asset_cfg,
                orbit_data_term_name=orbit_data_term_name,
            )
            accumulated_forces[:, 0, :] = drag_forces
            accumulated_torques[:, 0, :] = drag_torques
            DragGlobal.drag_forces = drag_forces
            DragGlobal.drag_torques = drag_torques
            DragGlobal.step = int(env.common_step_counter)
            DragGlobal.n_affected = len(env_ids)
        else:
            print("Surrogate model not available, skipping drag forces")
    elif apply_drag and not getattr(env, "_warned_no_drag", False):
        print("WARNING: apply_perturbations_model needs env.cfg.use_surrogate_model = True, no drag is applied")
        env._warned_no_drag = True
        
    if apply_gg:
        gg_forces, gg_torques, body_indices = calculate_gravity_gradient_optimized(
            env=env,
            env_ids=env_ids,
            asset_name=asset_cfg.name,
            target_bodies=target_bodies,
            min_mass_threshold=min_mass_threshold,
        )
        if body_indices:
            for i, body_idx in enumerate(body_indices):
                accumulated_forces[:, body_idx, :] += gg_forces[:, i, :]
                accumulated_torques[:, body_idx, :] += gg_torques[:, i, :]

    # Only write the environments with a significant force
    force_magnitudes = torch.sum(torch.norm(accumulated_forces, dim=2), dim=1)
    significant_forces_mask = force_magnitudes > 1e-16
    if torch.any(significant_forces_mask):
        final_env_indices = torch.nonzero(significant_forces_mask, as_tuple=True)[0]
        asset.set_external_force_and_torque(
            forces=accumulated_forces[final_env_indices],
            torques=accumulated_torques[final_env_indices],
            body_ids=None,  # all bodies
            env_ids=env_ids[final_env_indices],
        )
        asset.write_data_to_sim()


##
# Drag surrogate
##


class DragForceNN(nn.Module):
    """Feed-forward network of the drag surrogate

    Must match the trained architecture exactly
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 use_norm=False):
        super().__init__()
        self.__name__ = "DragForceNN"

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList() if use_norm else None

        self.layers.append(nn.Linear(input_size, hidden_size))
        if use_norm:
            self.norms.append(nn.LayerNorm(hidden_size))

        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            if use_norm:
                self.norms.append(nn.LayerNorm(hidden_size))

        self.output_layer = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.norms is not None:
                x = self.norms[i](x)
            x = self.activation(x)
        return self.output_layer(x)


class DragForceSurrogate:
    """Wrench surrogate with the scaling constants carried in the checkpoint.
    """

    def __init__(self, model_path, device=None, scalers_path=None,
                 fold_scalers=True, use_cuda_graph=False, validate_steps=8,
                 clamp_outputs=True, expected_fingerprint=None,
                 allow_tf32=True, verbose=True):
        self.device = (torch.device(device) if device is not None else
                       torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self._cuda = self.device.type == "cuda"

        if self._cuda and allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        fmt = ckpt.get("format_version", None)
        if fmt != "v3":
            # LEGACY PATH: a grid-search .pt (model_state_dict + hyperparameters)
            # plus the feature_scalers*.pt written by the training notebook. No
            # retraining and no re-export needed -- the weights are read as they
            # are and the scaling constants come from the scalers file
            ckpt = self._adapt_legacy(ckpt, model_path, scalers_path, verbose)
        if expected_fingerprint and ckpt.get("fingerprint") != expected_fingerprint:
            raise ValueError(f"fingerprint mismatch: file has "
                             f"{ckpt.get('fingerprint')}, expected {expected_fingerprint}")

        self.fingerprint = ckpt.get("fingerprint", "unknown")
        self.input_size = int(ckpt["input_size"])
        self.hidden_size = int(ckpt["hidden_size"])
        self.num_layers = int(ckpt["num_layers"])
        self.output_size = int(ckpt["output_size"])
        self.use_norm = bool(ckpt["use_norm"])

        net = DragForceNN(self.input_size, self.hidden_size, self.num_layers,
                          self.output_size, self.use_norm)
        net.load_state_dict(ckpt["model_state_dict"])
        net.eval()

        in_mean = ckpt["in_mean"].double()
        in_scale = ckpt["in_scale"].double()
        out_mean = ckpt["out_mean"].double()
        out_scale = ckpt["out_scale"].double()

        self.log_columns = list(ckpt.get("log_columns", []))
        self.log_floor = float(ckpt.get("log_floor", 1e-38))
        self.canonicalize_quat = bool(ckpt.get("canonicalize_quat", True))
        qc = ckpt.get("quat_columns", {"spacecraft": [0, 1, 2, 3],
                                       "orbit_lvlh": [8, 9, 10, 11]})
        self._quat_slices = [(int(v[0]), int(v[-1]) + 1) for v in qc.values()]

        self._has_log = len(self.log_columns) > 0
        if self._has_log:
            m = torch.zeros(self.input_size, dtype=torch.bool)
            m[torch.as_tensor(self.log_columns, dtype=torch.long)] = True
            self.log_mask = m.to(self.device)
        else:
            self.log_mask = None

        self.folded = False
        self._fold_check = None
        if fold_scalers:
            ref = DragForceNN(self.input_size, self.hidden_size, self.num_layers,
                              self.output_size, self.use_norm)
            ref.load_state_dict(ckpt["model_state_dict"])
            ref.eval()

            with torch.no_grad():
                W1 = net.layers[0].weight.data.double()          # [H, D]
                b1 = net.layers[0].bias.data.double()            # [H]
                net.layers[0].weight.data = (W1 / in_scale.unsqueeze(0)).float()
                net.layers[0].bias.data = (b1 - W1 @ (in_mean / in_scale)).float()

                Wo = net.output_layer.weight.data.double()       # [O, H]
                bo = net.output_layer.bias.data.double()         # [O]
                net.output_layer.weight.data = (out_scale.unsqueeze(1) * Wo).float()
                net.output_layer.bias.data = (out_scale * bo + out_mean).float()

            # Folding must not change the answer beyond float32 round-off
            with torch.no_grad():
                probe = torch.randn(512, self.input_size)
                if self._has_log:
                    probe = probe.abs() + 1e-12
                p = probe.clone()
                if self._has_log:
                    idx = torch.as_tensor(self.log_columns, dtype=torch.long)
                    p[:, idx] = torch.log10(p[:, idx].clamp_min(self.log_floor))
                unfolded = (ref((p - in_mean.float()) / in_scale.float())
                            * out_scale.float() + out_mean.float())
                folded = net(p)
                den = unfolded.abs().clamp_min(
                    unfolded.abs().median().clamp_min(1e-30))
                rel = float(((folded - unfolded).abs() / den).max())
            if rel > 1e-3:
                raise AssertionError(
                    f"scaler folding changed the output (max rel {rel:.3e}). "
                    "Pass fold_scalers=False and report this.")
            self._fold_check = rel
            self.folded = True
        else:
            self.in_mean = in_mean.float().to(self.device)
            self.in_scale_inv = (1.0 / in_scale).float().to(self.device)
            self.out_mean = out_mean.float().to(self.device)
            self.out_scale = out_scale.float().to(self.device)

        self.model = net.to(self.device).eval()
        for p_ in self.model.parameters():
            p_.requires_grad_(False)

        self.clamp_outputs = clamp_outputs
        cl = ckpt.get("clamp", {})
        self.clamp_f = float(cl.get("force_N", float("inf")))
        self.clamp_t = float(cl.get("torque_Nm", float("inf")))
        if clamp_outputs:
            lim = torch.empty(self.output_size, dtype=torch.float32)
            lim[:3] = self.clamp_f
            lim[3:] = self.clamp_t
            self._clamp_hi = lim.to(self.device)
            self._clamp_lo = -self._clamp_hi
        self.envelope = ckpt.get("envelope", {})

        self.validate_steps = int(validate_steps)
        self._n_calls = 0
        self._warned = set()
        self.use_cuda_graph = bool(use_cuda_graph and self._cuda)
        # Batch size is set by --num_envs and can differ between runs, and reset
        # paths may call with a subset. Keyed caches avoid rebuilding a graph
        # every time the size changes; MAX_CACHED bounds the memory
        self.MAX_CACHED = 4
        self._graphs = {}        # batch -> (graph, static_in, static_out)
        self._feat_bufs = {}     # batch -> [batch, D] float32

        if verbose:
            print(f"[DragForceSurrogate] {model_path}")
            print(f"  fingerprint {self.fingerprint}   {self.num_layers}L x "
                  f"{self.hidden_size}H  norm={self.use_norm}  device={self.device}")
            print(f"  folded={self.folded}"
                  + (f" (max rel dev {self._fold_check:.2e})" if self.folded else "")
                  + f"  cuda_graph={self.use_cuda_graph}"
                  + f"  log_columns={self.log_columns}")
            print("  UNITS: position km, velocity km/s, rho kg/m^3, booms 0-1")

    @staticmethod
    def _resolve_scalers_path(model_path, scalers_path):
        """Explicit path if given, else look beside and above the checkpoint.

        Grid-search checkpoints live in <dataset>/grid_search_results_*/ while
        feature_scalers*.pt is written to <dataset>/, so one or two levels up is
        the normal case.
        """
        if scalers_path is not None:
            if not os.path.isfile(scalers_path):
                raise FileNotFoundError(f"scalers_path not found: {scalers_path}")
            return scalers_path
        d = os.path.dirname(os.path.abspath(model_path))
        for _ in range(3):
            hits = sorted(glob.glob(os.path.join(d, "feature_scalers*.pt")))
            if hits:
                return hits[0]
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
        return None

    @staticmethod
    def _strip_wrapper_prefix(state):
        """A pickled ScaledDragModel stores keys as 'net.layers.0.weight' and
        carries the scaler buffers alongside. Keep only the network."""
        net = {k[len("net."):]: v for k, v in state.items() if k.startswith("net.")}
        return net if net else state

    @staticmethod
    def _adapt_legacy(ckpt, model_path, scalers_path, verbose=True):
        """Build a dict from a grid-search .pt + a scalers .pt.

        Nothing is retrained and no weight is modified: the state dict is copied
        through and the scaling constants are read from the scalers file.
        Envelope and clamp are absent, so envelope warnings and output
        saturation are disabled for a legacy load -- that information simply is
        not in these files. Prefer export_surrogate(), which records it.
        """
        state = ckpt.get("model_state_dict", ckpt.get("state_dict"))
        if state is None:
            raise ValueError(
                f"{model_path} is neither a checkpoint nor a grid-search "
                "checkpoint (no 'model_state_dict' or 'state_dict').")
        state = DragForceSurrogate._strip_wrapper_prefix(state)

        src = ckpt
        if "input_scaler" not in ckpt or "output_scaler" not in ckpt:
            resolved = DragForceSurrogate._resolve_scalers_path(
                model_path, scalers_path)
            if resolved is None:
                raise ValueError(
                    f"{model_path} carries no scaling constants (it is a bare "
                    "grid-search checkpoint) and no feature_scalers*.pt was "
                    "found beside or above it. Pass scalers_path=..., or point "
                    "model_path at a export. The transform cannot be "
                    "guessed: the constants are fitted on the training split.")
            src = torch.load(resolved, map_location="cpu", weights_only=False)
            if verbose:
                print(f"[DragForceSurrogate] scaling constants from {resolved}")
        try:
            ins, outs = src["input_scaler"], src["output_scaler"]
        except (KeyError, TypeError):
            raise ValueError("no 'input_scaler' / 'output_scaler' entries found")

        mode = outs.get("mode", src.get("output_scaler_mode", "zscore"))
        if mode not in ("zscore", None):
            raise ValueError(
                f"output scaler mode is {mode!r}. This class implements the "
                "plain-scaling surrogate only; a dynamic-pressure model is not "
                "supported.")

        hp = ckpt.get("hyperparameters", ckpt.get("net_hyperparameters", {})) or {}
        n_layers = hp.get("num_layers", len([k for k in state
                                             if k.startswith("layers.")
                                             and k.endswith(".weight")]))
        out = {
            "format_version": "v3",
            "parameterisation": "direct wrench (legacy load)",
            "input_size": int(state["layers.0.weight"].shape[1]),
            "hidden_size": int(hp.get("hidden_size", state["layers.0.weight"].shape[0])),
            "num_layers": int(n_layers),
            "output_size": int(state["output_layer.weight"].shape[0]),
            "use_norm": bool(hp.get("use_norm",
                                    any(k.startswith("norms.") for k in state))),
            "model_state_dict": state,
            "in_mean": ins["mean"], "in_scale": ins["scale"],
            "out_mean": outs["mean"], "out_scale": outs["scale"],
            "log_columns": list(ins.get("log_columns", [])),
            "log_floor": float(ins.get("log_floor", 1e-38)),
            "skip_columns": list(ins.get("skip_columns", [])),
            "canonicalize_quat": True,
            "envelope": {},          # unknown -> envelope checks skipped
            "clamp": {},             # unknown -> saturation disabled
            "fingerprint": "legacy",
            "known_limitations": [
                "Loaded from a legacy checkpoint. No training envelope and no "
                "clamp bounds are recorded, so envelope warnings and output "
                "saturation are DISABLED.",
            ],
        }
        if int(ins["mean"].shape[0]) != out["input_size"]:
            raise ValueError(f"scaler has {int(ins['mean'].shape[0])} input "
                             f"columns, network takes {out['input_size']} -- "
                             "wrong scalers file for this model")
        if int(outs["mean"].shape[0]) != out["output_size"]:
            raise ValueError(f"scaler has {int(outs['mean'].shape[0])} output "
                             f"columns, network emits {out['output_size']}")

        # Column order must match training, or every feature is misread
        expected = ["spacecraft_orientation(4)", "boom_extensions(4)",
                    "orbit_orientation(4)", "orbit_position(3)",
                    "orbit_velocity(3)", "atm_angular_velocity(3)",
                    "air_density(1)"]
        spec = src.get("input_features", ckpt.get("input_features", None))
        if spec is not None and list(spec) != expected:
            raise ValueError("input feature order in the checkpoint differs "
                             f"from the order assembled here:\n  checkpoint: "
                             f"{list(spec)}\n  events.py : {expected}")

        if verbose:
            print("[DragForceSurrogate] LEGACY load: envelope and clamp are off")
        return out

    def _warn(self, key, msg):
        if key not in self._warned:
            self._warned.add(key)
            print(f"[DragForceSurrogate] WARNING ({key}): {msg} (reported once)")

    def feature_buffer(self, num_envs):
        """Persistent [num_envs, D] float32 tensor to assemble features into.

        Use with `torch.cat(parts, dim=1, out=buf)` so the step allocates
        nothing. One buffer is kept per distinct batch size, up to MAX_CACHED.
        """
        num_envs = int(num_envs)
        buf = self._feat_bufs.get(num_envs)
        if buf is None:
            if len(self._feat_bufs) >= self.MAX_CACHED:
                self._feat_bufs.pop(next(iter(self._feat_bufs)))
            buf = torch.empty(num_envs, self.input_size,
                              device=self.device, dtype=torch.float32)
            self._feat_bufs[num_envs] = buf
        return buf

    def _preprocess_(self, x):
        """In place on the caller's buffer. Small kernels, no sync."""
        if self.canonicalize_quat:
            # q and -q are the same rotation but different network inputs, and
            # the training data stores w >= 0. The RL reset distribution
            # produces w < 0 in a substantial fraction of resets
            for lo, hi in self._quat_slices:
                sgn = torch.where(x[:, lo:lo + 1] < 0,
                                  torch.full_like(x[:, lo:lo + 1], -1.0),
                                  torch.full_like(x[:, lo:lo + 1], 1.0))
                x[:, lo:hi].mul_(sgn)
        if self._has_log:
            x.copy_(torch.where(self.log_mask,
                                torch.log10(x.clamp_min(self.log_floor)), x))
        return x

    def _core(self, x):
        if self.folded:
            y = self.model(x)
        else:
            y = (self.model((x - self.in_mean) * self.in_scale_inv)
                 * self.out_scale + self.out_mean)
        if self.clamp_outputs:
            y = torch.clamp(y, self._clamp_lo, self._clamp_hi)
        return y

    def _get_graph(self, num_envs):
        """Capture (once per batch size) and return (graph, static_in, static_out)."""
        num_envs = int(num_envs)
        got = self._graphs.get(num_envs)
        if got is not None:
            return got

        if len(self._graphs) >= self.MAX_CACHED:
            self._graphs.pop(next(iter(self._graphs)))

        g_in = torch.zeros(num_envs, self.input_size,
                           device=self.device, dtype=torch.float32)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.inference_mode():
                for _ in range(3):
                    self._core(g_in)
        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with torch.inference_mode():
                g_out = self._core(g_in)

        self._graphs[num_envs] = (graph, g_in, g_out)
        return self._graphs[num_envs]

    def warmup(self, num_envs=None, iters=5):
        """Allocate buffers, warm the kernels and capture the graph.

        `num_envs` is optional: it is only known once the environment exists,
        and it comes from --num_envs at runtime. Calling warmup() with no
        argument is a no-op, and the first predict_forces_torques() will set
        everything up lazily at the cost of one slow step.
        """
        if num_envs is None:
            return self
        num_envs = int(num_envs)
        buf = self.feature_buffer(num_envs)
        buf.normal_().abs_().add_(1e-9)
        with torch.inference_mode():
            for _ in range(iters):
                self._core(buf)
        if self.use_cuda_graph:
            self._get_graph(num_envs)
        if self._cuda:
            torch.cuda.synchronize()
        return self

    @torch.inference_mode()
    def predict_forces_torques(self, features, env=None, preprocessed=False,
                               out=None):
        """features: [B, D] raw physical units, on self.device.

        Returns (forces [B, 3], torques [B, 3]) in N and N*m, inertial frame.

        `features` is modified in place by the quaternion canonicalisation and
        the log columns unless `preprocessed=True`.

        With use_cuda_graph and out=None the returned tensors ALIAS a static
        buffer that the next call overwrites. Pass `out=` (the applier does) or
        clone if the values must outlive the step.
        """
        if not isinstance(features, torch.Tensor):
            features = torch.as_tensor(features, dtype=torch.float32,
                                       device=self.device)
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if features.shape[1] != self.input_size:
            raise ValueError(f"expected {self.input_size} features, "
                             f"got {features.shape[1]}")
        if features.device != self.device:
            features = features.to(self.device, non_blocking=True)
        if features.dtype != torch.float32:
            features = features.float()

        self._n_calls += 1
        # Validation SYNCS. Confined to the first few calls by design
        if self._n_calls <= self.validate_steps:
            self._validate(features)

        if not preprocessed:
            self._preprocess_(features)

        if self.use_cuda_graph:
            graph, g_in, g_out = self._get_graph(features.shape[0])
            g_in.copy_(features, non_blocking=True)
            graph.replay()
            y = g_out
        else:
            y = self._core(features)

        if out is not None:
            out.copy_(y, non_blocking=True)
            y = out
        return y[:, :3], y[:, 3:]

    def _validate(self, x):
        """Envelope and finiteness checks. Synchronises -- not for the hot path."""
        if not bool(torch.isfinite(x).all()):
            self._warn("nonfinite_input",
                       "NaN/Inf in the raw features. They are NOT sanitised on "
                       "the hot path; fix the source. A NaN here propagates into "
                       "the applied wrench and then into the policy.")
        e = self.envelope
        if not e:
            return
        v = float(x[:, 15:18].norm(dim=1).median())
        lo, hi = e.get("speed_km_s", [0.0, 1e9])
        if v < 0.5 * lo or v > 2.0 * hi:
            self._warn("units_velocity",
                       f"|v| median {v:.3f}, training range [{lo:.3f}, {hi:.3f}] "
                       "km/s. A factor ~1000 means the environment supplies m/s.")
        r = float(x[:, 12:15].norm(dim=1).median())
        lo, hi = e.get("radius_km", [0.0, 1e9])
        if r < 0.5 * lo or r > 2.0 * hi:
            self._warn("units_position",
                       f"|r| median {r:.1f}, training range [{lo:.1f}, {hi:.1f}] km.")
        lo, hi = e.get("rho_kg_m3", [0.0, 1e9])
        frac = float(((x[:, 21] < lo) | (x[:, 21] > hi)).float().mean())
        if frac > 0.05:
            self._warn("rho_envelope",
                       f"{100*frac:.1f}% of samples have rho outside "
                       f"[{lo:.3e}, {hi:.3e}] kg/m^3. The wrench scales with rho, "
                       "so input and target are both extrapolated.")
        b = x[:, 4:8]
        if float(b.min()) < -0.05 or float(b.max()) > 1.05:
            self._warn("boom_range",
                       f"boom extensions in [{float(b.min()):.3f}, "
                       f"{float(b.max()):.3f}], expected 0-1 normalised.")


class DragWrenchApplier:
    """Per-step drag wrench from the surrogate for all environments, allocation- and sync-free

    Build once after the scene creation, then call ``compute()`` every step
    """

    def __init__(self, env, surrogate, asset_cfg=None, orbit_data_term_name="orbit_data", return_world_frame=False):
        self.env = env
        self.sg = surrogate
        self.asset_cfg = asset_cfg if asset_cfg is not None else SceneEntityCfg("robot")
        self.term = orbit_data_term_name
        self.world = return_world_frame

        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self.num_envs = env.num_envs
        self.device = env.device

        # The boom extension is read from the master joints. Joint indices are resolved once
        self.master_joint_idx = torch.as_tensor(
            [self.asset.joint_names.index(f"d3_boom_{b + 1}_joint") for b in range(NUM_BOOMS)],
            device=self.device,
            dtype=torch.long,
        )

        # Persistent buffers
        self.feat = surrogate.feature_buffer(self.num_envs)
        self.booms = torch.empty(self.num_envs, NUM_BOOMS, device=self.device, dtype=torch.float32)
        self.rho = torch.empty(self.num_envs, 1, device=self.device, dtype=torch.float32)
        self.wrench = torch.empty(self.num_envs, surrogate.output_size, device=self.device, dtype=torch.float32)
        surrogate.warmup(self.num_envs)

    @torch.inference_mode()
    def compute(self):
        """Returns (forces, torques) for all environments, [E, 3] each"""
        a = self.asset
        od = self.env.command_manager.get_command(self.term)

        rho = od["air_density"]
        self.rho.copy_(rho.unsqueeze(1) if rho.dim() == 1 else rho[:, :1])

        # Boom extension normalized to [0, 1]
        torch.div(a.data.joint_pos[:, self.master_joint_idx], MAX_BOOM_LENGTH, out=self.booms)
        self.booms.clamp_(0.0, 1.0)

        # One kernel, writing straight into the persistent buffer
        torch.cat(
            [
                a.data.root_quat_w,  # 0:4    wxyz
                self.booms,  # 4:8    normalized 0-1
                od["orientation"],  # 8:12   LVLH orientation, wxyz
                od["position"],  # 12:15  km
                od["velocity"],  # 15:18  km/s
                od["atm_ang_velocity"],  # 18:21  rad/s
                self.rho,  # 21     kg/m^3
            ],
            dim=1,
            out=self.feat,
        )

        self.sg.predict_forces_torques(self.feat, out=self.wrench)
        wf, wt = self.wrench[:, :3], self.wrench[:, 3:]
        if self.world:
            return wf, wt

        # Inertial -> root body frame. For a unit quaternion the inverse is the conjugate
        q = a.data.body_quat_w[:, 0]
        qc = torch.cat([q[:, :1], -q[:, 1:]], dim=1)
        return math_utils.quat_apply(qc, wf), math_utils.quat_apply(qc, wt)


def calculate_atmospheric_drag_forces_surrogate(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    surrogate_model: DragForceSurrogate,
    asset_cfg: SceneEntityCfg = None,
    orbit_data_term_name: str = "orbit_data",
    return_world_frame: bool = False,
):
    """Drag wrench from the surrogate, for every environment in one forward pass

    """
    asset_cfg = asset_cfg if asset_cfg is not None else SceneEntityCfg("robot")

    ap = getattr(env, "_drag_applier", None)
    if ap is None or ap.sg is not surrogate_model or ap.num_envs != env.num_envs:
        ap = DragWrenchApplier(env, surrogate_model, asset_cfg, orbit_data_term_name, return_world_frame)
        env._drag_applier = ap
    ap.world = return_world_frame

    forces, torques = ap.compute()

    if env_ids is None:
        return forces, torques, torch.arange(env.num_envs, device=env.device)
    # Computing all environments and indexing is faster than gathering the inputs first:
    # the cost is dominated by the kernel launches, not by the batch size
    return forces[env_ids], torques[env_ids], env_ids


##
# Boom segments
##


def follow_boom_segments(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Make the segment joints of each boom follow its master joint

    Interval event (interval_range_s=(0.0, 0.0)), run at every step. Reads the master prismatic
    joint of each boom (d3_boom_{b}_joint) and sets its 34 segment joints so that their summed
    extension equals the master extension: full segments first, then the remainder
    """
    asset = env.scene[asset_cfg.name]
    device = env.device

    NUM_LINKS = NUM_LINKS_PER_BOOM
    LINK_EXT = MAX_BOOM_LENGTH / NUM_LINKS  # ~0.1088 m

    batch = env_ids.shape[0]
    joint_names = asset.joint_names

    master_indices = [joint_names.index(f"d3_boom_{b}_joint") for b in range(1, NUM_BOOMS + 1)]
    segment_indices = [
        [joint_names.index(f"d3_boom_{b}_p{j}_joint") for j in range(1, NUM_LINKS + 1)]
        for b in range(1, NUM_BOOMS + 1)
    ]

    write_pos = asset.data.joint_pos[env_ids].clone()
    write_vel = asset.data.joint_vel[env_ids].clone()

    for b in range(NUM_BOOMS):
        master_pos = asset.data.joint_pos[env_ids, master_indices[b]].clamp(0.0, MAX_BOOM_LENGTH)
        master_vel = asset.data.joint_vel[env_ids, master_indices[b]]

        ratios = master_pos / LINK_EXT
        num_full = torch.floor(ratios).long().clamp(0, NUM_LINKS)  # number of fully extended segments
        partial = (ratios - num_full.float()) * LINK_EXT  # remaining length

        seg_pos = torch.zeros((batch, NUM_LINKS), device=device)
        seg_vel = torch.zeros_like(seg_pos)
        for i in range(batch):
            nf = num_full[i].item()
            seg_pos[i, :nf] = LINK_EXT
            seg_vel[i, :nf] = master_vel[i]
            if nf < NUM_LINKS and partial[i] > 1e-6:
                seg_pos[i, nf] = partial[i]
                seg_vel[i, nf] = master_vel[i]

        write_pos[:, segment_indices[b]] = seg_pos
        write_vel[:, segment_indices[b]] = seg_vel

    # The master joint entries are preserved
    asset.write_joint_state_to_sim(write_pos, write_vel, env_ids=env_ids)


##
# Diagnostics
##


class ControllerDiagnostics:
    """Logs the boom controller and the spacecraft motion to Weights & Biases"""

    def __init__(self, env, log_frequency=10):
        import wandb

        self.wandb = wandb
        self.env = env
        self.device = env.device
        self.log_frequency = log_frequency
        self.num_envs = env.num_envs
        self.robot = env.scene["robot"]

        # Indices of the master joints
        joint_names = self.robot.joint_names
        self.master_indices = []
        for b in range(1, NUM_BOOMS + 1):
            name = f"d3_boom_{b}_joint"
            if name not in joint_names:
                raise KeyError(f"Master joint '{name}' not found in robot.joint_names")
            self.master_indices.append(joint_names.index(name))

        wandb.init(
            project="d3_boom_controller_debug",
            config={
                "num_envs": self.num_envs,
                "episode_length": env.max_episode_length,
                "sim_dt": env.step_dt,
                "controller_type": "position_implicit",
            },
        )

    def log(self, step: int):
        if step % self.log_frequency != 0:
            return

        action_term = self.env.action_manager._terms["boom_extension"]

        joint_actuals = self.robot.data.joint_pos.detach().cpu()
        joint_velocities = self.robot.data.joint_vel.detach().cpu()

        # Commanded boom extensions, and the same targets laid out over all joints
        high_level_targets = action_term.target_boom_extensions.detach().cpu()
        high_level_actuals = joint_actuals[:, self.master_indices]
        full_target = torch.zeros_like(joint_actuals)
        for i, master_idx in enumerate(self.master_indices):
            full_target[:, master_idx] = high_level_targets[:, i]

        joint_tracking_error = (full_target - joint_actuals).abs().mean().item()
        boom_tracking_error = (high_level_targets - high_level_actuals).abs().mean().item()

        # Spacecraft motion and error w.r.t. the target pose
        root_pos = self.robot.data.root_pos_w[:, :3].detach().cpu()
        root_vel_lin = self.robot.data.root_lin_vel_b.detach().cpu()
        root_vel_ang = self.robot.data.root_ang_vel_b.detach().cpu()
        root_quat = self.robot.data.root_quat_w.detach().cpu()

        cmd = self.env.command_manager.get_command("target_pose")
        pos_error = torch.norm(root_pos - cmd["position"].cpu(), dim=-1)

        # Orientation drift: angle between the body and the target quaternions
        root_quat = root_quat / torch.norm(root_quat, dim=-1, keepdim=True)
        goal_rot = cmd["orientation"].detach().cpu()
        goal_rot = goal_rot / torch.norm(goal_rot, dim=-1, keepdim=True)
        dot_goal = (goal_rot * root_quat).sum(dim=-1).abs()
        orient_drift = 2.0 * torch.acos(torch.clamp(dot_goal, 0.0, 1.0))

        print(f"[step {step}] Orient drift (rad): {orient_drift[0].item():.6f}")

        self.wandb.log(
            {
                "metrics/pos_error_mean": pos_error.mean().item(),
                "metrics/root_lin_vel_mean": root_vel_lin.norm(dim=-1).mean().item(),
                "metrics/root_ang_vel_mean": root_vel_ang.norm(dim=-1).mean().item(),
                "metrics/joint_vel_mean": joint_velocities.abs().mean().item(),
                "metrics/joint_tracking_error_mean": joint_tracking_error,
                "metrics/boom_tracking_error_mean": boom_tracking_error,
            },
            step=step,
        )

        # Environment 0: target, actual and velocity of the first joints
        env_idx = 0
        for j in range(min(25, joint_actuals.shape[1])):
            self.wandb.log(
                {
                    f"joints/env0/joint_{j}/target": full_target[env_idx, j].item(),
                    f"joints/env0/joint_{j}/actual": joint_actuals[env_idx, j].item(),
                    f"joints/env0/joint_{j}/vel": joint_velocities[env_idx, j].item(),
                },
                step=step,
            )

        # Environment 0: target vs actual extension of each boom
        for b in range(NUM_BOOMS):
            self.wandb.log(
                {
                    f"booms/env0/boom_{b}/target_extension": high_level_targets[env_idx, b].item(),
                    f"booms/env0/boom_{b}/actual_extension": high_level_actuals[env_idx, b].item(),
                },
                step=step,
            )

        self.wandb.log({"root/env0/orient_drift": orient_drift[0].item()}, step=step)