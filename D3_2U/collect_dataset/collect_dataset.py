#!/usr/bin/env python3

"""
Data collection for the drag-force surrogate model

The data collection is structured as follows:
- Episodes: different starting points in the orbit, evenly spaced over the orbit data file
  (e.g. every 30 min across a 60 h file)
- Orientations: for each episode, sample multiple spacecraft orientations
- Boom configurations: for each orientation, test multiple boom deployment patterns
- Stabilization: allow the physics to stabilize between configuration changes

Random attitudes are uniform on SO(3) (Haar measure) by default. Uniform Euler angles do cover
SO(3) but over-weight the gimbal-lock region and under-sample small rotations (0.49% below 30 deg
vs 0.76% for Haar), which is the near-nadir region where the station-keeping policy operates

Quaternions are stored in the w >= 0 hemisphere, for both the spacecraft and the LVLH quaternion.
q and -q are the same rotation but different network inputs. The same canonicalization must be
applied when the surrogate is queried during RL

Besides the dataset, each sample stores diagnostics (commanded vs measured attitude and boom
extensions, angular velocity, ray hits, orbit time), see metadata["diagnostic_columns"]

Usage (120 orbital windows, 60 attitudes, 32 boom configs -> 230400 samples):
    ./isaaclab.sh -p collect_dataset.py \
        --num_episodes 120 \
        --orientations_per_episode 60 \
        --configs_per_orientation 32 \
        --stabilization_steps 5 \
        --output_path /path/to/drag_dataset.pt \
        --headless --enable_cameras
"""

import argparse
import glob
import math
import os
import random
import time

import numpy as np
import torch
from tqdm import tqdm

# Import AppLauncher first to check for import errors early
try:
    from isaaclab.app import AppLauncher
except ImportError as e:
    print(f"Error importing AppLauncher: {e}")
    exit(1)

parser = argparse.ArgumentParser(description="Collect drag force data for surrogate model")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to run (orbit starting points)")
parser.add_argument("--orientations_per_episode", type=int, default=5, help="Number of spacecraft orientations to sample per episode")
parser.add_argument("--configs_per_orientation", type=int, default=4, help="Number of boom configurations to test per orientation")
parser.add_argument("--stabilization_steps", type=int, default=5, help="Number of steps to wait for physics stabilization")
parser.add_argument("--num_envs", type=int, default=5, help="Number of parallel environments")
parser.add_argument("--output_path", type=str, default="drag_dataset.pt", help="Output file path")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--orbit_data_path", type=str, default=None, help="Path to orbit data file (overrides default)")
parser.add_argument("--debug", action="store_true", help="Enable debug output")
parser.add_argument("--attitude_sampling", type=str, default="haar", choices=["haar", "euler"],
                    help="haar: uniform on SO(3) (recommended). euler: uniform Euler angles in [-pi,pi]^3")
parser.add_argument("--random_attitude_frame", type=str, default="inertial", choices=["inertial", "lvlh"],
                    help="inertial: random attitude is absolute. lvlh: random attitude is composed with the LVLH target quaternion")
parser.add_argument("--canonicalize_quat", type=int, default=1,
                    help="1: force w>=0 on spacecraft and LVLH quaternions. MUST match the RL inference path")
parser.add_argument("--merge_episodes", type=int, default=0, help="Merge the per-episode files into a single dataset at the end")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# The app must be launched before importing the Isaac Lab task modules
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

try:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_tasks.manager_based.my.D3_2U.d3_2U_env_cfg import CubeSatEnvCfg
    import isaaclab.utils.math as math_utils
    from isaaclab_tasks.manager_based.my.D3_2U.mdp.events import calculate_atmospheric_drag_forces, create_mesh_to_body_mapping
    from isaaclab.managers import SceneEntityCfg
except ImportError as e:
    print(f"Error importing required modules: {e}")
    simulation_app.close()
    exit(1)

NUM_BOOMS = 4
NUM_LINKS_PER_BOOM = 34
MAX_BOOM_LENGTH = 3.7  # m

os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

print("Setting up environment configuration...")
env_cfg = CubeSatEnvCfg()
env_cfg.scene.num_envs = args.num_envs
# A collection episode needs thousands of steps. A short episode length would trigger an
# automatic reset in the middle of the stabilization loop and corrupt the commanded state
env_cfg.episode_length_s = 99999

if args.orbit_data_path:
    env_cfg.orbit_data_path = args.orbit_data_path
    print(f"Using custom orbit data: {env_cfg.orbit_data_path}")

print("Creating environment...")
env = None
try:
    env = ManagerBasedRLEnv(cfg=env_cfg)
except Exception as e:
    print(f"Error creating environment: {e}")
    simulation_app.close()
    exit(1)

print("Environment created successfully!")

inputs = []
outputs = []
diagnostics = []
total_samples = 0

metadata = {
    "collection_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    "script_version": "v8",
    "num_samples": 0,
    "num_episodes": args.num_episodes,
    "orientations_per_episode": args.orientations_per_episode,
    "configs_per_orientation": args.configs_per_orientation,
    "stabilization_steps": args.stabilization_steps,
    "input_features": [
        "spacecraft_orientation(4)",
        "boom_extensions(4)",
        "orbit_orientation(4)",
        "orbit_position(3)",
        "orbit_velocity(3)",
        "atm_angular_velocity(3)",
        "air_density(1)"
    ],
    "output_features": [
        "net_force(3)",
        "net_torque(3)"
    ],
    "attitude_sampling": args.attitude_sampling,
    "attitude_support": "full SO(3)",
    "random_attitude_frame": args.random_attitude_frame,
    "quaternion_canonicalized_w_positive": bool(args.canonicalize_quat),
    "output_frame": "inertial (world)",
    "diagnostic_columns": [
        "commanded_quat_w", "commanded_quat_x", "commanded_quat_y", "commanded_quat_z",     # 0:4
        "measured_quat_raw_w", "measured_quat_raw_x", "measured_quat_raw_y", "measured_quat_raw_z",  # 4:8
        "attitude_drift_deg",                                                                # 8
        "commanded_boom_1", "commanded_boom_2", "commanded_boom_3", "commanded_boom_4",      # 9:13
        "measured_boom_segments_1", "measured_boom_segments_2",
        "measured_boom_segments_3", "measured_boom_segments_4",                              # 13:17
        "measured_boom_master_1", "measured_boom_master_2",
        "measured_boom_master_3", "measured_boom_master_4",                                  # 17:21
        "root_ang_vel_x", "root_ang_vel_y", "root_ang_vel_z",                                # 21:24
        "num_ray_hits",                                                                      # 24
        "orbit_time_s",                                                                      # 25
    ],
    "orbit_data_path": env_cfg.orbit_data_path,
    "seed": args.seed
}

robot = env.scene["robot"]
raycaster = env.scene["drag_sensor"]
if not hasattr(env, "_mesh_id_to_body_map"):
    try:
        print("Initializing mesh to body mapping...")
        env._mesh_id_to_body_map = create_mesh_to_body_mapping(robot, raycaster)
        print(f"Created mapping for {len(env._mesh_id_to_body_map)} meshes")
    except Exception as e:
        print(f"Error creating mesh mapping: {e}")
        env._mesh_id_to_body_map = {0: 0}  # map the first mesh to the first body as fallback

print("Starting data collection...")
device = env.device

orbit_data_command = env.command_manager._terms["orbit_data"]
orbit_manager = orbit_data_command.orbit_manager
total_time_range = orbit_manager.time_steps[-1].item() - orbit_manager.time_steps[0].item()
print(f"Orbit data spans {total_time_range:.2f}s, episodes will be spaced ~{total_time_range / args.num_episodes:.2f}s apart")


# -----------------------------------------------------------------------------
# Quaternion helpers
# -----------------------------------------------------------------------------
def canon_quat(q: torch.Tensor) -> torch.Tensor:
    """Map a quaternion to the canonical hemisphere (w >= 0)

    q and -q describe the same rotation but are different network inputs. Training on only one
    hemisphere and evaluating on both makes an in-distribution attitude look out-of-distribution
    to the surrogate
    """
    if not args.canonicalize_quat:
        return q
    sign = torch.sign(q[..., 0:1])
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return q * sign


def quat_geodesic_deg(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Geodesic angle (deg) between two quaternions, sign-invariant"""
    d = torch.abs(torch.sum(q1 * q2, dim=-1)).clamp(0.0, 1.0)
    return torch.rad2deg(2.0 * torch.acos(d))


def random_quat_haar(generator=None) -> torch.Tensor:
    """Haar-uniform random rotation on SO(3) (normalized Gaussian 4-vector)"""
    v = torch.randn(4, device=device, generator=generator)
    while torch.norm(v) < 1e-6:
        v = torch.randn(4, device=device, generator=generator)
    return v / torch.norm(v)


def random_quat_euler_cube() -> torch.Tensor:
    """Uniform Euler angles in [-pi, pi]^3. Covers SO(3) but non-uniformly"""
    roll = random.uniform(-math.pi, math.pi)
    pitch = random.uniform(-math.pi, math.pi)
    yaw = random.uniform(-math.pi, math.pi)
    return math_utils.quat_from_euler_xyz(
        torch.tensor([roll], device=device),
        torch.tensor([pitch], device=device),
        torch.tensor([yaw], device=device)
    )[0]


def rotate_about_axis(target_quat: torch.Tensor, axis: int, angle_rad: float) -> torch.Tensor:
    """Apply a rotation about a single axis (0=x, 1=y, 2=z) to the target orientation"""
    delta_rot = torch.zeros(3, device=device)
    delta_rot[axis] = angle_rad
    delta_quat = math_utils.quat_from_euler_xyz(delta_rot[0:1], delta_rot[1:2], delta_rot[2:3])[0]
    return math_utils.quat_mul(delta_quat.unsqueeze(0), target_quat.unsqueeze(0))[0]


# -----------------------------------------------------------------------------
# Sampling plans
# -----------------------------------------------------------------------------
def generate_orientation_set(num_orientations, episode_idx):
    """Generate a set of orientations to cover different attitude configurations

    This creates a mix of:
    1. Target (LVLH) aligned orientation
    2. Origin (identity) orientation
    3. Small deviations around target orientation (+-10, +-25 deg per axis)
    4. Specific challenging orientations (90 deg rotations around main axes)
    5. Random orientations covering the full SO(3) rotation group

    Blocks 1-4 give 17 deterministic orientations concentrated near the operating point,
    everything above that is random. With --orientations_per_episode 60 the split is
    17 deterministic / 43 random, which keeps the near-nadir region densely sampled while
    covering the whole rotation group for tumbling states

    Args:
        num_orientations: How many orientations to generate
        episode_idx: Used to vary patterns across episodes

    Returns:
        List of quaternion tensors
    """
    orientations = []

    # 1. Target orientation (LVLH aligned)
    target_quat = orbit_manager.get_data("orientation", num_envs=1)[0]  # shape: [4]
    orientations.append(target_quat.clone())

    # 2. Origin (identity) orientation
    orientations.append(torch.tensor([1.0, 0.0, 0.0, 0.0], device=device))

    # 3. Small deviations from the target orientation (+-10 and +-25 deg around each axis)
    for axis in range(3):
        for angle_deg in [10, -10, -25, 25]:
            if len(orientations) < num_orientations:
                orientations.append(rotate_about_axis(target_quat, axis, math.radians(angle_deg)))

    # 4. 90 deg rotations from the target orientation
    for axis in range(3):
        if len(orientations) < num_orientations:
            orientations.append(rotate_about_axis(target_quat, axis, math.pi / 2))

    # 5. Fill the rest with random orientations covering the full SO(3)
    # Deterministic per-episode seeding for reproducibility
    random.seed(args.seed + episode_idx)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + episode_idx)

    while len(orientations) < num_orientations:
        if args.attitude_sampling == "haar":
            rand_quat = random_quat_haar(generator=gen)
        else:
            rand_quat = random_quat_euler_cube()

        if args.random_attitude_frame == "lvlh":
            rand_quat = math_utils.quat_mul(rand_quat.unsqueeze(0), target_quat.unsqueeze(0))[0]

        orientations.append(rand_quat)

    orientations = [q / torch.norm(q) for q in orientations]

    if len(orientations) < 17:
        print(f"  WARNING: orientations_per_episode={num_orientations} truncates the "
              f"deterministic blocks; the random SO(3) block is empty.")

    return orientations[:num_orientations]


def generate_boom_patterns(num_patterns, random_seed):
    """Generate boom extension patterns for testing: 19 fixed ones followed by 21 random ones

    Args:
        num_patterns: Number of patterns to generate
        random_seed: Random seed for reproducibility

    Returns:
        List of extension patterns (each is a tensor of 4 values)
    """
    random.seed(random_seed)

    patterns = [
        # All equal
        [0.0, 0.0, 0.0, 0.0],      # all retracted
        [3.7, 3.7, 3.7, 3.7],      # all fully extended
        # Partial extensions
        [1.85, 1.85, 1.85, 1.85],  # all half extended
        [2.77, 2.77, 2.77, 2.77],  # all 3/4 extended
        [0.93, 0.93, 0.93, 0.93],  # all 1/4 extended

        # Individual booms
        [3.7, 0.0, 0.0, 0.0],
        [0.0, 3.7, 0.0, 0.0],
        [0.0, 0.0, 3.7, 0.0],
        [0.0, 0.0, 0.0, 3.7],

        # Opposing pairs
        [3.7, 0.0, 3.7, 0.0],
        [0.0, 3.7, 0.0, 3.7],
        [1.85, 0.0, 1.85, 0.0],
        [0.0, 1.85, 0.0, 1.85],
        [3.7, 1.85, 3.7, 1.85],
        [1.85, 3.7, 1.85, 3.7],

        # Asymmetrical patterns
        [3.7, 2.77, 1.85, 0.93],   # gradient
        [3.7, 3.7, 0.0, 0.0],      # two extended, two retracted
        [3.7, 3.7, 1.85, 1.85],    # two full, two half
        [0.0, 0.0, 1.85, 1.85],    # two none, two half
    ]

    # Random configurations
    for _ in range(21):
        patterns.append([random.uniform(0, 3.7) for _ in range(4)])

    return [torch.tensor(pattern, device=device) for pattern in patterns[:num_patterns]]


# -----------------------------------------------------------------------------
# Simulation state setters / readers
# -----------------------------------------------------------------------------
def set_spacecraft_orientation(env_ids, orientation):
    """Set the orientation of the spacecraft"""
    root_states = robot.data.root_state_w[env_ids].clone()
    root_states[:, 3:7] = orientation
    robot.write_root_state_to_sim(root_states, env_ids=env_ids)


def set_boom_extensions(env_ids, extensions):
    """Set the extension of each boom using the master joints

    Args:
        env_ids (Tensor): Tensor of environment indices
        extensions (Tensor): Either shape [4] (shared for all envs) or [N, 4] (per env)
    """
    master_joint_names = [f"d3_boom_{i+1}_joint" for i in range(NUM_BOOMS)]
    joint_indices = []
    for name in master_joint_names:
        if name not in robot.joint_names:
            raise KeyError(f"Joint {name} not found in robot.joint_names")
        joint_indices.append(robot.joint_names.index(name))

    if extensions.dim() == 1:
        boom_extensions = extensions.unsqueeze(0).repeat(len(env_ids), 1)
    else:
        boom_extensions = extensions

    boom_extensions = torch.clamp(boom_extensions, 0.0, MAX_BOOM_LENGTH)

    joint_pos = robot.data.joint_pos[env_ids].clone()
    joint_vel = robot.data.joint_vel[env_ids].clone()

    for boom_id, joint_idx in enumerate(joint_indices):
        joint_pos[:, joint_idx] = boom_extensions[:, boom_id]
        joint_vel[:, joint_idx] = 0.0

    robot.write_joint_state_to_sim(
        position=joint_pos,
        velocity=joint_vel,
        joint_ids=None,
        env_ids=env_ids
    )

    if args.debug:
        print(f"[Boom Extensions] Set to: {boom_extensions}")


def get_boom_extensions(env_ids):
    """Normalized total extension of each boom, from the sum of its segment joints"""
    boom_extensions = torch.zeros((len(env_ids), NUM_BOOMS), device=device)

    for boom_id in range(NUM_BOOMS):
        boom_joints = []
        base_name = f"d3_boom_{boom_id+1}_p"

        for link_id in range(1, NUM_LINKS_PER_BOOM + 1):
            joint_name = f"{base_name}{link_id}_joint"
            if joint_name in robot.joint_names:
                boom_joints.append(robot.joint_names.index(joint_name))

        if boom_joints:
            joint_positions = robot.data.joint_pos[env_ids][:, boom_joints]
            boom_extensions[:, boom_id] = torch.sum(joint_positions, dim=1)

    return torch.clamp(boom_extensions / MAX_BOOM_LENGTH, 0.0, 1.0)


def get_master_joint_positions(env_ids):
    """Read the 4 master joints directly

    The RL inference path may read the master joints while this collector sums the 34 segment
    joints. In that case the surrogate would be trained and queried on different features, so
    both are recorded in the diagnostics to verify they agree
    """
    out = torch.zeros((len(env_ids), NUM_BOOMS), device=device)
    for i in range(NUM_BOOMS):
        name = f"d3_boom_{i+1}_joint"
        if name in robot.joint_names:
            out[:, i] = robot.data.joint_pos[env_ids][:, robot.joint_names.index(name)]
    return out


def get_num_ray_hits(env_idx):
    """Number of rays that hit the spacecraft, -1 if the sensor data is not available

    The projected area is A_proj = num_hits * d_h * d_v, which gives A_max for the analytic drag
    envelope 0 <= ||F|| <= 0.5 * rho * v_rel^2 * C_D * A_max used in the boundedness analysis
    """
    try:
        hits = raycaster.data.ray_hits_w[env_idx]
        finite = torch.isfinite(hits).all(dim=-1)
        return float(finite.sum().item())
    except Exception:
        return -1.0


def get_relevant_features(env_ids):
    """Extract only the features relevant for drag force prediction

    Feature layout (22D), identical to the RL inference path:
        [0:4]   spacecraft quaternion (w,x,y,z), canonicalized w>=0
        [4:8]   normalized boom extensions in [0,1]
        [8:12]  LVLH quaternion, canonicalized w>=0
        [12:15] LVLH position
        [15:18] inertial velocity
        [18:21] angular velocity of the co-rotating atmosphere
        [21]    air density
    """
    spacecraft_quat = canon_quat(robot.data.root_quat_w[env_ids].to(dtype=torch.float32))

    try:
        boom_extensions = get_boom_extensions(env_ids).to(dtype=torch.float32)
    except Exception as e:
        print(f"Error getting boom states: {e}")
        boom_extensions = torch.zeros((len(env_ids), 4), device=device, dtype=torch.float32)

    orbit_data = env.command_manager.get_command("orbit_data")
    orbit_pos = orbit_data["position"][env_ids].to(dtype=torch.float32)
    orbit_vel = orbit_data["velocity"][env_ids].to(dtype=torch.float32)
    atm_ang_vel = orbit_data["atm_ang_velocity"][env_ids].to(dtype=torch.float32)
    orbit_quat = canon_quat(orbit_data["orientation"][env_ids].to(dtype=torch.float32))

    air_density = orbit_data["air_density"][env_ids].to(dtype=torch.float32)
    if air_density.dim() == 1:
        air_density = air_density.unsqueeze(0)
    if air_density.dim() == 2 and air_density.shape[1] != 1:
        air_density = air_density[:, :1]

    try:
        return torch.cat([
            spacecraft_quat,      # 4D
            boom_extensions,      # 4D
            orbit_quat,           # 4D
            orbit_pos,            # 3D
            orbit_vel,            # 3D
            atm_ang_vel,          # 3D
            air_density           # 1D
        ], dim=1)
    except Exception as e:
        print(f"Error concatenating features: {e}")
        return torch.zeros((len(env_ids), 22), device=device, dtype=torch.float32)


def get_diagnostics(env_idx, commanded_quat, commanded_ext):
    """Per-sample diagnostics, see metadata["diagnostic_columns"]"""
    ids = torch.tensor([env_idx], device=device)

    measured_quat_raw = robot.data.root_quat_w[ids][0].to(dtype=torch.float32)
    drift = quat_geodesic_deg(commanded_quat.to(dtype=torch.float32), measured_quat_raw)

    measured_seg = get_boom_extensions(ids)[0] * MAX_BOOM_LENGTH  # back to meters
    measured_master = get_master_joint_positions(ids)[0]

    try:
        ang_vel = robot.data.root_ang_vel_w[ids][0].to(dtype=torch.float32)
    except Exception:
        ang_vel = torch.zeros(3, device=device)

    try:
        orbit_time = torch.tensor([float(orbit_manager.current_time)], device=device)
    except Exception:
        orbit_time = torch.tensor([-1.0], device=device)

    return torch.cat([
        commanded_quat.to(dtype=torch.float32).reshape(4),
        measured_quat_raw.reshape(4),
        drift.reshape(1),
        commanded_ext.to(dtype=torch.float32).reshape(4),
        measured_seg.reshape(4),
        measured_master.reshape(4),
        ang_vel.reshape(3),
        torch.tensor([get_num_ray_hits(env_idx)], device=device),
        orbit_time.reshape(1),
    ]).cpu()


def collect_data_for_configuration(env_idx, stabilization_steps, commanded_quat, commanded_ext):
    """Collect data for the current spacecraft and boom configuration

    Args:
        env_idx: Environment index to collect from
        stabilization_steps: Number of steps to allow stabilization
        commanded_quat: Attitude that was written, for the drift diagnostic
        commanded_ext: Boom extensions that were written

    Returns:
        bool: True if the data collection was successful, False otherwise
    """
    global total_samples
    try:
        # Allow the physics to stabilize. Zero action = zero boom rate = hold
        for _ in range(stabilization_steps):
            env.step(torch.zeros_like(env.action_manager.action))
            orbit_manager.step()

        features = get_relevant_features(torch.tensor([env_idx], device=device))

        # Net atmospheric drag force and torque at the root (inertial frame)
        drag_force, drag_torque, valid_ids = calculate_atmospheric_drag_forces(
            env=env,
            env_ids=torch.tensor([env_idx], device=device),
            drag_coefficient=2.2,
            asset_cfg=SceneEntityCfg("robot"),
            drag_sensor_name="drag_sensor",
            orbit_data_term_name="orbit_data",
            return_net_at_root=True,
        )

        if drag_force is None or drag_torque is None or valid_ids is None or len(valid_ids) == 0:
            if args.debug:
                print(f"  No valid drag forces for environment {env_idx}")
            return False

        # Reject non-finite samples explicitly instead of silently storing them
        if not torch.isfinite(features).all():
            print(f"  Discarding sample: non-finite feature vector (env {env_idx})")
            return False
        if not (torch.isfinite(drag_force[0]).all() and torch.isfinite(drag_torque[0]).all()):
            print(f"  Discarding sample: non-finite drag wrench (env {env_idx})")
            return False

        # Single output vector [Fx, Fy, Fz, Tx, Ty, Tz]
        forces_torques = torch.cat([drag_force[0], drag_torque[0]], dim=0)

        inputs.append(features[0].cpu())
        outputs.append(forces_torques.cpu())
        diagnostics.append(get_diagnostics(env_idx, commanded_quat, commanded_ext))

        total_samples += 1

        if args.debug:
            print(f"  Collected sample {total_samples}")
            print(f"  Force: {drag_force[0]}, Torque: {drag_torque[0]}")

        return True

    except Exception as e:
        print(f"  Error collecting data for environment {env_idx}: {e}")
        return False


# -----------------------------------------------------------------------------
# Main data collection loop
# -----------------------------------------------------------------------------
total_configs = args.num_episodes * args.orientations_per_episode * args.configs_per_orientation

print(f"Starting data collection with:")
print(f"  - {args.num_episodes} episodes (orbit time points)")
print(f"  - {args.orientations_per_episode} orientations per episode "
      f"({args.attitude_sampling} sampling, frame={args.random_attitude_frame})")
print(f"  - {args.configs_per_orientation} boom configurations per orientation")
print(f"  - {args.stabilization_steps} stabilization steps between changes")
print(f"  - quaternion canonicalization (w>=0): {bool(args.canonicalize_quat)}")
print(f"  - Expected total: {total_configs} samples")

progress = tqdm(total=total_configs, desc="Collecting samples")

total_orbit_time = orbit_manager.time_steps[-1].item()
episode_spacing = total_orbit_time / args.num_episodes if args.num_episodes > 1 else 0

output_dir = os.path.join(os.path.dirname(os.path.abspath(args.output_path)), "episodes")
os.makedirs(output_dir, exist_ok=True)

for episode in range(args.num_episodes):
    orbit_start_time = episode * episode_spacing
    print(f"\nEpisode {episode+1}/{args.num_episodes} - Orbit time: {orbit_start_time:.2f}s")

    env.reset()
    env.step(torch.zeros_like(env.action_manager.action))

    # Force reset of the orbit data to a specific time
    orbit_manager.reset(start_time=orbit_start_time)
    print(f"  Reset orbit to time {orbit_start_time:.2f}s")

    # Retract all booms (action -1 = maximum retraction rate), so that the first samples of
    # the episode do not inherit the boom command of the previous one
    retract = torch.full((env.num_envs, 4), -1.0, device=env.device)
    for _ in range(5):
        env.step(retract)

    orientations = generate_orientation_set(
        num_orientations=args.orientations_per_episode,
        episode_idx=episode
    )
    boom_patterns = generate_boom_patterns(
        num_patterns=args.configs_per_orientation,
        random_seed=args.seed + episode * 100
    )

    for orient_idx, orientation in enumerate(orientations):
        if args.debug:
            print(f"  Orientation {orient_idx+1}/{len(orientations)}")

        try:
            set_spacecraft_orientation(
                env_ids=torch.arange(env.num_envs, device=device),
                orientation=orientation
            )

            env.step(torch.zeros_like(env.action_manager.action))

            for boom_pattern in boom_patterns:
                set_boom_extensions(
                    env_ids=torch.arange(env.num_envs, device=device),
                    extensions=boom_pattern
                )

                env.step(torch.zeros_like(env.action_manager.action))
                orbit_manager.step()

                for env_idx in range(env.num_envs):
                    collect_data_for_configuration(
                        env_idx, args.stabilization_steps,
                        commanded_quat=orientation, commanded_ext=boom_pattern
                    )

                progress.update(1)

                env.step(torch.zeros_like(env.action_manager.action))
                orbit_manager.step()

        except Exception as e:
            print(f"  Error processing orientation {orient_idx}: {e}")
            continue

    # Save the episode
    if inputs:
        episode_inputs_tensor = torch.stack(inputs)
        episode_outputs_tensor = torch.stack(outputs)
        episode_diag_tensor = torch.stack(diagnostics)

        episode_metadata = dict(metadata)
        episode_metadata["episode_index"] = episode
        episode_metadata["orbit_start_time"] = orbit_start_time
        episode_metadata["num_samples"] = len(episode_inputs_tensor)
        episode_metadata["cumulative_samples"] = total_samples
        episode_metadata["collection_completed"] = time.strftime("%Y-%m-%d %H:%M:%S")

        episode_file = os.path.join(output_dir, f"episode_{episode+1:03d}.pt")
        torch.save({
            'inputs': episode_inputs_tensor,
            'outputs': episode_outputs_tensor,
            'diagnostics': episode_diag_tensor,
            'metadata': episode_metadata
        }, episode_file)

        drift = episode_diag_tensor[:, 8]
        print(f"\n Saved episode {episode+1} with {len(episode_inputs_tensor)} samples to {episode_file}")
        print(f"   attitude drift during stabilization: median {drift.median():.2f} deg, "
              f"p95 {torch.quantile(drift, 0.95):.2f} deg, max {drift.max():.2f} deg")

        inputs.clear()
        outputs.clear()
        diagnostics.clear()

progress.close()


# -----------------------------------------------------------------------------
# Merge episodes and report coverage
# -----------------------------------------------------------------------------
if args.merge_episodes:
    print(f"\nMerging per-episode files into {args.output_path} ...")
    files = sorted(glob.glob(os.path.join(output_dir, "episode_*.pt")))
    all_in, all_out, all_diag, all_ep = [], [], [], []
    for f in files:
        d = torch.load(f, map_location="cpu")
        all_in.append(d['inputs'])
        all_out.append(d['outputs'])
        all_diag.append(d.get('diagnostics', torch.zeros((len(d['inputs']), 26))))
        all_ep.append(torch.full((len(d['inputs']),), d['metadata'].get('episode_index', -1)))

    X = torch.cat(all_in)
    Y = torch.cat(all_out)
    D = torch.cat(all_diag)
    E = torch.cat(all_ep)   # episode index, for stratified train/val/test splitting

    metadata["num_samples"] = int(X.shape[0])
    metadata["collection_completed"] = time.strftime("%Y-%m-%d %H:%M:%S")

    torch.save({'inputs': X, 'outputs': Y, 'diagnostics': D,
                'episode_index': E, 'metadata': metadata}, args.output_path)

    # Coverage report
    qB, qL = X[:, 0:4], X[:, 8:12]
    fmag = torch.norm(Y[:, :3], dim=1)
    tmag = torch.norm(Y[:, 3:], dim=1)

    print("\n================ DATASET SUMMARY ================")
    print(f"  samples                 : {X.shape[0]}  (inputs {tuple(X.shape)}, outputs {tuple(Y.shape)})")
    print(f"  spacecraft quat w<0     : {100.0*(qB[:,0]<0).float().mean():.2f} %   (0.00 % expected if canonicalized)")
    print(f"  LVLH quat w<0           : {100.0*(qL[:,0]<0).float().mean():.2f} %")
    print(f"  boom seg-vs-master max |diff| : {(D[:,13:17]-D[:,17:21]).abs().max():.4f} m   "
          f"(must be ~0, otherwise the RL inference path reads different features)")
    print(f"  attitude drift          : median {D[:,8].median():.2f} deg, max {D[:,8].max():.2f} deg")
    print(f"  ray hits per sample     : min {D[:,24].min():.0f}, max {D[:,24].max():.0f}  (-1 = unavailable)")
    print(f"  |F| range               : {fmag.min():.3e} .. {fmag.max():.3e} N")
    print(f"  |tau| range             : {tmag.min():.3e} .. {tmag.max():.3e} N.m")
    print(f"  air density range       : {X[:,21].min():.3e} .. {X[:,21].max():.3e} kg/m^3")
    print(f"  atm ang. vel. std       : {X[:,18:21].std(dim=0).tolist()}  "
          f"(near zero => these 3 inputs carry no information)")
    print("================================================\n")

    print("Next: verify attitude coverage before retraining, e.g.")
    print("  q_rel = quat_mul(quat_conjugate(qL), qB)  ->  nearest-neighbour geodesic")
    print("  distance from a Haar-uniform test set to q_rel; report the p95.")

print("Data collection complete")
if env is not None:
    env.close()
simulation_app.close()