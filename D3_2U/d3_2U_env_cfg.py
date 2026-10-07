# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Author: Celia Redondo Verdú

"""

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import isaaclab_tasks.manager_based.my.D3_2U.mdp as mdp
from isaaclab_tasks.manager_based.my.D3_2U.config.d3_2U_cfg import CUBESAT_CFG  # isort:skip

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))


##
# Scene definition
##
def generate_boom_mesh_paths(num_booms=4, parts_per_boom=34):   #4, 34
    mesh_paths = []
    
    for boom_id in range(1, num_booms + 1):
        for part_id in range(1, parts_per_boom + 1):
            path = f"/World/envs/env_.*/d32U/d3_boom_{boom_id}_p{part_id}_link/visuals/D3_f_thick_subpart/mesh"
            mesh_paths.append(path)            
    mesh_paths.append("/World/envs/env_.*/d32U/cubesat2U_link/visuals/Cubesat_2U/mesh")
    return mesh_paths


@configclass
class CubeSatSceneCfg(InteractiveSceneCfg):
    """Configuration for the CubeSat simulation scene in zero gravity"""

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # spacecraft
    robot: ArticulationCfg = CUBESAT_CFG.replace(prim_path="{ENV_REGEX_NS}/d32U")

    # raycaster simulating drag
    '''uncomment when using the raycaster'''
    # drag_sensor = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/d32U/cubesat2U_link",
    #     update_period=0.0,
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
    #     attach_yaw_only=False,
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.012, size=[7.0, 7.0]),  # update to modify the raycaster resolution and size
    #     debug_vis=True,  # MUST BE TRUE
    #     mesh_prim_paths=generate_boom_mesh_paths(),
    # )




##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for controlling the CubeSat booms"""

    boom_extension = mdp.BoomVelocityActionCfg(
        asset_name="robot",
        max_len=3.7,
        min_len=0.0,
        max_vel=0.05,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group"""

        # Position error (body frame)
        pos_error = ObsTerm(func=mdp.get_position_error_body, params={"asset_cfg": SceneEntityCfg("robot")})

        # Attitude error (body frame)
        attitude_error = ObsTerm(func=mdp.get_attitude_error_body, params={"asset_cfg": SceneEntityCfg("robot")})

        # Relative angular velocity (body frame)
        rate_error = ObsTerm(
            func=mdp.get_lvlh_rate_error_body,
            params={"asset_cfg": SceneEntityCfg("robot"), "omega_scale": 0.01, "clip": 1.0},
        )

        body_lin_vel = ObsTerm(func=mdp.get_body_linear_vel, params={"asset_cfg": SceneEntityCfg("robot")})

        # Boom states (position and velocity)
        boom_states = ObsTerm(func=mdp.get_boom_states, params={"asset_cfg": SceneEntityCfg("robot")})

        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            """Post initialization for policy configuration"""
            # Whether to enable observation corruption
            self.enable_corruption = False
            # Whether to concatenate all terms into a single tensor
            self.concatenate_terms = True

    # Define observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for aligning with the reference trajectory"""

    pose_tracking_global = RewTerm(
        func=mdp.pose_tracking_reward_l2,
        weight=5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std_pos": 20.0,
            "std_orient": math.pi / 2.0,
            "pos_weight": 2.0,
            "orient_weight": 1.0,
        },
    )

    pose_tracking_wide = RewTerm(
        func=mdp.pose_tracking_reward_l2,
        weight=15.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std_pos": 5.0,
            "std_orient": 0.523599, 
            "pos_weight": 1.0,
            "orient_weight": 1.0,
        },
    )

    pose_tracking_precision = RewTerm(
        func=mdp.pose_tracking_reward_l2,
        weight=50.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std_pos": 0.5,
            "std_orient": 0.08,
            "pos_weight": 1.0,
            "orient_weight": 1.0,
        },
    )

    ang_vel_damping = RewTerm(
        func=mdp.angular_velocity_penalty,
        weight=10.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    lin_vel_penalty = RewTerm(
        func=mdp.linear_velocity_penalty,
        weight=15.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    action_rate = RewTerm(
            func=mdp.action_rate_l2_penalty,
            weight=-0.2, 
        )

    boom_velocity = RewTerm(
        func=mdp.boom_velocity_physical_penalty,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class EventCfg:
    """Configuration for events"""

    # Reset CubeSat body
    reset_cubesat_body = EventTerm(
        func=mdp.reset_cubesat_body,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (-math.pi/2.0, math.pi/2.0),
                "pitch": (-math.pi/2.0, math.pi/2.0),
                "yaw": (-math.pi/2.0, math.pi/2.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (-0.000349, 0.000349),
                "pitch": (-0.000349, 0.000349),
                "yaw": (-0.000349, 0.000349),
            },
        },
    )

    '''uncomment when using the raycaster'''
    # reset_d3_booms = EventTerm(
    #     func=mdp.reset_booms_visual_noPM,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "extension_range": (0.0, 1.7),
    #     },
    # )
    
    # control_boom_segments = EventTerm(
    #     func=mdp.follow_boom_segments,
    #     mode="interval",
    #     interval_range_s=(0.0, 0.0),
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )

    # # Update the pose of the raycaster
    # update_raycaster = EventTerm(
    #     func=mdp.update_raycaster_orientation,
    #     mode="interval",
    #     interval_range_s=(0.0, 0.0),
    #     params={
    #         "orbit_data_term_name": "orbit_data",
    #         "offset_raycaster": 5.0,  # Offset the raycaster to the opposite direction of LVLH Y-axis
    #                                   # Update for the specific application                  
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "raycaster_name": "drag_sensor",
    #     },
    # )

    # apply_perturbations = EventTerm(
    #     func=mdp.apply_perturbations_raycaster,
    #     mode="interval",
    #     interval_range_s=(0.0, 0.0),
    #     params={
    #         "apply_drag": True,  # Enable/disable atmospheric drag
    #         "apply_gg": True,  # Enable/disable gravity gradient
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "drag_coefficient": 2.2,
    #         "drag_sensor_name": "drag_sensor",
    #         "orbit_data_term_name": "orbit_data",
    #         "target_bodies": ["cubesat2U_link"],
    #         "min_mass_threshold": 0.0,
    #     },
    # )
    
    reset_d3_booms = EventTerm(
        func=mdp.reset_booms,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "extension_range": (0.0, 3.7),  
            # "velocity_range": (-0.0, 0.0),  
        },
    )
    
    apply_perturbations_model = EventTerm(
        func=mdp.apply_perturbations_model,
        mode="interval",
        interval_range_s=(0.0, 0.0), 
        params={
            "apply_drag": True,  # Enable/disable atmospheric drag
            "apply_gg": True,    # Enable/disable gravity gradient
            "asset_cfg": SceneEntityCfg("robot"),
            "drag_coefficient": 2.2,
            "orbit_data_term_name": "orbit_data",
            "target_bodies": ["cubesat2U_link", "d3_boom_1_pointmass_link", "d3_boom_2_pointmass_link", "d3_boom_3_pointmass_link", "d3_boom_4_pointmass_link",],
            "min_mass_threshold": 0.0,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for when the spacecraft drifts too far off course"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CommandsCfg:
    """Command specifications for the CubeSat environment"""

    # The orbit reference command must be first to update other commands
    orbit_data = mdp.OrbitDataCommandCfg()
    target_pose = mdp.TargetPoseCommandCfg()


##
# Environment configuration
##


@configclass
class CubeSatEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the CubeSat environment"""

    # Scene settings
    scene: CubeSatSceneCfg = CubeSatSceneCfg(num_envs=100, env_spacing=5.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Orbit data path 
    orbit_data_path: str = "/home/huro/Documentos/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/my/D3_2U/orbit_data/orbit_data.h5"

    # Surrogate model for the atmospheric drag
    use_surrogate_model: bool = True
    surrogate_model_path = "/home/huro/Documentos/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/my/D3_2U/train_model/drag_surrogate_layers3_hidden128_batch256_normTrue.pt"
    surrogate_scalers_path: str | None = None
    surrogate_cuda_graph: bool = False
    surrogate_fingerprint: str | None = None

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization"""
        # General settings
        self.decimation = 30
        self.episode_length_s = 5460
        # Simulation settings
        self.sim.dt = 0.5
        self.sim.render_interval = self.decimation  
        self.sim.disable_contact_processing = True

        self.time_scale = 1.0  # Real-time simulation

        self.sim.use_fabric = True  # recommendation from https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.html

        # Update orbit data path in commands config
        self.commands.orbit_data.orbit_data_path = self.orbit_data_path

        self.sim.device = "cuda"

        # Just set a flag that will be checked later
        self._surrogate_initialized = False

    def get_drag_surrogate(self):
        """Lazily initialise the surrogate model when first needed"""
        if getattr(self, "_drag_surrogate", None) is not None:
            return self._drag_surrogate
        if not self.use_surrogate_model or self._surrogate_initialized:
            self._drag_surrogate = None
            return None

        try:
            self._drag_surrogate = mdp.events.DragForceSurrogate(
                model_path=self.surrogate_model_path,
                device=self.sim.device,
                scalers_path=self.surrogate_scalers_path,
                fold_scalers=True,
                use_cuda_graph=self.surrogate_cuda_graph,
                validate_steps=8,
                clamp_outputs=True,
                expected_fingerprint=self.surrogate_fingerprint,
            )
            # The env count is only known once the cfg is overwritten from --num_envs,
            self._drag_surrogate.warmup(self.scene.num_envs)
            self._surrogate_initialized = True
        except Exception as exc:
            self._drag_surrogate = None
            self._surrogate_initialized = False
            raise RuntimeError(f"could not initialise the drag surrogate from {self.surrogate_model_path}: {exc}") from exc

        print(f"Initialized drag force surrogate from {self.surrogate_model_path}")
        return self._drag_surrogate