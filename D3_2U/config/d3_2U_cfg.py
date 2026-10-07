
""" Configuration for the 2U 3D spacecraft """

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

# Define the configuration for the CubeSat with deployable booms
CUBESAT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/d3_2U",
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"usd_path=f"/home/huro/Documentos/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/my/D3_2U/usd/d3_2U_IS_visual/d3_2U_IS_visual.usd",  # visual model to use with the raycaster
        usd_path=f"/home/huro/Documentos/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/my/D3_2U/usd/d3_2U_IS_simplified/d3_2U_IS_simplified.usd", # model to use with the FFNN

        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
            
            linear_damping=0.0,
            angular_damping=0.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.000,  
            stabilization_threshold=0.000, 
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),  # Initial position in world space
        joint_pos={f"d3_boom_{i}_joint": 0.0 for i in range(1, 5)},  # Booms start retracted
    ),
    actuators={
        f"boom_{i}_actuator": ImplicitActuatorCfg(
            joint_names_expr=[f"d3_boom_{i}_joint"],  # Apply to all boom segments
            effort_limit=1.0, 
            velocity_limit=0.05, 
            stiffness=2.0, 
            damping=0.5, 
        ) for i in range(1, 5)  # Configure all four booms
    },
)

"""Configuration for the CubeSat 2U with Drag Deorbit Device (DDD)."""