"""
Cubesat 2U with Drag Deorbit Device environment
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Cubesat-D3-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.d3_2U_env_cfg:CubeSatEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)