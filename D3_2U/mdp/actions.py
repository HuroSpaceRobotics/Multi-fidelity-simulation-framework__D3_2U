from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class BoomVelocityAction(ActionTerm):
    """Boom extension-rate control.

    The policy outputs a value in [-1, 1] per boom, mapped to an extension rate in
    [-max_vel, max_vel] m/s. The rate is integrated over the control step into an
    extension setpoint, clamped to [min_len, max_len] and tracked by the joint's PD drive.
    """

    cfg: BoomVelocityActionCfg

    def __init__(self, cfg: BoomVelocityActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.asset_name]

        # Master prismatic joints; the segment joints follow them (see events.follow_boom_segments)
        self.joint_indices = [self.asset.joint_names.index(f"d3_boom_{i + 1}_joint") for i in range(4)]

        self._raw_actions = torch.zeros(env.num_envs, 4, device=self.device)
        self._processed_actions = torch.zeros(env.num_envs, self.asset.num_joints, device=self.device)
        self._current_extension = torch.zeros(env.num_envs, 4, device=self.device)  # [m]

        self.v_max = cfg.max_vel
        self.dt = env.step_dt
        self.max_len = cfg.max_len
        self.min_len = cfg.min_len

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def target_boom_extensions(self) -> torch.Tensor:
        """Commanded extension of each boom [m], shape (num_envs, 4)."""
        return self._current_extension

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions.clamp(-1.0, 1.0)

        self._current_extension += self._raw_actions * self.v_max * self.dt
        self._current_extension.clamp_(self.min_len, self.max_len)

        self._processed_actions.zero_()
        self._processed_actions[:, self.joint_indices] = self._current_extension

    def apply_actions(self):
        self.asset.set_joint_position_target(self._processed_actions)

    def reset(self, env_ids: torch.Tensor | None = None):
        # Re-sync the integrator with the joint state written by the reset events,
        # otherwise the first step after a reset would command a jump.
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._current_extension[env_ids] = self.asset.data.joint_pos[env_ids][:, self.joint_indices]
        self._raw_actions[env_ids] = 0.0


@configclass
class BoomVelocityActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = BoomVelocityAction
    asset_name: str = "robot"
    max_len: float = 3.7
    min_len: float = 0.0
    max_vel: float = 0.01  # [m/s]