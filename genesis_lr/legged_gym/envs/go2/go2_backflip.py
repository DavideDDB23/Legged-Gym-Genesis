from legged_gym.envs.go2.go2 import GO2
from legged_gym.utils.gs_utils import *
import torch
import genesis as gs

class GO2Backflip(GO2):
    """Backflip task for GO2 robot.
    Inherits the generic GO2 walking environment and overrides
    observation construction, termination condition and reward function
    set to reproduce the behaviour defined in reward_wrapper.Backflip
    while leveraging the modular Legged-Gym infrastructure."""

    # ---------------------------------------------------------------------
    # Buffers & bookkeeping
    # ---------------------------------------------------------------------
    def _init_buffers(self):
        super()._init_buffers()
        self.foot_positions = torch.zeros(
            (self.num_envs, len(self.feet_indices), 3), device=self.device, dtype=gs.tc_float
        )
        # Not strictly required but useful for potential future shaping
        self.feet_max_height = torch.zeros(
            (self.num_envs, len(self.feet_indices)), device=self.device, dtype=gs.tc_float
        )

    # ---------------------------------------------------------------------
    # Physics-step callback – capture feet positions for reward terms
    # ---------------------------------------------------------------------
    def _post_physics_step_callback(self):
        # retain generic command logic, pushes, etc.
        super()._post_physics_step_callback()
        # Cache foot world positions for reward computation
        self.foot_positions[:] = self.robot.get_links_pos(self.feet_indices)

    # ---------------------------------------------------------------------
    # Observations
    # ---------------------------------------------------------------------
    def compute_observations(self):
        phase = torch.pi * self.episode_length_buf[:, None] * self.dt / 2
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales.ang_vel,               # 3
                self.projected_gravity,                                    # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 12
                self.dof_vel * self.obs_scales.dof_vel,                    # 12
                self.actions,                                              # 12
                self.last_actions,                                         # 12
                torch.sin(phase),
                torch.cos(phase),
                torch.sin(phase / 2),
                torch.cos(phase / 2),
                torch.sin(phase / 4),
                torch.cos(phase / 4),
            ],
            dim=-1,
        )

        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.cat(
                [
                    self.base_pos[:, 2:3],                                  # 1
                    self.base_lin_vel * self.obs_scales.lin_vel,            # 3
                    self.base_ang_vel * self.obs_scales.ang_vel,            # 3
                    self.projected_gravity,                                 # 3
                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, # 12
                    self.dof_vel * self.obs_scales.dof_vel,                 # 12
                    self.actions,                                           # 12
                    self.last_actions,                                      # 12
                    torch.sin(phase),
                    torch.cos(phase),
                    torch.sin(phase / 2),
                    torch.cos(phase / 2),
                    torch.sin(phase / 4),
                    torch.cos(phase / 4),
                ],
                dim=-1,
            )

    # ---------------------------------------------------------------------
    # Termination – no early termination other than timeout
    # ---------------------------------------------------------------------
    def check_termination(self):
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        # No time-out flag required for reward shaping here
        self.time_out_buf = self.reset_buf.clone()

    # ---------------------------------------------------------------------
    # Reward terms – mirror reward_wrapper.Backflip
    # ---------------------------------------------------------------------
    def _reward_orientation_control(self):
        current_time = self.episode_length_buf * self.dt
        phase = (current_time - 0.5).clamp(min=0, max=0.5)
        quat_pitch = gs_quat_from_angle_axis(
            4 * phase * torch.pi,
            torch.tensor([0, 1, 0], device=self.device, dtype=torch.float),
        )

        desired_base_quat = gs_quat_mul(
            quat_pitch, self.base_init_quat.reshape(1, -1).repeat(self.num_envs, 1)
        )
        inv_desired_base_quat = gs_inv_quat(desired_base_quat)
        desired_projected_gravity = gs_transform_by_quat(
            self.global_gravity, inv_desired_base_quat
        )

        return torch.sum(
            torch.square(self.projected_gravity - desired_projected_gravity), dim=1
        )

    def _reward_ang_vel_y(self):
        current_time = self.episode_length_buf * self.dt
        ang_vel = -self.base_ang_vel[:, 1].clamp(max=7.2, min=-7.2)
        return ang_vel * torch.logical_and(current_time > 0.5, current_time < 1.0)

    def _reward_ang_vel_z(self):
        return torch.abs(self.base_ang_vel[:, 2])

    def _reward_lin_vel_z(self):
        current_time = self.episode_length_buf * self.dt
        lin_vel = self.robot.get_vel()[:, 2].clamp(max=3)
        return lin_vel * torch.logical_and(current_time > 0.5, current_time < 0.75)

    def _reward_height_control(self):
        current_time = self.episode_length_buf * self.dt
        target_height = 0.3
        height_diff = torch.square(target_height - self.base_pos[:, 2]) * torch.logical_or(
            current_time < 0.4, current_time > 1.4
        )
        return height_diff

    def _reward_actions_symmetry(self):
        actions_diff = torch.square(self.actions[:, 0] + self.actions[:, 3])
        actions_diff += torch.square(self.actions[:, 1:3] - self.actions[:, 4:6]).sum(dim=-1)
        actions_diff += torch.square(self.actions[:, 6] + self.actions[:, 9])
        actions_diff += torch.square(self.actions[:, 7:9] - self.actions[:, 10:12]).sum(dim=-1)
        return actions_diff

    def _reward_gravity_y(self):
        return torch.square(self.projected_gravity[:, 1])

    def _reward_feet_distance(self):
        cur_footsteps_translated = self.foot_positions - self.base_pos.unsqueeze(1)
        footsteps_body = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_body[:, i, :] = gs_quat_apply(
                gs_quat_conjugate(self.base_quat), cur_footsteps_translated[:, i, :]
            )

        stance_width = 0.3 * torch.zeros([self.num_envs, 1], device=self.device)
        desired_ys = torch.cat(
            [
                stance_width / 2,
                -stance_width / 2,
                stance_width / 2,
                -stance_width / 2,
            ],
            dim=1,
        )
        return torch.square(desired_ys - footsteps_body[:, :, 1]).sum(dim=1)

    def _reward_feet_height_before_backflip(self):
        current_time = self.episode_length_buf * self.dt
        foot_height = self.foot_positions[:, :, 2].view(self.num_envs, -1) - 0.02
        return foot_height.clamp(min=0).sum(dim=1) * (current_time < 0.5)

    def _reward_collision(self):
        return (
            1.0
            * (
                torch.norm(
                    self.link_contact_forces[:, self.penalized_indices, :], dim=-1
                )
                > 0.1
            )
        ).sum(dim=1) 