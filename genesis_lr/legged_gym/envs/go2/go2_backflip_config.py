from legged_gym.envs.go2.go2_config import GO2Cfg, GO2CfgPPO

class GO2BackflipCfg(GO2Cfg):
    """Configuration for the GO2 Backflip task."""

    class env(GO2Cfg.env):
        num_observations = 60  # customised observation size
        episode_length_s = 2.0
        # keep the rest of default values (num_envs etc.)

    class init_state(GO2Cfg.init_state):
        pos = [0.0, 0.0, 0.36]
        # quaternion unchanged (identity) – same as base class

    class commands(GO2Cfg.commands):
        curriculum = False
        heading_command = False
        resampling_time = 4.0

        class ranges:
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class rewards(GO2Cfg.rewards):
        soft_dof_pos_limit = 0.9

        class scales:
            # Desired back-flip specific terms
            ang_vel_y = 5.0
            ang_vel_z = -1.0
            lin_vel_z = 20.0
            orientation_control = -1.0
            feet_height_before_backflip = -30.0
            height_control = -10.0
            actions_symmetry = -0.1
            gravity_y = -10.0
            feet_distance = -1.0
            action_rate = -0.001
            # Disable locomotion tracking terms inherited from walking
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            feet_air_time = 0.0
            # keep generic safety/regularisation terms from parent
            dof_pos_limits = -10.0
            collision = -1.0
            lin_vel_z_penalty = 0.0  # placeholder – not used
            ang_vel_xy = 0.0
            orientation = 0.0
            dof_vel = 0.0
            dof_acc = 0.0
            base_height = 0.0
            torques = 0.0

class GO2BackflipCfgPPO(GO2CfgPPO):
    """Training hyper-parameters for GO2 backflip."""

    class runner(GO2CfgPPO.runner):
        experiment_name = 'go2_backflip'
        save_interval = 100
        max_iterations = 1000 