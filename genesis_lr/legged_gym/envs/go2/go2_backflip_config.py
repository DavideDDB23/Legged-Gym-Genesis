from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2BackflipCfg(LeggedRobotCfg):
    """Configuration for the GO2 Backflip task."""

    class env(LeggedRobotCfg.env):
        num_observations = 60  # customised observation size
        num_privileged_obs = 64
        num_history = 1
        episode_length_s = 2.0
        num_envs = 2048
        num_actions = 12

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = "plane" # none, plane, heightfield
        friction = 1.0
        restitution = 0.
        
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.32]
        rot = [1.0, 0.0, 0.0, 0.0]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,   # [rad]
            'RL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': 0.0,  # [rad]
            'RR_hip_joint': 0.0,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.0,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.0,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        stiffness = {'joint': 70.0}   # [N*m/rad]
        damping = {'joint': 3.0}     # [N*m*s/rad]
        action_scale = 0.5
        dt =  0.02
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        dof_names = [
            'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
            'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
            'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
            'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
        ]
        foot_name = "foot"
        penalize_contacts_on = ["base", "thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = True

    class termination:
        # termination condition from original script
        roll_threshold = 0.4
        pitch_threshold = 0.4
        height_threshold = 0.2

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        heading_command = False
        resampling_time = 4.0
        num_commands = 4

        class ranges:
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]

    class rewards(LeggedRobotCfg.rewards):
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
            dof_pos_limits = 0
            collision = 0.0
            lin_vel_z_penalty = 0.0 
            ang_vel_xy = 0.0
            orientation = 0.0
            dof_vel = 0.0
            dof_acc = 0.0
            base_height = 0.0
            torques = 0.0

    class domain_rand:
        randomize_friction = True
        friction_range = [0.2, 1.5]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        randomize_com_displacement = True
        com_displacement_range = [-0.01, 0.01]
        randomize_motor_strength = False
        motor_strength_range = [0.9, 1.1]
        randomize_motor_offset = True
        motor_offset_range = [-0.02, 0.02]
        randomize_kp_scale = True
        kp_scale_range = [0.8, 1.2]
        randomize_kd_scale = True
        kd_scale_range = [0.8, 1.2]
        push_robots = False
        push_interval_s = -1
        max_push_vel_xy = 1.0
        simulate_action_latency = False # 1 step delay

    class observation_scales:
        lin_vel = 2.0
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05
    
    class noise:
        class noise_scales:
            lin_vel = 0.0
            ang_vel = 0.1
            gravity = 0.02
            dof_pos = 0.01
            dof_vel = 0.5
        
        noise_level = 1.0
        add_noise = True


class GO2BackflipCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        # algorithm training hyperparameters
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 0.001
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class policy:
        # network architectures
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'
        init_noise_std = 1.0

    class runner( LeggedRobotCfgPPO.runner ):
        # runner hyperparameters
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24
        max_iterations = 1000
        # logging
        run_name = ''
        experiment_name = 'go2_backflip'
        save_interval = 100
        load_run = "-1"
        # load and resume
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None