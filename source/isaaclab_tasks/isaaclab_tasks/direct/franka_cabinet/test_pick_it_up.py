# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom, Usd

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, RigidObject, ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform, quat_inv, quat_mul, transform_points, quat_conjugate, quat_apply
from torch.utils.tensorboard import SummaryWriter
# from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors.contact_sensor.contact_sensor import ContactSensor
from isaaclab.sensors import ContactSensorCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

import pdb
from isaaclab_eureka.utils import eureka_root_dir, read_pkl

@configclass
class TestPickItUpCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.65  # 519 timesteps
    decimation = 2
    action_space = 8
    observation_space = 25 # to modify
    state_space = 0
    seed=42
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048, env_spacing=3.0, replicate_physics=True, clone_in_fabric=False, 
    )


    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 1.157,
                "panda_joint2": -1.066,
                "panda_joint3": -0.155,
                "panda_joint4": -2.239,
                "panda_joint5": -1.841,
                "panda_joint6": 1.003,
                "panda_joint7": 0.469,
                "panda_finger_joint.*": 0.035,
            },
            pos=(0.4, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit_sim=87.0,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit_sim=12.0,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit_sim=200.0,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )


    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # objects
    # rotate around x rot=[0.7071, 0.7071, 0, 0]
    # rotate around y rot=[0.7071, 0, 0.7071, 0]

    ketchup = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ketchup",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.3, 0, 0], rot=[0.7071, 0.7071, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/admin_01/workspace_eureka/IsaacLabEureka/libero/COMMON/stable_hope_objects/ketchup/usd/ketchup.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False
            )
        ),  
    )

    basket = RigidObjectCfg(
        prim_path="/World/envs/env_.*/basket",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0, 0, 0], rot=[0, 0, 0, 1]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/admin_01/workspace_eureka/IsaacLabEureka/libero/COMMON/stable_hope_objects/basket/usd/basket.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False
            )
        ),
    )

    cream_cheese = RigidObjectCfg(
        prim_path="/World/envs/env_.*/alphabet_soup",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.2, 0.3, 0], rot=[0.7071, 0.7071, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/admin_01/workspace_eureka/IsaacLabEureka/libero/COMMON/stable_hope_objects/alphabet_soup/usd/alphabet_soup.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False
            )
        ),
    )

    alphabet_soup = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cream_cheese",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0, 0.5, 0], rot=[0.7071, 0.7071, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/admin_01/workspace_eureka/IsaacLabEureka/libero/COMMON/stable_hope_objects/cream_cheese/usd/cream_cheese.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False
            )
        ),
    )

    tomato_sauce = RigidObjectCfg(
        prim_path="/World/envs/env_.*/tomato_sauce",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.2, 0.5, 0], rot=[0.7071, 0.7071, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/admin_01/workspace_eureka/IsaacLabEureka/libero/COMMON/stable_hope_objects/tomato_sauce/usd/tomato_sauce.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False
            )
        ),
    )

    action_scale = 7.5 # 7.5 originally
    dof_velocity_scale = 0.1

    # reward scales
    dist_reward_scale = 1.5
    rot_reward_scale = 1.5
    open_reward_scale = 10.0
    action_penalty_scale = 0.05
    finger_reward_scale = 2.0


class TestPickItUp(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: TestPickItUpCfg

    def __init__(self, cfg: TestPickItUpCfg, render_mode: str | None = None, **kwargs):
        seed = cfg.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

        self.debug_vis = True
        # Only when debug_vis is true:
        self.show_robot_grasp=True
        self.show_target_object=False
        self.show_target_grasp_pose=True

        self.log_mine = False
        self.start_in_air = False
        self.root = eureka_root_dir()
        self.target_object_name = "alphabet_soup"
        self.target_site_name = "basket"
        self.input_direction = 2 # z axis
        if self.input_direction != 2:
            raise NotImplementedError("Change the _get_dones method and other calculations for deciding the entry size")
        self.path = f"{self.root}/libero/trajs/libero90/libero_90_living_room_scene1_pick_up_the_{self.target_object_name}_and_put_it_in_the_basket_traj_v2.pkl"
        if self.log_mine:
            import os
            log_dir = os.path.join(self.root, "logs", "replay_test")
            self.summary_writer_mine = SummaryWriter(log_dir)
        start_idx_in_episode_dict = {
            "alphabet_soup":80,
            "cream_cheese":80,
            "ketchup":80, # 80
            "tomato_sauce":90,
        }
        if not self.start_in_air:
            self.start_idx_in_episode = 0
        else:
            self.start_idx_in_episode = start_idx_in_episode_dict[self.target_object_name] # for cream_cheese
        # self.start_idx_in_episode = 80 # For living room scene 1 pick up the alphabet soup

        # some calculations to get max episode length from demonstrations
        self.data = read_pkl(self.path) 
        self.episodes = self.data["franka"]
        lengths = [len(ep["states"]) for ep in self.episodes]
        max_len = max(lengths) # not subtracting start idx from this to let RL explore

        frequency_ratio = 3
        episode_length = (max_len - 1) * frequency_ratio + 1
        cfg.episode_length_s = episode_length * cfg.sim.dt * cfg.decimation # handling this dynamically
        
        # pdb.set_trace()
        self.history_len = 2
        self.num_stages = 5
        self.debug = False  
        super().__init__(cfg, render_mode, **kwargs)

        # deal with the offset needed for each object
        offset_dict = {
            "alphabet_soup":torch.tensor([0, 0, 0.05], device=self.device),
            "tomato_sauce":torch.tensor([0, 0, 0.05], device=self.device),
            "cream_cheese":torch.tensor([0, 0, 0.04], device=self.device),
            "ketchup":torch.tensor([0, 0, 0.08], device=self.device),
        }
        offset = offset_dict[self.target_object_name]

        self.sample = self.episodes[0]["states"][0].copy()
        self.sample.pop("franka", None)
        
        # 1. Determine object names and ordering

        self.object_names = sorted(list(self.sample.keys()))
        self.object_indices = {name: i for i, name in enumerate(self.object_names)}

        # expose the following to llm
        # self.task_type = "placement"

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint1")[0]] = 0.1
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint2")[0]] = 0.1

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        stage = get_current_stage()
        # compute local corners once from env_0 then broadcast to all envs and transform per-env as needed
        prim = stage.GetPrimAtPath(f"/World/envs/env_0/{self.target_object_name}/object")
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(prim)
        min_wc = bbox.GetRange().GetMin()
        max_wc = bbox.GetRange().GetMax()

        # 8 corners in env_0 world coords, placed directly on device
        corners_world0 = torch.tensor(
            [
                [min_wc[0], min_wc[1], min_wc[2]],
                [max_wc[0], min_wc[1], min_wc[2]],
                [min_wc[0], max_wc[1], min_wc[2]],
                [min_wc[0], min_wc[1], max_wc[2]],
                [max_wc[0], max_wc[1], min_wc[2]],
                [max_wc[0], min_wc[1], max_wc[2]],
                [min_wc[0], max_wc[1], max_wc[2]],
                [max_wc[0], max_wc[1], max_wc[2]],
            ],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)  # [1, 8, 3]

        # env_0 object pose (to compute object-local coords once)
        pos0 = self.target_object.data.root_pos_w[0]      # [3]
        quat0 = self.target_object.data.root_quat_w[0]   # [4]
        quat_obj0 = quat_conjugate(quat0)
        inv_pos = -quat_apply(quat_obj0, pos0)

        # world -> object-root-local for env_0
        local_corners0 = transform_points(
            points=corners_world0,
            pos=inv_pos,
            quat=quat_obj0,
        ).squeeze(0)  # [8,3]

        # broadcast same local corners to all envs (object-local frame is identical by design)
        self.local_corners_init = local_corners0.unsqueeze(0).repeat(self.num_envs, 1, 1)  # [num_envs, 8, 3]

        # compute per-env size/centers in object-local frame (then shift to env-local if needed)
        local_min = self.local_corners_init.min(dim=1).values  # [num_envs, 3]
        local_max = self.local_corners_init.max(dim=1).values  # [num_envs, 3]
        self.target_object_size = (local_max - local_min)     # [num_envs, 3]

        # centers relative to scene.env_origins
        self.local_centers_init = (local_min + local_max).unsqueeze(1) / 2.0 # [num_envs, 1, 3]
        self.local_centers = torch.zeros_like(self.local_centers_init) # [num_envs, 1, 3]
        self.corners_target_obj = torch.zeros_like(self.local_corners_init)
        self.corners_target_obj_to_hand_pos = torch.zeros((self.num_envs, 24), device=self.device) # hard coded



        # copy over all envs, requiring that each object has the same init state
        self.quat = torch.zeros([self.num_envs, 4],device=self.device)
        self.target_to_hand_pos = torch.zeros((self.num_envs, 3),device=self.device) # relative position
        self.site_to_target_pos = torch.zeros((self.num_envs, 3),device=self.device) # relative position
        debug = False
        if debug:
            pos = self.target_object.data.root_pos_w - self.scene.env_origins
            quat_1=torch.tensor([0.7071, 0.7071, 0, 0],device = self.device) # x
            quat_2=torch.tensor([0.7071, 0, 0, 0.7071],device = self.device) # z
            quat_3=torch.tensor([0.7071, 0, 0.7071, 0],device = self.device) # y
            quat = self.target_object.data.root_quat_w
            quat[-1,:] = quat_mul(quat_2, self.target_object.data.root_quat_w[-1,:])
            corners_target_obj = transform_points(points=self.local_corners_init,
                                                pos=pos,
                                                quat=quat)
            quat_1_inv = quat_inv(quat_1)
            target_to_hand_rot = quat_mul(quat_1, quat_1_inv)





        # TARGET SITE

        prim = stage.GetPrimAtPath(f"/World/envs/env_0/{self.target_site_name}/object")
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(prim)
        min_wc = bbox.GetRange().GetMin()
        max_wc = bbox.GetRange().GetMax()
        self.target_site_corners_world = torch.tensor(
            [
                [min_wc[0], min_wc[1], min_wc[2]],
                [max_wc[0], max_wc[1], max_wc[2]],
            ],
            dtype=torch.float32,
            device=self.device,
        )  # [2, 3]
        self.target_site_corners_world -= self.scene.env_origins[0] # in env-local frame
        target_site_size = torch.tensor((max_wc - min_wc), device=self.device)
        # Since I never want the site to be moved, I don't need to calculate its local corners and later transform in each env.
        obj_xy_extent = self.target_object_size[:2].min()
        site_xy_extent = target_site_size[:2].max()
        # termination condition
        # half extents
        obj_half = 0.5 * obj_xy_extent
        site_half = 0.5 * site_xy_extent
        self.target_site_radius = site_half - obj_half
        if self.target_site_radius.item() < 0:
            self.target_site_radius = torch.tensor(0.02)

        hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_link7")),
            self.device,
        )
        lfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_leftfinger")),
            self.device,
        )
        rfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_rightfinger")),
            self.device,
        )

        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        robot_local_grasp_pose_rot, robot_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        # robot_local_pose_pos += torch.tensor([0, 0.0, 0.04], device=self.device) # already works well
        #TODO: check if this is object specific
        robot_local_pose_pos += offset # This one is tuned from libero demo

        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1)) # 3
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1)) # 4

        self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
        self.left_finger_body_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_body_idx = self._robot.find_bodies("panda_rightfinger")[0][0]
        self.left_finger_joint_idx = self._robot.find_joints("panda_finger_joint2")[0][0]
        self.right_finger_joint_idx = self._robot.find_joints("panda_finger_joint1")[0][0]



        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.hand_quat = torch.zeros((self.num_envs, 4), device=self.device) 
        self.helper_variable = torch.zeros((self.num_envs, 10), device=self.device)
        self.manipulability = torch.zeros((self.num_envs), device=self.device)
        self.to_desired_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.q_rel = torch.tensor([0.0014,  0.9270,  0.3749,  0.0036], device=self.device).repeat(self.num_envs, 1)
        self.past_relative_dist = torch.ones((self.num_envs, self.history_len), device=self.device)
        self.grasped = torch.zeros((self.num_envs), device=self.device, dtype=bool)
        self.stage = torch.zeros((self.num_envs,self.num_stages), device=self.device, dtype=bool)
        self.prev_actions = torch.zeros((self.num_envs, cfg.action_space), device=self.device)

        self.high_enough = torch.zeros((self.num_envs), device=self.device, dtype=bool)
        self.small_rotation = torch.zeros((self.num_envs), device=self.device, dtype=bool)
        self.low_enough = torch.zeros((self.num_envs), device=self.device, dtype=bool)
        self.inside_site = torch.zeros((self.num_envs), device=self.device, dtype=bool)

        # # Specific to in the air task, randomize the initialization
        self.update_rate = 0
        self.base = torch.tensor([ 91,  66,  67, 110,  66,  84, 106,  81,  54,  83,  72,  63,  68,  63,
                    57,  58,  72,  84,  76,  72,  62,  63,  85,  70,  72,  60,  68,  70,
                    64,  69,  85,  89,  77,  71,  74,  69,  64,  65,  70, 101,  72,  80,
                    61,  58,  74,  72,  61,  84,  60,  67], device=self.device)

        # # Repeat/tile until we have at least num_envs elements
        # repeat_times = (self.num_envs + base.numel() - 1) // base.numel()  # ceiling division
        # tiled = base.repeat(repeat_times)[:self.num_envs]  # crop to exact length

        # # Add random integers [0,20)
        # rand_add = torch.randint(low=0, high=21, size=(self.num_envs,), device=self.device)

        # # Final tensor
        # in_the_air_matrix = tiled + rand_add
        # self.object_default_state = torch.zeros((self.num_envs, 13), device=self.device)
        # num_episodes = len(self.episodes)
        # for i in range(num_episodes):
        #     # i-th episode gets a column in object_default_state
        #     # Use modulo if num_envs < num_episodes
        #     self.object_default_state[:, i % self.object_default_state.shape[1]] = in_the_air_matrix

        # Adding a visualizer
        if self.debug_vis:
            self.visualizer = self.define_markers()
            print("Debug visualizer initialized")

    def _setup_scene(self):
        # init_states = self.data['franka'][0]["init_state"] # this is from the first scene as set up. Init states of traj should be updated in reset_idx
        # init_states = self.data['franka'][0]["states"][100]

        init_states = self.data['franka'][0]["states"][self.start_idx_in_episode] # 92 in the air. Used 100 For two round training, now use 80, starting from ground
        robot_data = init_states['franka'] # seems that joint pos for isaaclab is always positive
        robot_joint_pos = robot_data["dof_pos"]
        # print(f"robot_joint_pos:{robot_joint_pos}")
        robot_joint_pos['panda_finger_joint2'] = robot_joint_pos['panda_finger_joint1']
        # print(f"robot_joint_pos:{robot_joint_pos}")
        robot_joint_pos = {k: v.item() for k, v in robot_joint_pos.items()}
        robot_cfg = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=5.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=4
                ),
            ),
            debug_vis=True,
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos=robot_joint_pos,
                pos=robot_data["pos"],
                rot=robot_data["rot"],
            ),
            actuators={
                "panda_shoulder": ImplicitActuatorCfg(
                    joint_names_expr=["panda_joint[1-4]"],
                    effort_limit_sim=200.0,
                    stiffness=600.0,
                    damping=80.0,
                ),
                "panda_forearm": ImplicitActuatorCfg(
                    joint_names_expr=["panda_joint[5-7]"],
                    effort_limit_sim=120.0,
                    stiffness=500.0,
                    damping=70.0,
                ),
                "panda_hand": ImplicitActuatorCfg(
                    joint_names_expr=["panda_finger_joint.*"],
                    effort_limit_sim=200.0,
                    stiffness=2e3,
                    damping=1e2,
                ),
                }
        )
        self._robot = Articulation(robot_cfg)
        self.scene.articulations["robot"] = self._robot

        #TODO: for the rest: consider if the task has articulation
        self.rigid_objects = {}
        keys = list(init_states.keys())
        keys.remove("franka")   # remove the one you don't want

        for k in keys:
            # if k in ["ketchup", "cream_cheese", "tomato_sauce"]:
            #     self.object_names.remove(k) # remove distractors for now, to be added in later ablations
            if k == self.target_object_name:
                print(f"target object {k} initial z height: {init_states[k]['pos'][2]}")
                activate_contact_sensors=True
                debug_vis=True
            else:
                activate_contact_sensors = False
                debug_vis=False
            # if k == self.target_object_name or k==self.target_site_name: # Disable other objects for now
            cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/{k}",
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=init_states[k]["pos"],
                    rot=init_states[k]["rot"],
                ),
                debug_vis=debug_vis,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=f"{self.root}/libero/COMMON/stable_hope_objects/{k}/usd/{k}.usd",
                    activate_contact_sensors=activate_contact_sensors,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        articulation_enabled=False
                    )
                ),  
            )
            object = RigidObject(cfg)
            self.rigid_objects[k] = object
            self.scene.rigid_objects[k] = object # can I directly assign the dict?


        self.target_object : RigidObject = self.rigid_objects[self.target_object_name]
        self.target_site = self.rigid_objects[self.target_site_name]

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        left_contact_sensor_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/panda_leftfinger", 
            update_period=0.0,
            history_length=2*self.history_len,
            track_air_time=True,
            filter_prim_paths_expr=[f"/World/envs/env_.*/{self.target_object_name}/object"],
            track_contact_points=True,
            max_contact_data_count_per_prim=4, # What is this
        )
        right_contact_sensor_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/panda_rightfinger", 
            update_period=0.0,
            history_length=2*self.history_len, 
            track_air_time=True,
            filter_prim_paths_expr=[f"/World/envs/env_.*/{self.target_object_name}/object"],
            track_contact_points=True,
            max_contact_data_count_per_prim=4,

        )

        self._left_contact_sensors = ContactSensor(left_contact_sensor_cfg)
        self._right_contact_sensors = ContactSensor(right_contact_sensor_cfg)
        self.scene.sensors["left_contact_sensor"] = self._left_contact_sensors
        self._left_contact_sensors.set_debug_vis(True)
        self.scene.sensors["right_contact_sensor"] = self._right_contact_sensors
        self._right_contact_sensors.set_debug_vis(True)

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions = self.actions.clone()
        # actions = torch.cat([actions,actions[:,-1].unsqueeze(1)], dim=1) # duplicate the gripper motion
        self.actions = actions.clone().clamp(-1.0, 1.0)
        arm_actions = actions[:, :-1]
        grip_action = actions[:, -1]

        # Arm: continuous
        arm_targets = self.robot_dof_targets[:, :-2] + (
            self.robot_dof_speed_scales[:-2] * self.dt * arm_actions * self.cfg.action_scale
        )
        self.robot_dof_targets[:, :-2] = torch.clamp(
            arm_targets,
            self.robot_dof_lower_limits[:-2],
            self.robot_dof_upper_limits[:-2]
        )

        # Gripper: binary
        open_mask = grip_action >= 0
        self.robot_dof_targets[open_mask, -2:] = self.robot_dof_upper_limits[-1]
        self.robot_dof_targets[~open_mask, -2:] = self.robot_dof_lower_limits[-1]
        # targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        # self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)


    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # manual update for observation needed values
        self._compute_intermediate_values()
        self.to_desired_rot = quat_mul(quat_conjugate(self.robot_grasp_rot), self.q_rel)
        self.manipulability = self._compute_manipulability()
        self.grasped = self._grasp_detection() # [num_envs]
        self._update_past_relative_dist()

        # condition for termination: for dropping the object in the basket
        self.low_enough = self.target_object.data.root_pos_w[:, 2] <self.target_site_corners_world[1,2] # change the harded coded height
        obj_xy = self.target_object.data.root_pos_w[:, :2]
        site_pos = self.target_site.data.root_pos_w[:, :2]
        dist2 = ((obj_xy - site_pos)**2).sum(dim=-1)
        self.inside_site = dist2 < self.target_site_radius**2
        # terminated = inside_site & low_enough
        # self.high_enough = object_lowest_point > site_height

        # condition for picking up the object

        # # quaternions are (w, x, y, z)
        target_object_current_rot = self.target_object.data.root_quat_w        # shape (N, 4)
        target_object_desired_rot = self.target_object.data.default_root_state[:,3:7]         # shape (N, 4)
        # inverse of current
        target_object_current_rot_inv = quat_conjugate(target_object_current_rot)
        # relative rotation
        q_error = quat_mul(target_object_desired_rot, target_object_current_rot_inv)
        q_error = q_error / torch.norm(q_error, dim=-1, keepdim=True).clamp_min(1e-9)
        q_error = torch.where(q_error[:, 0:1] < 0, -q_error, q_error)
        angle_error = 2 * torch.acos(torch.clamp(q_error[:, 0], -1.0, 1.0))
        self.small_rotation = angle_error < 0.8

        lowest_z_object= self._get_target_object_lowest_points()
        self.high_enough = lowest_z_object > self.target_site_corners_world[1,2] + 0.02
        # terminated = self.high_enough & self.grasped
        terminated = self.inside_site & self.low_enough

        # print(f"small_rotation{small_rotation}")
        # print(f"high_enough{high_enough}")
        # print(f"grasped{grasped}")
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        # terminated_count = high_enough.sum().item()
        # truncated_count = truncated.sum().item()

        # ratio = terminated_count / (truncated_count + 1e-8)
        # print(ratio)

        self.stage = self._current_stage_detection()

        if self.debug_vis:
            self.debug_vis_mine()

        if self.log_mine:
            for i in range(20):
                pass
                # target_object_current_pose = self.target_object.data.root_quat_w        # shape (N, 4)
                # target_object_desired_pose = object_default_state[:,3:7]         # shape (N, 4) 
                # # inverse of current
                # target_object_current_pose_inv = quat_conjugate(target_object_current_pose)
                # # relative rotation
                # q_error = quat_mul(target_object_desired_pose, target_object_current_pose_inv)
                # q_error = q_error / torch.norm(q_error, dim=-1, keepdim=True).clamp_min(1e-9)
                # q_error = torch.where(q_error[:, 0:1] < 0, -q_error, q_error)
                # angle_error = 2 * torch.acos(torch.clamp(q_error[:, 0], -1.0, 1.0))
                # self.summary_writer_mine.add_scalars(
                #     f"q_error/env_{i}",
                #     {
                #         "grasped": angle_error[i].item(),
                #     },
                #     self._sim_step_counter
                # )

        
        return terminated, truncated
    
    def _get_rewards(self) -> torch.Tensor:
            

        # rewards_pick, individual_rewards = self._get_rewards_test_pick()
        # rewards_pick /=10 # around 2
        # rewards_put, _ = self._get_rewards_test_put()   
        # rewards_put += 2
        # grasped_mask = self.grasped.bool()  # shape: (n,)
        # rewards = torch.where(grasped_mask, rewards_put, rewards_pick)
        # rewards = rewards_pick


        rewards, individual_rewards = self._get_rewards_test()


        if self.log_mine:
            for k,v in individual_rewards.items():
                for i in range(20):
                    self.summary_writer_mine.add_scalar(k, v[i].item(), self._sim_step_counter) 

        self.extras["log"] = individual_rewards
        return rewards


    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        self.update_rate += 1
        # if self.update_rate%2 == 0:
        # update default root states for random initialization
        rand_episode_idx = torch.randint(low=0, high=50, size=(1,), device=self.device).item()
        if self.start_in_air:
            start_idx_in_episode = self.base[rand_episode_idx]
        else:
            start_idx_in_episode = 0
        rand_int = torch.randint(low=0, high=21, size=(1,), device=self.device).item()
        idx = start_idx_in_episode + rand_int
        # print(idx)
        # print(len(self.data['franka'][rand_episode_idx]["states"]))
        init_states = self.data['franka'][rand_episode_idx]["states"][idx]
        robot_data = init_states['franka'] # seems that joint pos for isaaclab is always positive
        robot_joint_pos = robot_data["dof_pos"]
        # print(f"robot_joint_pos:{robot_joint_pos}")
        robot_joint_pos['panda_finger_joint2'] = robot_joint_pos['panda_finger_joint1']
        # print(f"robot_joint_pos:{robot_joint_pos}")
        robot_joint_pos = {k: v.item() for k, v in robot_joint_pos.items()}
        joint_values = torch.tensor(list(robot_joint_pos.values()), device=self.device)
        self._robot.data.default_joint_pos[env_ids] = joint_values

        for object_name in self.object_names:
            object = self.rigid_objects[object_name]
            pos = torch.tensor(init_states[object_name]["pos"], device=self.device)
            rot = torch.tensor(init_states[object_name]["rot"], device=self.device)

            object.data.default_root_state[env_ids, 0:3] = pos
            object.data.default_root_state[env_ids, 3:7] = rot

        # robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids] # TODO: recover randomization and test
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # for objects
        for object_name in self.object_names:
            object = self.rigid_objects[object_name]
            object_default_state = object.data.default_root_state.clone()[env_ids]

            object_default_state[:, 0:3] = (
                object_default_state[:, 0:3] + self.scene.env_origins[env_ids]
            )

            object_default_state[:, 7:] = torch.zeros_like(object.data.default_root_state[env_ids, 7:])
            object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
            object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)

        self._compute_intermediate_values(env_ids=env_ids)
        q_rel = self.q_rel[env_ids]
        self.to_desired_rot[env_ids] = quat_mul(quat_conjugate(self.robot_grasp_rot[env_ids]), q_rel)
        self.helper_variable[env_ids].zero_()
        self.grasped[env_ids].fill_(False) # stop updating with grasp detection cause no physics simulation
        self.stage[env_ids].zero_()
        self.past_relative_dist[env_ids].zero_()
        self.prev_actions[env_ids].zero_()


    def _get_observations(self) -> dict:
        """
        All texts in _get_observations() are very important hints for the task!

        Objects' root_pos_w is their center, and site's root_pos_w is the bottom center.  
        self.rigid_objects is a list with all RigidObjects
        Action space is 8-dim, and gripper action is symmetric: last dim being larger than 0 means opening gripper.
        self.to_desired_rot, # the quaterion that goes from current to desired quat, apply reward on this: specifically high reward for the first element w to be 1!!!
        self.helper_variable = torch.zeros((self.num_envs, 10), device=self.device)
        self.target_site_corners_world # you can read the three dim from the [2, 3] tensor storing min(first row) and max corner of the site in local env frame
        self.stage=self._current_stage_detection() # [self.num_envs, 5]

        def _pregrasp_detection(self, env_ids: torch.Tensor | None = None):
            if env_ids is None:
                env_ids = self._robot._ALL_INDICES

            current_pose = self.robot_grasp_rot  # shape (N, 4)
            desired_pose = self.q_rel   # shape (N, 4)
            # inverse of current
            current_pose_inv = quat_conjugate(current_pose)
            # relative rotation
            q_error = quat_mul(desired_pose, current_pose_inv)
            q_error = q_error / torch.norm(q_error, dim=-1, keepdim=True).clamp_min(1e-9)
            q_error = torch.where(q_error[:, 0:1] < 0, -q_error, q_error)
            angle_error = 2 * torch.acos(torch.clamp(q_error[:, 0], -1.0, 1.0))
            small_rotation = angle_error < 0.2

            pregrasp = dist_small_xy
            return pregrasp 
        

        def _current_stage_detection(self, env_ids: torch.Tensor | None = None):
            if env_ids is None:
                env_ids = self._robot._ALL_INDICES
            obj_xy = self.target_object.data.root_pos_w[:, :2]
            site_pos = self.target_site.data.root_pos_w[:, :2]
            dist2 = ((obj_xy - site_pos)**2).sum(dim=-1)
            # self.inside_site = dist2 < (2*self.target_site_radius)**2

        
        
            # Stage 1: pregrasp
            stage = self._pregrasp_detection().float()

            # Stage 2:
            stage = torch.where(
                self.grasped,
                torch.full_like(stage, 2.0),
                stage
            )

            # Stage 3:
            stage = torch.where(
                self.grasped & self.high_enough,
                torch.full_like(stage, 3.0),
                stage
            )

            # Stage 4: inside_site
            stage = torch.where(
                self.inside_site,
                torch.full_like(stage, 4.0),
                stage
            )

            stage_long = stage.long()
            stage_one_hot = torch.nn.functional.one_hot(stage_long, num_classes=self.num_stages).float()

            return stage_one_hot # [self.num_envs, 5]
            )
        """


        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        self.corners_target_obj_to_hand_pos =  (self.corners_target_obj - self.robot_grasp_pos.unsqueeze(1)).reshape(self.num_envs, -1) # distance from corners of target objects to robot, described in world coordinate.
        self.target_to_hand_pos = self.target_object.data.root_pos_w - self.robot_grasp_pos # [self.num_envs, 3]
        tcp_vel = (self._robot.data.body_link_lin_vel_w[:,self.left_finger_body_idx] + self._robot.data.body_link_lin_vel_w[:,self.right_finger_body_idx])/2
        target_to_hand_vel = self.target_object.data.root_lin_vel_w - tcp_vel

        self.site_to_target_pos = self.target_site.data.root_pos_w - self.target_object.data.root_pos_w
        site_to_target_vel = self.target_object.data.root_lin_vel_w - self.target_site.data.root_lin_vel_w
        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
                self.corners_target_obj_to_hand_pos, # this should be small
                self.target_to_hand_pos, # relative position from target object center to hand should be small
                self.to_desired_rot,
                self.site_to_target_pos, # relative position from target site to target object
                target_to_hand_vel,
                site_to_target_vel,
                self.stage,
                self.prev_actions - self.actions # add penalty on delta action
            ),
            dim=-1,
        )

        return {"policy": torch.clamp(obs, -5.0, 5.0)}


    def _compute_rewards(
        self,
        actions,
        action_penalty_scale,
    ):
    
        actions_copy = torch.zeros_like(actions) * self.common_step_counter # disable
        # regularization on the actions (summed for each environment)
        action_penalty = torch.sum(actions_copy**2, dim=-1)
        rewards = (
            - action_penalty_scale * action_penalty
        )

        self.extras["log"] = {
            "action_penalty": (-action_penalty_scale * action_penalty).mean(),
        }

        rewards = self.hand_quat[:,1].abs() + self.hand_quat[:,2].abs() + action_penalty_scale * action_penalty
        return rewards



    def run_replay(self, log_dir:str, render:bool=False, detect_grasp:bool=False):
        libero_dt = 1/20
        import numpy as np
        #TODO: make sure number of envs are enough for replay
        print(f"REPLAY LOG DIR {log_dir}")
        num_episodes = len(self.episodes)
        if num_episodes > self.num_envs:
            raise ValueError(
                f"num_episodes ({num_episodes}) "
                f"exceeds self.num_envs ({self.num_envs})"
            )
        env_ids = torch.arange(num_episodes, device=self.device, dtype=torch.long)
        self._reset_idx(env_ids) # reset twice to make sure the states are correct
        # [num_envs, step, states]
        # first loop over, add padding, then stack.
        writer = SummaryWriter(log_dir)
        

        eureka_episode_sums = dict()
        eureka_episode_sums["eureka_total_rewards"] = torch.zeros(num_episodes, device=self.device)
        eureka_episode_sums["oracle_total_rewards"] = torch.zeros(num_episodes, device=self.device)

        num_objects = len(self.object_names)

        # 2. Compute max episode length
        lengths = [len(ep["states"]) for ep in self.episodes]
        max_len = max(lengths)
        frequency_ratio = libero_dt / (self.cfg.decimation * self.cfg.sim.dt)
        max_replay_episode_length = (max_len - 1) * frequency_ratio + 1
        if max_replay_episode_length > self.max_episode_length:
            raise ValueError(
                f"Replay episode length ({max_replay_episode_length}) "
                f"exceeds max_episode_length ({self.max_episode_length})"
            )
        
        start_step = self.start_idx_in_episode * frequency_ratio # In case the task doesn't start from the beginning.

        rest_episode_length = int(max_replay_episode_length - start_step)
        # 3. Determine state dimensions
        # robot_pos_dim = len(self.episodes[0]["states"][0]["franka"]["pos"])      # 3
        # robot_rot_dim = len(self.episodes[0]["states"][0]["franka"]["rot"])      # 4
        robot_dof_dim = len(self.episodes[0]["states"][0]["franka"]["dof_pos"])  # num_joints

        # rigid objects (all share same structure)
        one_obj = next(iter({k: v for k, v in self.sample.items() if k != "franka"}.values()))
        object_pos_dim = len(one_obj["pos"])              # 3
        object_rot_dim = len(one_obj["rot"])              # 4

        # 4. Preallocate tensors
        # robot_pos = torch.zeros((num_episodes, max_len, robot_pos_dim))
        # robot_rot = torch.zeros((num_episodes, max_len, robot_rot_dim))


        robot_dof = torch.zeros((self.num_envs,  rest_episode_length, robot_dof_dim),device=self.device,dtype=torch.float32)
        object_pos = torch.zeros((self.num_envs,  rest_episode_length, num_objects, object_pos_dim),device=self.device,dtype=torch.float32)
        object_rot = torch.zeros((self.num_envs,  rest_episode_length, num_objects, object_rot_dim),device=self.device,dtype=torch.float32)
        padding_mask = torch.ones((self.num_envs,  rest_episode_length),device=self.device, dtype=torch.bool)
        

        def lerp(a, b, u):
            return (1.0 - u) * a + u * b

        def slerp(q0, q1, u, eps=1e-8):
            """
            q0, q1: quaternions as (4,) arrays in (x,y,z,w) format (your data looks like that)
            """
            q0 = np.asarray(q0, dtype=np.float64)
            q1 = np.asarray(q1, dtype=np.float64)

            # Normalize (safe)
            q0 = q0 / (np.linalg.norm(q0) + eps)
            q1 = q1 / (np.linalg.norm(q1) + eps)

            # Ensure shortest path: if dot < 0, negate q1
            dot = np.dot(q0, q1)
            if dot < 0.0:
                q1 = -q1
                dot = -dot

            # If very close, fall back to lerp + renormalize
            if dot > 0.9995:
                q = lerp(q0, q1, u)
                return (q / (np.linalg.norm(q) + eps)).astype(np.float64)

            theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
            sin_theta_0 = np.sin(theta_0)
            theta = theta_0 * u
            sin_theta = np.sin(theta)

            s0 = np.sin(theta_0 - theta) / (sin_theta_0 + eps)
            s1 = sin_theta / (sin_theta_0 + eps)
            return (s0 * q0 + s1 * q1).astype(np.float64)

        def interp_state(s0, s1, u):
            out = {}

            # --- franka dof ---
            out["franka"] = {
                "pos": lerp(np.array(s0["franka"]["pos"]), np.array(s1["franka"]["pos"]), u).tolist(),
                "rot": slerp(s0["franka"]["rot"], s1["franka"]["rot"], u).tolist(),
                "dof_pos": {}
            }

            # Keep DOF key order consistent by iterating keys from s0
            for k in s0["franka"]["dof_pos"].keys():
                v0 = float(np.array(s0["franka"]["dof_pos"][k]).reshape(-1)[0])
                v1 = float(np.array(s1["franka"]["dof_pos"][k]).reshape(-1)[0])
                out["franka"]["dof_pos"][k] = np.array([lerp(v0, v1, u)], dtype=np.float64)

            # --- objects (everything besides franka) ---
            for name in s0.keys():
                if name == "franka":
                    continue
                out[name] = {
                    "pos": lerp(np.array(s0[name]["pos"]), np.array(s1[name]["pos"]), u).tolist(),
                    "rot": slerp(s0[name]["rot"], s1[name]["rot"], u).tolist()
                }

            return out

        def insert_two_between(states):
            """
            states: list of state dicts length T
            returns: list length (T-1)*3 + 1
            """
            new_states = []
            for t in range(len(states) - 1):
                s0, s1 = states[t], states[t+1]
                new_states.append(s0)
                new_states.append(interp_state(s0, s1, 1/3))
                new_states.append(interp_state(s0, s1, 2/3))
            new_states.append(states[-1])
            return new_states
        
        if detect_grasp:
            in_the_air_matrix = torch.full(
                (num_episodes,), -1, dtype=torch.long, device=self.device
            )
            print(f"in_the_air_matrix{in_the_air_matrix}")
        # 5. Fill tensors
        with torch.inference_mode():
            for env_idx, ep in enumerate(self.episodes): # each episode
                new_ep = ep["states"][self.start_idx_in_episode:].copy() # avoid changing in-place
                new_ep = insert_two_between(new_ep)
                ep_len = len(new_ep)  # updated length
                for t, state in enumerate(new_ep):
                    # robot
                    # robot_pos[env_idx, t] = torch.tensor(state["franka"]["pos"])
                    # robot_rot[env_idx, t] = torch.tensor(state["franka"]["rot"])

                    dof_vals = [v[0] for v in state["franka"]["dof_pos"].values()]
                    dof_vals[-1] = - dof_vals[-1]
                    # print(dof_vals)
                    robot_dof[env_idx, t] = torch.tensor(dof_vals)

                    # objects
                    for obj_name, obj_idx in self.object_indices.items():
                        obj = state[obj_name]
                        object_pos[env_idx, t, obj_idx] = torch.tensor(obj["pos"])
                        object_rot[env_idx, t, obj_idx] = torch.tensor(obj["rot"])
                # padding mask
                padding_mask[env_idx, :ep_len] = False


            # Next step: Set states, call reward function, log

            for t in range(rest_episode_length):
                # set states
                joint_pos = robot_dof[:,t,:]
                # print(joint_pos[0,-2:])
                joint_vel = torch.zeros_like(joint_pos)
                self._robot.write_joint_state_to_sim(joint_pos, joint_vel)
                for obj_name, obj_idx in self.object_indices.items():
                    object = self.rigid_objects[obj_name]
                    object_state = torch.zeros((self.num_envs, 13), device=self.device)
                    object_state[:,0:3] = object_pos[:,t,obj_idx,:] + self.scene.env_origins
                    object_state[:,3:7] = object_rot[:,t,obj_idx,:]
                    object.write_root_pose_to_sim(object_state[:, :7])
                    object.write_root_velocity_to_sim(object_state[:, 7:])
            
                # similar to step function
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                if render:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
                    if t%2 ==0:
                        self.sim.render()
                self.scene.update(dt=self.physics_dt)

               
                if detect_grasp:
                    # update self.robot_grasp_pos
                    # _,_ = self._get_dones()
                    # _ = self._get_observations()
                    # for i in range(50):
                    #     writer.add_scalar("Grasp/"+str(i), self.grasped[i], t) 
                    diff = self.target_object.data.body_pos_w.squeeze(1) - self.robot_grasp_pos
                    # print(diff[0]) # play the first episode to check the constant object-hand distance
                    diff_norm = torch.norm(diff, dim=-1)  # shape: (n,)
                    # 0.09 ketchup/ 0.03 cream_cheese 0.1 alphabet_soup
                    # stable_left_contact, stable_right_contact, stable_opposing_contact, always_small = self._grasp_detection()
                    # for k in range(50):
                    #     writer.add_scalars("grasp/"+str(k), 
                    #                     {
                    #                     "stable_left_contact": stable_left_contact[k], 
                    #                     "stable_right_contact": stable_right_contact[k], 
                    #                     "stable_opposing_contact": stable_opposing_contact[k], 
                    #                     "always_small": always_small[k], },
                    #                     t)

                    if t > 20:
                        obj_z = self.target_object.data.root_pos_w[env_ids, 2]  # (num_envs,)
                        in_air = obj_z > 0.1                              # (num_envs,) bool
                        new_air_envs = torch.nonzero(in_air & (in_the_air_matrix == -1), as_tuple=False).squeeze(-1)
                        in_the_air_matrix[new_air_envs] = t



                    # if self.target_object.data.root_pos_w[0,2] > 0.1: # check if the grasp is stable (object-hand distance is small) and the object is lifted up (z is large), which should happen at the end of episode when the agent learns to lift up the object while keeping a stable grasp
                    #     print("logging")
                    #     for k in range(50):
                    #         writer.add_scalars("Position_error/"+str(k), 
                    #                         {"x": diff[k,0], 
                    #                         "y": diff[k,1], 
                    #                         "z": diff[k,2], },
                    #                         t)
                    #     # print(f"timestep {t}: object in the air")
                    #     print(self.robot_grasp_rot[0])

                        
                    # print(diff_norm[0]) # should be small and constant if the grasp is stable, which is the case for most of the episode, except at the end when the object is lifted up and the grasp is broken. This matches with the observation that the agent learns to keep a stable grasp and lift up the object at the end of training.
                    # print(self.target_object.data.root_pos_w[0,2]) # also check the object position, should be lifted up at the end of episode
                    # _ = self._get_observations() # this will update the corners_target_obj_to_hand_pos, which is the relative position from object corners to hand in world coordinate, should be small if the grasp is stable. This is a more direct way to check the grasp stability, and also provides more information about the relative position between the hand and the object, which can be useful for reward design.
                    # print(self.corners_target_obj_to_hand_pos[0])


                # call reward function, should be from eureka
                if hasattr(self, "_get_rewards_eureka"):
                    # dict value size is equal to num of envs
                    # rewards_oracle = self._get_rewards_oracle()
                    rewards_eureka, rewards_dict = self._get_rewards_eureka()
                    mask = padding_mask[:, t]
                    for k, v in rewards_dict.items(): # if episode ends, the value should be zero
                        v.masked_fill_(mask, 0.0)
                    # rewards_oracle_replay =rewards_oracle[:num_episodes]
                    rewards_eureka_replay = rewards_eureka[:num_episodes]
                    rewards_dict_replay = { k: v[:num_episodes] for k, v in rewards_dict.items() }
                    eureka_episode_sums["eureka_total_rewards"] += rewards_eureka_replay
                    # eureka_episode_sums["oracle_total_rewards"] += rewards_oracle_replay
                    for key in rewards_dict_replay.keys():
                        if key not in eureka_episode_sums:
                            eureka_episode_sums[key] = torch.zeros(num_episodes, device=self.device)
                        eureka_episode_sums[key] += rewards_dict_replay[key]
                else:
                    print("WARNING: No function is named _get_rewards_eureka")
            if detect_grasp:
                print(in_the_air_matrix)
                # After all replay is done, devide by episode length, and record
            for k in eureka_episode_sums.keys():
                per_ep_value = eureka_episode_sums[k] /self.max_episode_length_s
                writer.add_scalar("Replay/"+k +"_mean", per_ep_value.mean().item(), t) 
                writer.add_scalar("Replay/"+k +"_std", per_ep_value.std().item(), t) 


                    # enhanced_feedback=True
                    # min_max_feedback=False
                    # for k in eureka_episode_sums.keys():
                    #     if enhanced_feedback:
                    #         if min_max_feedback:
                    #             writer.add_scalars("Replay/"+k, 
                    #                             {"mean": eureka_episode_sums[k].mean().item(), 
                    #                             "min": eureka_episode_sums[k].min().item(), 
                    #                             "max": eureka_episode_sums[k].max().item(), },
                    #                                 t) 
                    #         else:
                    #             mu = eureka_episode_sums[k].mean()
                    #             var = eureka_episode_sums[k].var(unbiased=False)
                    #             writer.add_scalars("Replay/"+k, 
                    #                             {"mu": mu.item(), 
                    #                             "var": var.item(), },
                    #                                 t) 
                    #     else:
                    #         writer.add_scalar("Replay/"+k, eureka_episode_sums[k].mean().item(), t)                        
            self._reset_idx(env_ids)




    def get_target_object_to_hand_pose(self, init_states: dict):
        # Firstly find out the relative transformation, and then adjust the object pose according to robot pose.
        # Assume that the scene has been initialized
        pass



    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        self._compute_robot_intermediate_values(env_ids)
        self._compute_target_object_corners(env_ids)
        
        # self._compute_manipulability(env_ids) # This is only needed for reward computatoin


    def _compute_robot_intermediate_values(self, env_ids: torch.Tensor):
        hand_pos = self._robot.data.body_pos_w[env_ids, self.hand_link_idx]
        hand_rot = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]
        self.robot_grasp_rot[env_ids], self.robot_grasp_pos[env_ids] = tf_combine(
            hand_rot, hand_pos, self.robot_local_grasp_rot[env_ids], self.robot_local_grasp_pos[env_ids]
        )

    def _compute_target_object_corners(self, env_ids: torch.Tensor):
        pos = self.target_object.data.root_pos_w[env_ids]
        quat= quat_conjugate(self.target_object.data.root_quat_w[env_ids])
        self.corners_target_obj[env_ids] = transform_points(points=self.local_corners_init[env_ids],
                                              pos=pos,
                                              quat=quat)
        self.local_centers[env_ids] = transform_points(points=self.local_centers_init[env_ids],
                                              pos=pos,
                                              quat=quat)
        
    def _update_past_relative_dist(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        new_dist = torch.linalg.norm(self.target_to_hand_pos[env_ids], dim=-1)
        self.past_relative_dist[env_ids] = torch.cat([new_dist.unsqueeze(1), self.past_relative_dist[env_ids, :-1]], dim=1)


    def _compute_manipulability(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        debug=False
        jacobians = self._robot.root_physx_view.get_jacobians() # shape [num_envs, num_bodies, task_space, joint_space]

        left_finger_jacobians = jacobians[:, self.left_finger_joint_idx]
        right_finger_jacobians= jacobians[:, self.right_finger_joint_idx]

            
        finger_jacobians = (left_finger_jacobians + right_finger_jacobians)/2
        j_j_T = finger_jacobians @ finger_jacobians.transpose(-1, -2)  # [E, 6, 6]
        det = torch.linalg.det(j_j_T)
        m = torch.sqrt(torch.clamp(det, min=1e-12))
        return m
        # Should handle the case for reset
        # Compare the analytical solution and that from autograd


    def _grasp_detection(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        # Distance must remain small over the full history window
        threshold = 0.04
        always_small = torch.all(self.past_relative_dist[env_ids] < threshold, dim=1)

        left_forces_hist = self.scene["left_contact_sensor"].data.force_matrix_w_history[env_ids]
        right_forces_hist = self.scene["right_contact_sensor"].data.force_matrix_w_history[env_ids]

        # Force magnitudes
        left_force_mag = torch.linalg.norm(left_forces_hist, dim=-1)
        right_force_mag = torch.linalg.norm(right_forces_hist, dim=-1)

        # Contact existence per history/body/contact slot
        left_contact = left_force_mag > 0
        right_contact = right_force_mag > 0

        # Require contact to be present at every past step / slot for both sides
        stable_left_contact = torch.all(left_contact, dim=tuple(range(1, left_contact.ndim)))
        stable_right_contact = torch.all(right_contact, dim=tuple(range(1, right_contact.ndim)))

        # Normalize safely
        eps = 1e-8
        left_dir = left_forces_hist / (left_force_mag.unsqueeze(-1) + eps)
        right_dir = right_forces_hist / (right_force_mag.unsqueeze(-1) + eps)

        # Dot product < 0 means opposing directions
        dir_dot = torch.sum(left_dir * right_dir, dim=-1)

        # Only valid where both sides actually have contact
        mutual_contact = left_contact & right_contact

        # Require all past contacts to be opposing / pointing toward each other
        # You can tighten threshold to e.g. dir_dot < -0.3 if needed
        opposing_contact = mutual_contact & (dir_dot < 0.0)

        # For a valid grasp, every historical contact entry must be stable and opposing
        stable_opposing_contact = torch.all(
            opposing_contact, dim=tuple(range(1, opposing_contact.ndim))
        )
        # print("grasp: ",stable_opposing_contact, always_small)
        grasped = (
            stable_left_contact
            & stable_right_contact
            & stable_opposing_contact
            & always_small
        )

        return grasped


   

    def _get_target_object_lowest_points(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        # shape: (N, num_corners, 3)
        corners = self.corners_target_obj[env_ids]
        # Extract z coordinates
        z_coords = corners[..., 2]
        # Get minimum z per environment
        lowest_z, _ = torch.min(z_coords, dim=1) # num_corners
        return lowest_z
    

    def _get_inside_site(self, env_ids: torch.Tensor):
        obj_xy = self.target_object.data.root_pos_w[env_ids, :2]
        site_pos = self.target_site.data.root_pos_w[env_ids, :2]
        dist2 = ((obj_xy - site_pos)**2).sum(dim=-1)
        inside_site = dist2 < self.target_site_radius**2
        return inside_site
    

    def _pregrasp_detection(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        
        pos_diff = self.robot_grasp_pos - self.target_object.data.root_pos_w
        pos_diff[:,2] -= 0.2
        dist_small_z = torch.abs(pos_diff[:, 2]) < 0.05
        dist_small_xy = torch.norm(pos_diff[:,:2],dim=-1) < 0.02


        current_pose = self.robot_grasp_rot  # shape (N, 4)
        desired_pose = self.q_rel   # shape (N, 4)
        # inverse of current
        current_pose_inv = quat_conjugate(current_pose)
        # relative rotation
        q_error = quat_mul(desired_pose, current_pose_inv)
        q_error = q_error / torch.norm(q_error, dim=-1, keepdim=True).clamp_min(1e-9)
        q_error = torch.where(q_error[:, 0:1] < 0, -q_error, q_error)
        angle_error = 2 * torch.acos(torch.clamp(q_error[:, 0], -1.0, 1.0))
        small_rotation = angle_error < 0.2

        pregrasp = dist_small_xy
        return pregrasp 
    

    def _current_stage_detection(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
       
        # Stage 1: pregrasp
    
        stage = self._pregrasp_detection().float()

        # Stage 2: cautious grasp
        stage = torch.where(
            self.grasped,
            torch.full_like(stage, 2.0),
            stage
        )
        # Stage 3: grasped AND height > 0.10 m
        
        stage = torch.where(
            self.grasped & self.high_enough,
            torch.full_like(stage, 3.0),
            stage
        )

        # Stage 4: grasped AND inside_site in x y direction
        stage = torch.where(
            self.inside_site,
            torch.full_like(stage, 4.0),
            stage
        )

        # stage values are 1,2,3 → subtract 1 to index from 0
        stage_long = stage.long()
        stage_one_hot = torch.nn.functional.one_hot(stage_long, num_classes=self.num_stages).float()


        if self.log_mine and self.log_stage:
            for i in range(20):
                self.summary_writer_mine.add_scalars(
                    f"r_lift_progress/env_{i}",
                    {
                        "grasped": self.grasped.float()[i].item(),
                        "stage": stage[i].item(),
                        "inside_site": self.inside_site.float()[i].item(),
                        "self.high_enough": self.high_enough.float()[i].item(),
                    },
                    self._sim_step_counter
                )

        return stage_one_hot # [self.num_envs, 5]


    def define_markers(self) -> VisualizationMarkers:
        """Define markers with various different shapes."""
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/myMarkers",
            markers={
                "frame": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                    scale=(0.1, 0.1, 0.1),
                ),
            },
        )
        return VisualizationMarkers(marker_cfg)

    def debug_vis_mine(self):
        translations_list = [
            x for flag, x in [
                (self.show_robot_grasp, self.robot_grasp_pos),
                (self.show_target_object, self.target_object.data.root_pos_w),
                (self.show_target_grasp_pose, self.target_object.data.root_pos_w),
            ] if flag
        ]
        translations = torch.cat(translations_list, dim=0) if translations_list else None
        
        orientations_list = [
            x for flag, x in [
                (self.show_robot_grasp, self.robot_grasp_rot),
                (self.show_target_object, self.target_object.data.root_quat_w),
                (self.show_target_grasp_pose, self.q_rel),
            ] if flag
        ]
        orientations = torch.cat(orientations_list, dim=0) if orientations_list else None
        self.visualizer.visualize(translations=translations,
                                orientations=orientations,
                                )

    def _get_rewards_test_manipulability(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        import torch

        eps = 1e-8

        # ------------------------------------------------------------------
        # Inputs
        # ------------------------------------------------------------------
        tcp_pos = torch.nan_to_num(self.robot_grasp_pos)
        target_pos = torch.nan_to_num(self.target_object.data.root_pos_w.clone())
        target_pos[:, 2] += 0.2

        manipulability = torch.nan_to_num(self.manipulability).clamp(min=0.0)

        # ------------------------------------------------------------------
        # Position tracking
        # ------------------------------------------------------------------
        pos_delta = tcp_pos - target_pos
        pos_error = torch.linalg.norm(pos_delta, dim=-1)

        # Gaussian-style reward: smoother and more standard than exp(-d / c)
        pos_sigma = 0.10
        r_position = torch.exp(-(pos_error ** 2) / (2 * pos_sigma ** 2))

        # Optional: success-style bonus when very close
        pos_close_thresh = 0.03
        r_pos_bonus = (pos_error < pos_close_thresh).float()

        # ------------------------------------------------------------------
        # Orientation tracking
        # Assumes self.to_desired_rot is the relative quaternion from current to desired
        # Quaternion format assumed: [w, x, y, z]
        # ------------------------------------------------------------------
        q_error = torch.nan_to_num(self.to_desired_rot)
        q_error = q_error / torch.linalg.norm(q_error, dim=-1, keepdim=True).clamp_min(eps)

        # Resolve quaternion sign ambiguity
        q_error = torch.where(q_error[:, 0:1] < 0.0, -q_error, q_error)

        # Angle in [0, pi]
        qw = torch.clamp(q_error[:, 0], -1.0 + eps, 1.0 - eps)
        angle_error = 2.0 * torch.acos(qw)
        # print(angle_error)
        # print(self.robot_grasp_rot)
        ori_sigma = 0.30
        r_orientation_raw = torch.exp(-(angle_error ** 2) / (2 * ori_sigma ** 2))

        # Gate orientation reward by position proximity
        # This prevents the policy from trying to perfectly orient while still far away
        orientation_gate = torch.exp(-(pos_error ** 2) / (2 * (0.15 ** 2)))
        r_orientation = orientation_gate * r_orientation_raw

        # Optional orientation success bonus
        ori_close_thresh = 0.15  # radians
        r_ori_bonus = ((angle_error < ori_close_thresh) & (pos_error < 0.05)).float()

        # ------------------------------------------------------------------
        # Manipulability reward
        # Keep small so it does not dominate tracking
        # ------------------------------------------------------------------
        r_manipulability = torch.tanh(manipulability)

        # ------------------------------------------------------------------
        # Final reward
        # ------------------------------------------------------------------
        reward = (
            2.5 * r_position +
            1.5 * r_orientation +
            0.1 * r_manipulability
        )

        individual_rewards = {
            # "position_tracking": r_position,
            # "position_bonus": r_pos_bonus,
            # "orientation_tracking": r_orientation,
            # "orientation_tracking_raw": r_orientation_raw,
            # "orientation_gate": orientation_gate,
            # "orientation_bonus": r_ori_bonus,
            # "manipulability": r_manipulability,
            "pos_error": pos_error,
            "angle_error": angle_error,
        }

        return reward, individual_rewards


    def _get_rewards_test(self):
        import torch

        stage0_mask = self.stage[:, 0]
        stage1_mask = self.stage[:, 1]
        stage2_mask = self.stage[:, 2]
        stage3_mask = self.stage[:, 3]
        stage4_mask = self.stage[:, 4]
        
        stage1_bonus = stage1_mask.float() * 1
        stage2_bonus = stage2_mask.float() * 2 # grasped
        stage3_bonus = stage3_mask.float() * 3
        stage4_bonus = stage4_mask.float() * 4
        self.helper_variable[:, 0] = torch.max(self.helper_variable[:, 0], stage0_mask)  # If stage0 pregrasp has ever been reached

        # target position: 20 cm above object
        target_pos = self.target_object.data.root_pos_w.clone()
        target_pos[:, 2] += 0.2

        pos_diff = self.robot_grasp_pos - target_pos
        xy_err = torch.norm(pos_diff[:, :2], dim=-1)
        z_err = torch.abs(pos_diff[:, 2])
        pos_error = torch.norm(pos_diff, dim=-1)

        # smoother position shaping
        r_xy = torch.exp(-xy_err / 0.10)
        r_z = torch.exp(-z_err / 0.10)
        r_pos = torch.exp(-pos_error / 0.10)

        # rotation error
        current_q = self.robot_grasp_rot
        desired_q = self.q_rel

        current_q_inv = quat_conjugate(current_q)
        q_error = quat_mul(desired_q, current_q_inv)
        q_error = q_error / torch.norm(q_error, dim=-1, keepdim=True).clamp_min(1e-9)
        # assumes [w, x, y, z]
        q_error = torch.where(q_error[:, 0:1] < 0, -q_error, q_error)
        angle_error = 2.0 * torch.acos(torch.clamp(q_error[:, 0], -1.0, 1.0))

        r_rot = torch.exp(-angle_error / 2)

        stage0_reward = stage0_mask * (0.5 * r_pos + 0.5 * r_rot)




        pos_diff_stage_1 = self.robot_grasp_pos - self.target_object.data.root_pos_w
        pos_error_stage_1 = torch.norm(pos_diff_stage_1, dim=-1)

        # gripper action
        gripper_action = self.actions[:, -1]

        # open reward (target = +1)
        open_error = torch.abs(gripper_action - 1.0)
        open_reward = torch.exp(-open_error / 0.5)
        open_reward *= 0.2

        # close reward (target = -1)
        close_error = torch.abs(gripper_action + 1.0)
        close_reward = torch.exp(-close_error / 0.5)
        close_reward += 0.2  # keep your monotonic shaping

        # condition stays the same
        close_condition = pos_error_stage_1 < 0.04

        # final reward
        gripper_reward = torch.where(close_condition, close_reward, open_reward)
        # if close_condition.float().mean() > 0.6:
        #     pdb.set_trace()

        # reward descending while staying close in xy
        # pos_diff[:, 2] > 0 means gripper is above the target
        above_target = torch.clamp(pos_error_stage_1, min=0.0)
        descend_reward = (torch.exp(-above_target / 0.1) - torch.exp(torch.tensor(-1.0, device=self.device))) # not constrained to 

        stage1_reward = stage1_bonus + stage1_mask * (
            0.4 * gripper_reward +
            0.5 * descend_reward +
            0.1 * r_rot 

        )

        lowest_z_object= self._get_target_object_lowest_points()
        ascend_diff = lowest_z_object - self.target_site_corners_world[1,2] - 0.05
        ascend_reward = torch.exp(- torch.abs(ascend_diff)/ 0.1)
        # print(self.high_enough)
        stage2_reward = stage2_bonus + stage2_mask * (
            1 * ascend_reward
        )



        pos_diff_site = self.target_object.data.root_pos_w - self.target_site.data.root_pos_w
        pos_diff_site_xy = pos_diff_site[:,:2]
        xy_err_site = torch.norm(pos_diff_site_xy, dim=-1)
        xy_reward = torch.exp(-xy_err_site / 0.10)
        # print(xy_err_site)
        stage3_reward = stage3_bonus + stage3_mask * (
            0.2 * ascend_reward +
            0.8 * xy_reward
        )




        vel_xy = torch.norm(self.target_object.data.root_lin_vel_w[:,:2], dim=-1)
        vel_gate = vel_xy < 0.6
        vel_reward = torch.exp(-vel_xy / 0.2)
        stage4_reward = stage4_bonus + stage4_mask * (
            0.4 * xy_reward +
            # 0.3 * vel_reward +
            0.3 * vel_gate * open_reward * 5 # Was scaled before
        )

        success = self.inside_site & self.low_enough
        success_reward = 16 * 40 * success


        pos_delta = torch.norm(self.prev_actions - self.actions, dim=-1)

        reward = (stage0_reward + 
        stage1_reward + 
        stage2_reward + 
        stage3_reward + 
        stage4_reward + 
        success_reward -
        pos_delta * 0.01
        )
        return reward, {
            "xy_err": xy_err,
            "z_err": z_err,
            "pos_error": pos_error,
            "angle_error": angle_error,
            "r_xy": r_xy,
            "r_z": r_z,
            "r_rot": r_rot,
            "pos_error_stage_1": pos_error_stage_1,
            "gripper_pos": self._robot.data.joint_pos[:,-1],
            "ascend_diff": ascend_diff,
            "xy_err_site" : xy_err_site,
            "vel_xy": vel_xy,
            "open_reward": open_reward,
            "close_reward": close_reward,
            "gripper_reward": gripper_reward,
            "descend_reward": descend_reward,
            "close_condition": close_condition.float(),
            "ascend_reward": ascend_reward,
            "xy_reward": xy_reward,
            "stage0_mask": stage0_mask,
            "stage1_mask": stage1_mask,
            "stage2_mask": stage2_mask,
            "stage3_mask": stage3_mask,
            "stage4_mask": stage4_mask,
            "stage0_reward": stage0_reward,
            "stage1_reward": stage1_reward,
            "stage2_reward": stage2_reward,
            "stage3_reward": stage3_reward,      
            "stage4_reward": stage4_reward,  
            "success": success,  
        }   

