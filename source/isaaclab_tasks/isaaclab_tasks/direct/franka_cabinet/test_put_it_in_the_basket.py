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

import pdb
from isaaclab_eureka.utils import eureka_root_dir, read_pkl

@configclass
class TestPutItInTheBasketCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.65  # 519 timesteps
    decimation = 2
    action_space = 9
    observation_space = 25 # to modify
    state_space = 0

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

    action_scale = 7.5
    dof_velocity_scale = 0.1

    # reward scales
    dist_reward_scale = 1.5
    rot_reward_scale = 1.5
    open_reward_scale = 10.0
    action_penalty_scale = 0.05
    finger_reward_scale = 2.0


class TestPutItInTheBasket(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: TestPutItInTheBasketCfg

    def __init__(self, cfg: TestPutItInTheBasketCfg, render_mode: str | None = None, **kwargs):
    


        self.root = eureka_root_dir()
        self.target_object_name = "alphabet_soup"
        self.target_site_name = "basket"
        self.input_direction = 2 # z axis
        if self.input_direction != 2:
            raise NotImplementedError("Change the _get_dones method and other calculations for deciding the entry size")
        self.path = f"{self.root}/libero/trajs/libero90/libero_90_living_room_scene1_pick_up_the_{self.target_object_name}_and_put_it_in_the_basket_traj_v2.pkl"

        start_idx_in_episode_dict = {
            "alphabet_soup":80,
            "cream_cheese":80,
            "ketchup":80,
            "tomato_sauce":90,
        }
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

        # prim = stage.GetPrimAtPath(f"/World/envs/env_0/Robot/panda_leftfinger")
        # bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(prim)
        # min_wc = bbox.GetRange().GetMin()
        # max_wc = bbox.GetRange().GetMax()



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



        # prim = stage.GetPrimAtPath(f"/World/envs/env_0/{self.target_object_name}/object")
        # bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(prim)
        # min_corner = bbox.GetRange().GetMin()
        # max_corner = bbox.GetRange().GetMax()

        # min_corner = torch.tensor(
        #     [min_corner[0], min_corner[1], min_corner[2]],
        #     dtype=torch.float32,
        #     device=self.device
        # )

        # max_corner = torch.tensor(
        #     [max_corner[0], max_corner[1], max_corner[2]],
        #     dtype=torch.float32,
        #     device=self.device
        # )
        # min_max_corners_world = torch.stack([min_corner, max_corner], dim=0).unsqueeze(0)  # [1, 2, 3]
        # min_max_corners_world = min_max_corners_world.repeat((self.num_envs, 1, 1))  # [num_envs, 2, 3]
        # pos =  self.target_object.data.root_pos_w
        # quat= quat_conjugate(self.target_object.data.root_quat_w)
        # min_max_corners_obj = transform_points(points=min_max_corners_world,
        #                                       pos=pos,
        #                                       quat=quat)
        
        # print(f"min_max_corners_obj{min_max_corners_obj}")
        # min_corner = min_max_corners_obj[0]
        # max_corner = min_max_corners_obj[1]

        # local_corners = torch.tensor([
        #     [min_corner[0], min_corner[1], min_corner[2]],
        #     [max_corner[0], min_corner[1], min_corner[2]],
        #     [min_corner[0], max_corner[1], min_corner[2]],
        #     [min_corner[0], min_corner[1], max_corner[2]],
        #     [max_corner[0], max_corner[1], min_corner[2]],
        #     [max_corner[0], min_corner[1], max_corner[2]],
        #     [min_corner[0], max_corner[1], max_corner[2]],
        #     [max_corner[0], max_corner[1], max_corner[2]],
        # ], device=self.device)

        # self.target_object_size = torch.tensor((max_corner - min_corner), device=self.device)
        # # local_corners = local_corners - self.scene.env_origins[0, :]  # in the local world frame
        
        # # # compute center (now torch math)
        # local_centers_init = (max_corner + min_corner) / 2.0 -self.scene.env_origins[0, :]  # in the local world frame
        # self.local_centers_init = local_centers_init.repeat((self.num_envs, 1)).unsqueeze(1) # [num_envs, 1, 3] p = 1
        # self.local_centers = torch.zeros_like(self.local_centers_init)
        # # # repeat correctly (NO device argument)
        # # self.offset = (
        # #     local_center.unsqueeze(0)
        # #     .repeat(self.num_envs, 1)
        # #     - self.target_object.data.root_pos_w[0,:]
        # # )

        # self.local_corners_init = local_corners.unsqueeze(0).repeat(self.num_envs, 1, 1) # object local frame
        # self.corners_target_obj = torch.zeros_like(self.local_corners_init)

        # # self.corners_target_obj = transform_points(points=self.local_corners_init,
        # #                                       pos=pos,
        # #                                       quat=self.target_object.data.root_quat_w)
        # # # subtract translation
        # # shifted = world_points - pos.unsqueeze(1)

        # # # inverse rotation
        # # quat_inv = quat_conjugate(self.target_object.data.root_quat_w)

        # # local_points = quat_rotate(quat_inv, shifted)



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


    def _setup_scene(self):
        # init_states = self.data['franka'][0]["init_state"] # this is from the first scene as set up. Init states of traj should be updated in reset_idx
        # init_states = self.data['franka'][0]["states"][100]
        # self.start_idx_in_episode = 70 # for cream_cheese
        # self.start_idx_in_episode = 80 # For living room scene 1 pick up the alphabet soup
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
                    enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos=robot_joint_pos,
                pos=robot_data["pos"],
                rot=robot_data["rot"],
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
            else:
                activate_contact_sensors = False
            cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/{k}",
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=init_states[k]["pos"],
                    rot=init_states[k]["rot"],
                ),
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
            prim_path="/World/envs/env_.*/Robot/panda_leftfinger", update_period=0.0, history_length=10, 
            track_air_time=True, filter_prim_paths_expr=[f"/World/envs/env_.*/{self.target_object_name}/object"],
        )
        right_contact_sensor_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/panda_rightfinger", update_period=0.0, history_length=10, 
            track_air_time=True, filter_prim_paths_expr=[f"/World/envs/env_.*/{self.target_object_name}/object"],
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
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)


    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # manual update for observation needed values
        self._compute_intermediate_values()

        # condition for termination
        site_height = self.target_site_corners_world[1,2] - self.target_site_corners_world[0,2]
        low_enough = self.target_object.data.root_pos_w[:, 2] <site_height # change the harded coded height
        obj_xy = self.target_object.data.root_pos_w[:, :2]
        site_pos = self.target_site.data.root_pos_w[:, :2]
        dist2 = ((obj_xy - site_pos)**2).sum(dim=-1)
        inside_site = dist2 < self.target_site_radius**2
        terminated = inside_site & low_enough
        # terminated = self.target_object.data.root_pos_w[:, 2] > 0.4
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        # print(f"inside_site{inside_site}")
        # print(f"low_enough{low_enough}")
        return terminated, truncated
    
    def _get_rewards(self) -> torch.Tensor:
        rewards, individual_rewards = self._get_rewards_test()
        self.extras["log"] = individual_rewards
        return rewards


    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        self.update_rate += 1
        if self.update_rate%10 == 0:
        # update default root states for random initialization
            rand_episode_idx = torch.randint(low=0, high=50, size=(1,), device=self.device).item()
            start_idx_in_episode = self.base[rand_episode_idx]
            rand_int = torch.randint(low=0, high=21, size=(1,), device=self.device).item()
            idx = start_idx_in_episode + rand_int
            print(idx)
            print(len(self.data['franka'][rand_episode_idx]["states"]))
            init_states = self.data['franka'][rand_episode_idx]["states"][idx]
            robot_data = init_states['franka'] # seems that joint pos for isaaclab is always positive
            robot_joint_pos = robot_data["dof_pos"]
            # print(f"robot_joint_pos:{robot_joint_pos}")
            robot_joint_pos['panda_finger_joint2'] = robot_joint_pos['panda_finger_joint1']
            # print(f"robot_joint_pos:{robot_joint_pos}")
            robot_joint_pos = {k: v.item() for k, v in robot_joint_pos.items()}
            joint_values = torch.tensor(list(robot_joint_pos.values()), device=self.device)
            self._robot.data.default_joint_pos[env_ids] = joint_values.unsqueeze(0).repeat(len(env_ids), 1)

            for object_name in self.object_names:
                object = self.rigid_objects[object_name]
                pos = torch.tensor(init_states[object_name]["pos"], device=self.device)
                rot = torch.tensor(init_states[object_name]["rot"], device=self.device)

                object.data.default_root_state[env_ids, 0:3] = pos.unsqueeze(0).repeat(len(env_ids), 1)
                object.data.default_root_state[env_ids, 3:7] = rot.unsqueeze(0).repeat(len(env_ids), 1)


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
            # object_default_state = self.object_default_state.clone()[env_ids]
            object_default_state[:, 0:3] = (
                object_default_state[:, 0:3] + self.scene.env_origins[env_ids]
            )

            object_default_state[:, 7:] = torch.zeros_like(object.data.default_root_state[env_ids, 7:])
            object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
            object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)

        self._compute_intermediate_values(env_ids=env_ids)
        
        self.helper_variable[env_ids] = torch.zeros((len(env_ids), 10), device=self.device)


    def _get_observations(self) -> dict:
        # self.helper_variable = torch.zeros((self.num_envs, 10), device=self.device)
        # self.rigid_objects is a list with all RigidObjects
        # Objects' root_pos_w is their center, and site's root_pos_w is the bottom center.

        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        self.corners_target_obj_to_hand_pos =  (self.corners_target_obj - self.robot_grasp_pos.unsqueeze(1)).reshape(self.num_envs, -1) # corners to robot dist described in world coordinate.
        self.target_to_hand_pos = self.target_object.data.root_pos_w - self.robot_grasp_pos # n, 3
        self.hand_quat = self._robot.data.body_quat_w[:, self.hand_link_idx]
        tcp_vel = (self._robot.data.body_link_lin_vel_w[:,self.left_finger_body_idx] + self._robot.data.body_link_lin_vel_w[:,self.right_finger_body_idx])/2
        target_to_hand_vel = self.target_object.data.root_lin_vel_w - tcp_vel
        # add velocity

        self.site_to_target_pos = self.target_site.data.root_pos_w - self.target_object.data.root_pos_w
        site_to_target_vel = self.target_object.data.root_lin_vel_w - self.target_site.data.root_lin_vel_w
        # self.target_site_corners_world # you can read the three dim from the [2, 3] tensor storing min(first row) and max corner of the site in local env frame

        grasped = self._grasp_detection().float() # [num_envs, 1] 1 means two fingers have contact force against object, thereby grasping

        self.manipulability = self._compute_manipulability(self._robot._ALL_INDICES) # [num_envs] Always have reward on this to have correct robot motion
        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
                self.corners_target_obj_to_hand_pos, # this should be small
                self.target_to_hand_pos, # relative position from target object center to hand should be small
                self.hand_quat, # be close to inital orientation (pointing downwards is good, allows z axis rotation(Hint:When 0 and 3 element is zero, hand is pointing downwards)
                self.site_to_target_pos, # relative position in x y from target site to target object
                target_to_hand_vel,
                site_to_target_vel,
                grasped, # 1 or 0, indicating whether is grasped or not.
            ),
            dim=-1,
        )
        # print(self.target_object.data.root_pos_w[0,2])
        # print(self.scene["left_contact_sensor"].data.current_contact_time)
        # print(self.scene["right_contact_sensor"].data.current_contact_time)
        # print(self.scene.sensors["finger_contact_sensor"].data.force_matrix_w)
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



    def run_replay(self, log_dir:str, render:bool=False):
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
        
        detect_grasp = False
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
                    _,_ = self._get_dones()
                    diff = self.target_object.data.body_pos_w.squeeze(1) - self.robot_grasp_pos
                    # print(diff[0]) # play the first episode to check the constant object-hand distance
                    diff_norm = torch.norm(diff, dim=-1)  # shape: (n,)
                    # 0.09 ketchup/ 0.03 cream_cheese 0.1 alphabet_soup
                    
                    if t > 20:
                        obj_z = self.target_object.data.root_pos_w[env_ids, 2]  # (num_envs,)
                        in_air = obj_z > 0.1                              # (num_envs,) bool
                        new_air_envs = torch.nonzero(in_air & (in_the_air_matrix == -1), as_tuple=False).squeeze(-1)
                        in_the_air_matrix[new_air_envs] = t

                    if self.target_object.data.root_pos_w[0,2] > 0.1: # check if the grasp is stable (object-hand distance is small) and the object is lifted up (z is large), which should happen at the end of episode when the agent learns to lift up the object while keeping a stable grasp
                        print(f"timestep {t}: object in the air")
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


    def _compute_manipulability(self, env_ids: torch.Tensor):
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


    def _grasp_detection(self):
        grasped_left = self.scene["left_contact_sensor"].data.current_contact_time > 0
        grasped_right = self.scene["right_contact_sensor"].data.current_contact_time > 0
        grasped = grasped_left & grasped_right
        return grasped
   


    def _get_rewards_test(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        import torch

        device = self.device
        eps: float = 1e-6

        def _safe(x: torch.Tensor) -> torch.Tensor:
            return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # ------------------------------------------------------------
        # Fetch & sanitize (all tensors on self.device)
        # ------------------------------------------------------------
        obj_pos = _safe(self.target_object.data.root_pos_w)  # (N,3)
        obj_vel = _safe(self.target_object.data.root_lin_vel_w)  # (N,3)
        site_pos = _safe(self.target_site.data.root_pos_w)  # (N,3)
        site_vel = _safe(self.target_site.data.root_lin_vel_w)  # (N,3)

        robot_q = _safe(self._robot.data.joint_pos)  # (N,dof)
        robot_qd = _safe(self._robot.data.joint_vel)  # (N,dof)

        target_to_hand = _safe(self.target_to_hand_pos)  # (N,3)
        site_to_target = _safe(self.site_to_target_pos)  # (N,3)
        hand_quat = _safe(self.hand_quat)  # (N,4)

        grasped = _safe(self._grasp_detection().float())
        if grasped.ndim == 2 and grasped.shape[-1] == 1:
            grasped = grasped.squeeze(-1)
        grasped = torch.clamp(grasped, 0.0, 1.0)

        manipulability = torch.clamp(_safe(self.manipulability), 0.0, 1e3)

        # TCP vel (slip proxy)
        left_v = _safe(self._robot.data.body_link_lin_vel_w[:, self.left_finger_body_idx])
        right_v = _safe(self._robot.data.body_link_lin_vel_w[:, self.right_finger_body_idx])
        tcp_vel = 0.5 * (left_v + right_v)
        rel_vel = torch.linalg.norm(obj_vel - tcp_vel, dim=-1)

        # Optional action regularization (if available)
        act = getattr(self, "actions", None)
        if act is None:
            act_pen = torch.zeros((self.num_envs,), device=device)
        else:
            act = _safe(act)
            act_pen = torch.sum(torch.clamp(act, -1.0, 1.0) ** 2, dim=-1)

        # ------------------------------------------------------------
        # Geometry / success metric
        # ------------------------------------------------------------
        rad: float = float(self.target_site_radius)
        d_xy = torch.linalg.norm(site_to_target[:, :2], dim=-1)
        obj_z = obj_pos[:, 2]
        site_z = site_pos[:, 2]
        h = obj_z - site_z  # height above basket bottom plane

        inside_site = (d_xy * d_xy) < (rad * rad)
        low_enough = obj_z < 0.1
        success = (inside_site & low_enough).float()

        hand_obj_dist = torch.linalg.norm(target_to_hand, dim=-1)

        # ------------------------------------------------------------
        # helper_variable progress / memory (robust stage machine)
        # hv[0]=stage (0..5)
        # hv[1]=stable_cnt
        # hv[2]=prev_dxy
        # hv[3]=prev_h
        # hv[4]=prev_grasp
        # hv[5]=armed_cnt
        # hv[6]=tsr (time since proper release)
        # hv[7]=had_proper_release
        # hv[8]=success_latch
        # hv[9]=unused
        # ------------------------------------------------------------
        hv = self.helper_variable
        if hv is None or hv.shape[0] != self.num_envs or hv.shape[1] < 10:
            hv = torch.zeros((self.num_envs, 10), device=device)

        stage = _safe(hv[:, 0])
        stable_cnt = _safe(hv[:, 1])
        prev_dxy = _safe(hv[:, 2])
        prev_h = _safe(hv[:, 3])
        prev_grasp = torch.clamp(_safe(hv[:, 4]), 0.0, 1.0)
        armed_cnt = _safe(hv[:, 5])
        tsr = _safe(hv[:, 6])
        had_release = torch.clamp(_safe(hv[:, 7]), 0.0, 1.0)
        success_latch = torch.clamp(_safe(hv[:, 8]), 0.0, 1.0)

        prev_dxy = torch.where(prev_dxy > 0.0, prev_dxy, d_xy.detach())
        prev_h = torch.where(prev_h != 0.0, prev_h, h.detach())

        # Stable grasp proxy (slightly easier than previous; avoid blocking exploration)
        stable_grasp = (grasped > 0.5) & (rel_vel < 0.60) & (hand_obj_dist < 0.17)
        stable_cnt = torch.clamp(stable_cnt + stable_grasp.float(), 0.0, 300.0)
        stable_enough = stable_cnt >= 3.0

        # Stages / thresholds (make "place above basket" crisper to improve conversion to success)
        lift_h: float = 0.16
        place_h: float = 0.15  # height to be "above opening" before release

        lifted_now = (h > lift_h) & (grasped > 0.5) & stable_enough
        align_xy: float = max(0.028, 0.65 * rad)  # tighter than before to increase inside_site probability
        aligned_now = (d_xy < align_xy) & (h > place_h) & (grasped > 0.5) & stable_enough

        # Require being above opening AND near center for a couple steps before "armed"
        in_band_lo: float = 0.14
        in_band_hi: float = 0.24
        in_release_band = (h > in_band_lo) & (h < in_band_hi)
        arm_ready = aligned_now & in_release_band
        armed_cnt = torch.clamp(armed_cnt + arm_ready.float(), 0.0, 20.0)
        armed_enough = armed_cnt >= 3.0

        # Release event and proper release detection
        release_event = ((prev_grasp > 0.5) & (grasped < 0.5)).float()
        proper_release = (release_event > 0.5) & armed_enough & (d_xy < max(0.05, 0.95 * rad)) & in_release_band

        had_release = torch.maximum(had_release, proper_release.float())

        # Time since proper release (for limited post-shaping)
        tsr = torch.where(proper_release, torch.zeros_like(tsr), tsr + 1.0)
        tsr = torch.where(had_release > 0.5, tsr, torch.zeros_like(tsr))
        tsr = torch.clamp(tsr, 0.0, 80.0)

        # Success latch to give a terminal-ish bonus without relying on env termination
        success_latch = torch.maximum(success_latch, success)

        # Monotonic stage
        stage = torch.maximum(stage, stable_enough.float() * 1.0)
        stage = torch.maximum(stage, lifted_now.float() * 2.0)
        stage = torch.maximum(stage, aligned_now.float() * 3.0)
        stage = torch.maximum(stage, had_release * 4.0)
        stage = torch.maximum(stage, success_latch * 5.0)

        # Reset arming if grasp lost before proper release; reset after success
        armed_cnt = torch.where((grasped < 0.5) & (had_release < 0.5), torch.zeros_like(armed_cnt), armed_cnt)
        armed_cnt = torch.where(success_latch > 0.5, torch.zeros_like(armed_cnt), armed_cnt)

        # Write back memory
        hv[:, 0] = stage.detach()
        hv[:, 1] = stable_cnt.detach()
        hv[:, 2] = d_xy.detach()
        hv[:, 3] = h.detach()
        hv[:, 4] = grasped.detach()
        hv[:, 5] = armed_cnt.detach()
        hv[:, 6] = tsr.detach()
        hv[:, 7] = had_release.detach()
        hv[:, 8] = success_latch.detach()
        self.helper_variable = hv

        g_stable = (stage >= 1.0).float()
        g_lifted = (stage >= 2.0).float()
        g_aligned = (stage >= 3.0).float()
        g_released = (stage >= 4.0).float()

        # ------------------------------------------------------------
        # Reward terms (changes from best-so-far)
        # 1) Reduce "release_event" dominance by: smaller weight + shaped "armed staying" reward.
        # 2) Increase conversion to actual success by: stronger centering while aligned, and post-release inside_xy.
        # 3) Replace/soften progress deltas with bounded potential differences (less noisy).
        # 4) Add explicit action penalty.
        # ------------------------------------------------------------

        # HOLD: stable grasp + low slip
        temp_slip: float = 0.30
        r_low_slip = torch.exp(-torch.clamp(rel_vel, 0.0, 8.0) / temp_slip)
        temp_hand: float = 0.14
        r_hand_close = torch.exp(-torch.clamp(hand_obj_dist, 0.0, 0.9) / temp_hand)
        r_hold = (0.65 * r_low_slip + 0.35 * r_hand_close) * (grasped > 0.5).float()

        # LIFT: be above lift_h
        temp_lift: float = 0.07
        lift_err = torch.clamp(torch.relu(lift_h - h), 0.0, 2.0)
        r_lift = torch.exp(-lift_err / temp_lift) * (grasped > 0.5).float() * g_stable

        # MOVE XY while lifted
        temp_xy: float = 0.10
        r_xy = torch.exp(-torch.clamp(d_xy, 0.0, 3.0) / temp_xy) * (grasped > 0.5).float() * g_lifted

        # Potential-based progress (less spiky than raw delta; also works near goal)
        temp_phi: float = 0.16
        phi = torch.exp(-torch.clamp(d_xy, 0.0, 3.0) / temp_phi)
        phi_prev = torch.exp(-torch.clamp(prev_dxy, 0.0, 3.0) / temp_phi)
        r_phi_prog = torch.clamp(phi - phi_prev, -0.10, 0.10) * (grasped > 0.5).float() * g_lifted

        # PRE-PLACE: tight center + height band, only when grasped
        temp_band: float = 0.06
        band_err = torch.clamp(torch.relu(in_band_lo - h) + torch.relu(h - in_band_hi), 0.0, 3.0)
        r_band = torch.exp(-band_err / temp_band)

        # Stronger centering when approaching place stage (helps actual inside_site)
        temp_center: float = max(0.020, 0.45 * rad)
        r_center = torch.exp(-torch.clamp(d_xy, 0.0, 3.0) / temp_center)
        r_pre = r_center * r_band * (grasped > 0.5).float() * g_lifted

        # "Stay armed" reward: encourages hovering aligned+inband before release (replaces over-reliance on release bonus)
        r_armed_stay = arm_ready.float() * (grasped > 0.5).float() * g_lifted

        # RELEASE: reward only for proper release; penalize premature releases
        temp_rel_xy: float = 0.06
        r_rel_xy = torch.exp(-torch.clamp(d_xy, 0.0, 3.0) / temp_rel_xy)
        r_release_good = release_event * proper_release.float() * r_rel_xy * r_band
        r_release_bad = -release_event * (1.0 - proper_release.float())  # includes unarmed + wrong position

        # POST-RELEASE window
        post_window = torch.exp(-torch.clamp(tsr, 0.0, 80.0) / 14.0)  # in (0,1]
        post_mask = (1.0 - grasped) * g_released * post_window

        # Keep object inside in XY during fall (important for conversion)
        temp_inside: float = max(0.020, 0.55 * rad)
        r_inside_xy = torch.exp(-torch.clamp(d_xy, 0.0, 3.0) / temp_inside) * post_mask

        # Encourage it to go down after release (bounded)
        dh = torch.clamp(prev_h - h, -0.12, 0.12)
        r_down = torch.relu(dh) * post_mask

        # Low + settle only if inside_site (avoid rewarding dropping outside)
        temp_low: float = 0.055
        z_err = torch.clamp(torch.relu(obj_z - 0.1), 0.0, 2.0)
        r_low = torch.exp(-z_err / temp_low)
        temp_settle: float = 1.05
        settle_speed = torch.linalg.norm(obj_vel - site_vel, dim=-1)
        r_settle = torch.exp(-torch.clamp(settle_speed, 0.0, 12.0) / temp_settle)

        r_drop = post_mask * inside_site.float() * r_low * r_settle

        # Success bonus (latching gives at least one-step big reward)
        r_success = success_latch

        # ------------------------------------------------------------
        # Regularization / safety (smooth motion, avoid singularities)
        # ------------------------------------------------------------
        qd_l2 = torch.sum(torch.clamp(robot_qd, -60.0, 60.0) ** 2, dim=-1)

        q_min = self.robot_dof_lower_limits
        q_max = self.robot_dof_upper_limits
        q_center = 0.5 * (q_min + q_max)
        q_half = torch.clamp(0.5 * (q_max - q_min), min=eps)
        q_norm = _safe(torch.abs((robot_q - q_center) / q_half))
        jl_margin: float = 0.88
        jl_pen = torch.sum(torch.relu(q_norm - jl_margin) ** 2, dim=-1)

        # manipulability reward (bounded [0,1))
        temp_manip: float = 0.55
        r_manip = 1.0 - torch.exp(-torch.clamp(manipulability, 0.0, 12.0) / temp_manip)

        # mild orientation prior: keep near "pointing down" (qx,qw near 0)
        qx = hand_quat[:, 0]
        qw = hand_quat[:, 3]
        temp_ori: float = 0.40
        ori_err = torch.clamp(torch.abs(qx) + torch.abs(qw), 0.0, 2.0)
        r_ori = torch.exp(-ori_err / temp_ori)

        # keep speeds controlled
        obj_speed = torch.linalg.norm(obj_vel, dim=-1)
        tcp_speed = torch.linalg.norm(tcp_vel, dim=-1)
        temp_speed: float = 1.35
        r_speed = torch.exp(-torch.clamp(obj_speed + tcp_speed, 0.0, 25.0) / temp_speed)

        # ------------------------------------------------------------
        # Weights (rebalance: reduce release-event farming; increase centering & post-release inside_xy)
        # ------------------------------------------------------------
        w_hold = 0.18
        w_lift = 0.30
        w_xy = 0.18
        w_phi_prog = 7.0
        w_pre = 0.95
        w_armed_stay = 0.45

        w_release_good = 5.0
        w_release_bad = 4.0

        w_inside_xy = 4.5
        w_down = 7.0
        w_drop = 5.0

        w_success = 95.0

        w_manip = 0.06
        w_ori = 0.02
        w_speed = 0.10

        w_qd = 0.0017
        w_jl = 0.06
        w_act = 0.02

        rew_hold = w_hold * r_hold
        rew_lift = w_lift * r_lift
        rew_xy = w_xy * r_xy
        rew_phi_prog = w_phi_prog * r_phi_prog
        rew_pre = w_pre * r_pre
        rew_armed_stay = w_armed_stay * r_armed_stay

        rew_release = w_release_good * r_release_good + w_release_bad * r_release_bad

        rew_inside_xy = w_inside_xy * r_inside_xy
        rew_down = w_down * r_down
        rew_drop = w_drop * r_drop

        rew_success = w_success * r_success

        rew_manip = w_manip * r_manip
        rew_ori = w_ori * r_ori
        rew_speed = w_speed * r_speed

        rew_qd = -w_qd * qd_l2
        rew_jl = -w_jl * jl_pen
        rew_act = -w_act * act_pen

        reward = (
            rew_hold
            + rew_lift
            + rew_xy
            + rew_phi_prog
            + rew_pre
            + rew_armed_stay
            + rew_release
            + rew_inside_xy
            + rew_down
            + rew_drop
            + rew_success
            + rew_manip
            + rew_ori
            + rew_speed
            + rew_qd
            + rew_jl
            + rew_act
        )

        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        assert torch.isfinite(reward).all(), "Non-finite reward detected."

        individual_rewards = {
            "hold": rew_hold,
            "lift": rew_lift,
            "move_xy": rew_xy,
            "phi_xy_progress": rew_phi_prog,
            "pre_place": rew_pre,
            "armed_stay": rew_armed_stay,
            "release_event": rew_release,
            "inside_xy_post": rew_inside_xy,
            "down_progress": rew_down,
            "drop_in": rew_drop,
            "success": rew_success,
            "manipulability": rew_manip,
            "hand_ori": rew_ori,
            "controlled_speed": rew_speed,
            "joint_vel_reg": rew_qd,
            "joint_limit_reg": rew_jl,
            "action_reg": rew_act,
        }
        return reward, individual_rewards