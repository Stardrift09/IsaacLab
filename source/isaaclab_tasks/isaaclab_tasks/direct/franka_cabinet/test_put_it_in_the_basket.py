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

@configclass
class TestPutItInTheBasketCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2
    action_space = 9
    observation_space = 25 # to modify
    state_space = 0
    # train_with_isaaclab =True

    # if train_with_isaaclab:
    from isaaclab_eureka.utils import eureka_root_dir
    root = eureka_root_dir()
    path = f"{root}/libero/trajs/libero90/libero_90_living_room_scene1_pick_up_the_alphabet_soup_and_put_it_in_the_basket_traj_v2.pkl"
    # else:
    #     path = "libero/trajs/libero90/libero_90_living_room_scene1_pick_up_the_alphabet_soup_and_put_it_in_the_basket_traj_v2.pkl"
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
        num_envs=2048, env_spacing=3.0, replicate_physics=True, clone_in_fabric=True, 
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
        self.debug = False  
        self.path = cfg.path # scene init is called in super
        self.root = cfg.root
        from isaaclab_eureka.utils import read_pkl
        self.data = read_pkl(self.path) 
        # [num_envs, step, states]
        # first loop over, add padding, then stack.
        self.episodes = self.data["franka"]
        self.target_object_name = "alphabet_soup"
        self.target_site_name = "basket"
        self.input_direction = 2 # z axis
        super().__init__(cfg, render_mode, **kwargs)
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

        # import pdb
        # pdb.set_trace()

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
        # import pdb
        # pdb.set_trace()


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
        self.input_direction = torch.tensor([0, 0, 1], device=self.device) # z axis
        debug = False
        if debug:
            import pdb
            pdb.set_trace()

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
        robot_local_pose_pos += torch.tensor([0, 0, 0.05], device=self.device) # This one is tuned from libero demo

        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1)) # 3
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1)) # 4

        self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("panda_rightfinger")[0][0]

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.hand_quat = torch.zeros((self.num_envs, 4), device=self.device) 
        self.helper_variable = torch.zeros((self.num_envs, 10), device=self.device)
        

    def _setup_scene(self):
        # init_states = self.data['franka'][0]["init_state"] # this is from the first scene as set up. Init states of traj should be updated in reset_idx
        # init_states = self.data['franka'][0]["states"][100]

        self.start_idx_in_episode = 80
        init_states = self.data['franka'][0]["states"][80] # 92 in the air. Used 100 For two round training, now use 80, starting from ground
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
            cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/{k}",
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=init_states[k]["pos"],
                    rot=init_states[k]["rot"],
                ),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=f"{self.root}/libero/COMMON/stable_hope_objects/{k}/usd/{k}.usd",
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
        low_enough = self.target_object.data.root_pos_w[:, 2] <0.1
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
    
    # def _get_rewards(self) -> torch.Tensor:

    #     return self._compute_rewards(self.actions,self.cfg.action_penalty_scale,)


    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
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
        # 
        self.site_to_target_pos = self.target_site.data.root_pos_w - self.target_object.data.root_pos_w
        
        # self.target_site_corners_world # you can read the three dim from the [2, 3] tensor storing min(first row) and max corner of the site in local env frame
        # you can check the direction where you enter with self.input_direction
        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
                self.corners_target_obj_to_hand_pos, # this should be small
                self.target_to_hand_pos, # relative position from target object center to hand should be small
                self.hand_quat, # be close to inital orientation (pointing downwards is good, allows z axis rotation(Hint:When 0 and 3 element is zero, hand is pointing downwards)
                self.site_to_target_pos, # relative position in x y from target site to target object
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



    def run_replay(self, log_dir:str, render:bool=False):
        #TODO: make sure number of envs are enough for replay
        print(f"REPLAY LOG DIR {log_dir}")
        num_episodes = len(self.episodes)
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
        lengths_tensor = torch.tensor(lengths,device=self.device)
        # if self.debug:
            # print(lengths)
        max_len = max(lengths)
        # if self.debug:
        #     print(f"max_len {max_len}")
        #     print(f"num_envs: {self.num_envs}")
        # 3. Determine state dimensions
        # robot
        # robot_pos_dim = len(self.episodes[0]["states"][0]["franka"]["pos"])      # 3

        # actually I am not using this..
        # robot_rot_dim = len(self.episodes[0]["states"][0]["franka"]["rot"])      # 4
        robot_dof_dim = len(self.episodes[0]["states"][0]["franka"]["dof_pos"])  # num_joints

        # rigid objects (all share same structure)
        one_obj = next(iter({k: v for k, v in self.sample.items() if k != "franka"}.values()))
        object_pos_dim = len(one_obj["pos"])              # 3
        object_rot_dim = len(one_obj["rot"])              # 4

        # 4. Preallocate tensors
        # robot_pos = torch.zeros((num_episodes, max_len, robot_pos_dim))
        # robot_rot = torch.zeros((num_episodes, max_len, robot_rot_dim))
        robot_dof = torch.zeros((self.num_envs, max_len, robot_dof_dim),device=self.device,dtype=torch.float32)

        object_pos = torch.zeros((self.num_envs, max_len, num_objects, object_pos_dim),device=self.device,dtype=torch.float32)
        object_rot = torch.zeros((self.num_envs, max_len, num_objects, object_rot_dim),device=self.device,dtype=torch.float32)

        padding_mask = torch.ones((self.num_envs, max_len),device=self.device, dtype=torch.bool)
        
        # 5. Fill tensors
        with torch.inference_mode():
            for env_idx, ep in enumerate(self.episodes): # each episode
                ep_len = lengths[env_idx]

                for t, state in enumerate(ep["states"]):
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

            for t in range(max_len):
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

                detect_grasp = True
                if detect_grasp:
                    # update self.robot_grasp_pos
                    _,_ = self._get_dones()
                    diff = self.target_object.data.body_pos_w.squeeze(1) - self.robot_grasp_pos
                    # print(diff[0]) # play the first episode to check the constant object-hand distance
                    diff_norm = torch.norm(diff, dim=-1)  # shape: (n,)
                    if diff_norm[0] < 0.01 and self.target_object.data.root_pos_w[0,2] > 0.1: # check if the grasp is stable (object-hand distance is small) and the object is lifted up (z is large), which should happen at the end of episode when the agent learns to lift up the object while keeping a stable grasp
                        print(f"timestep {t}: object in the air")
                    print(diff_norm[0]) # should be small and constant if the grasp is stable, which is the case for most of the episode, except at the end when the object is lifted up and the grasp is broken. This matches with the observation that the agent learns to keep a stable grasp and lift up the object at the end of training.
                    print(self.target_object.data.root_pos_w[0,2]) # also check the object position, should be lifted up at the end of episode
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


                # After all replay is done, devide by episode length, and record
            for k in eureka_episode_sums.keys():
                # max_episode_length_s
                libero_dt = 1/20 # Not using this, not comparable
                max_episode_length_libero = 200
                max_episode_length = 500
                per_ep_value = eureka_episode_sums[k] * max_episode_length / max_episode_length_libero /self.max_episode_length_s # Should also consider dt and maybe interpolate velocity/simulate
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





    def _get_rewards(self):
        import torch

        device = self.device
        eps: float = 1e-6
        N = self.num_envs

        # -----------------------------
        # Sanitize key state (GPU-safe)
        # -----------------------------
        obj_pos = torch.nan_to_num(self.target_object.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)  # (N,3)
        site_pos = torch.nan_to_num(self.target_site.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)  # (N,3)
        hand_pos = torch.nan_to_num(self.robot_grasp_pos, nan=0.0, posinf=0.0, neginf=0.0)  # (N,3)
        hand_quat = torch.nan_to_num(self.hand_quat, nan=0.0, posinf=0.0, neginf=0.0)  # (N,4)

        obj_xy = obj_pos[:, :2]
        site_xy = site_pos[:, :2]
        obj_z = obj_pos[:, 2]
        site_z = site_pos[:, 2]

        # Distances
        d_xy = torch.linalg.norm(obj_xy - site_xy, dim=-1)
        d_hand = torch.linalg.norm(obj_pos - hand_pos, dim=-1)

        # Given success metric
        r_true: float = float(self.target_site_radius)
        low_enough = obj_z < 0.1
        inside_site = d_xy < r_true
        success = (inside_site & low_enough).float()

        # ---------------------------------------------------------
        # Core issue from feedback:
        # - obj_z goes to ~10m: exploit; penalties not preventing.
        # - d_xy remains huge: agent not truly moving above basket.
        # - shaped rewards still high: stage gates saturate and/or
        #   agent gets reward without satisfying the real conditions.
        #
        # Fixes:
        # 1) Make an explicit *height ceiling* penalty based on absolute z
        #    (not relative to site only). This clamps the "fly to sky" exploit.
        # 2) Use a strict stage machine with hard-ish gates (ramps), and
        #    ensure each stage reward is only active when its gate is active.
        # 3) Greatly increase the cost of being far in XY while "high" (carry),
        #    and cost of being "low" while not centered (bad drop).
        # 4) Keep all per-step components bounded ~[0,1], and total reward ~[-10,10].
        # ---------------------------------------------------------

        # -----------------------------
        # Targets / tolerances
        # -----------------------------
        carry_clearance: float = 0.10
        z_carry = site_z + carry_clearance

        # Soft radii for gradients
        r_soft: float = r_true + 0.10
        r_carry_need: float = r_true + 0.25  # looser for "approach above basket"

        # -----------------------------
        # Helper: ramp functions (0..1)
        # -----------------------------
        def ramp(x: torch.Tensor, x0: float, x1: float) -> torch.Tensor:
            # 0 when x<=x0, 1 when x>=x1
            denom = max(x1 - x0, eps)
            return torch.clamp((x - x0) / denom, 0.0, 1.0)

        # Height gates
        g_high = ramp(obj_z, float((z_carry - 0.05).mean().item()) if z_carry.ndim == 0 else 0.0, 1.0)  # placeholder safe
        # Compute g_high properly without item() to stay vectorized:
        g_high = torch.clamp((obj_z - (z_carry - 0.05)) / 0.10, 0.0, 1.0)  # 0 below, 1 above carry band

        # Center gate (0..1) using bounded exp
        temp_center: float = 6.0
        center_arg = torch.clamp((d_xy / max(r_soft, eps)) ** 2, 0.0, 25.0)
        g_center = torch.exp(-temp_center * center_arg)

        # "Ready to drop": must be high AND centered
        g_ready_drop = g_high * g_center

        # Low gate for completion of drop
        g_low = torch.clamp((0.12 - obj_z) / 0.08, 0.0, 1.0)  # 1 when <=0.04, 0 when >=0.12

        # -----------------------------
        # Stage weights (harder gating)
        # -----------------------------
        # Lift until high; carry when high but not centered; drop when ready (centered&high)
        w_lift = 1.0 - g_high
        w_drop = g_ready_drop
        w_carry = torch.clamp(g_high - w_drop, 0.0, 1.0)

        w_sum = torch.clamp(w_lift + w_carry + w_drop, min=eps)
        w_lift = w_lift / w_sum
        w_carry = w_carry / w_sum
        w_drop = w_drop / w_sum

        # -----------------------------
        # Bounded rewards (0..1)
        # -----------------------------
        # Lift reward: reward reaching carry height, but DO NOT reward going higher.
        # Use a "below-only" error with exp shaping.
        temp_lift: float = 8.0
        z_below = torch.clamp(z_carry - obj_z, min=0.0)
        lift_arg = torch.clamp((z_below / 0.20) ** 2, 0.0, 25.0)
        r_lift = torch.exp(-temp_lift * lift_arg)  # 1 when at/above z_carry

        # Height band reward for carry: stay near z_carry (prevents huge z)
        temp_zband: float = 6.0
        z_err = torch.clamp(((obj_z - z_carry) / 0.12) ** 2, 0.0, 25.0)
        r_z_band = torch.exp(-temp_zband * z_err)

        # XY reward: be near center (dominant in carry/drop)
        temp_xy: float = 6.0
        xy_arg = torch.clamp((d_xy / max(r_carry_need, eps)) ** 2, 0.0, 25.0)
        r_xy = torch.exp(-temp_xy * xy_arg)

        # Drop reward: only meaningful when centered; encourage going low
        # Use a smooth term that is ~1 when obj_z <= 0.1
        temp_drop: float = 10.0
        drop_arg = torch.clamp((torch.clamp(obj_z - 0.10, min=0.0) / 0.08) ** 2, 0.0, 25.0)
        r_drop = torch.exp(-temp_drop * drop_arg)

        # Hold reward (very weak): keep hand near object to reduce "throwing"
        temp_hold: float = 4.0
        hold_arg = torch.clamp((d_hand / 0.30) ** 2, 0.0, 25.0)
        r_hold = torch.exp(-temp_hold * hold_arg)

        # Orientation reward (weak): keep qx and qw near 0
        qx = hand_quat[:, 0]
        qw = hand_quat[:, 3]
        ori_err2 = torch.clamp(qx * qx + qw * qw, 0.0, 25.0)
        temp_ori: float = 2.5
        r_ori = torch.exp(-temp_ori * ori_err2)

        # -----------------------------
        # Progress shaping (tiny, stable)
        # -----------------------------
        if not (hasattr(self, "helper_variable") and (self.helper_variable is not None) and (self.helper_variable.shape[0] == N)):
            self.helper_variable = torch.zeros((N, 3), device=device)
        elif self.helper_variable.shape[1] < 3:
            hv = torch.zeros((N, 3), device=device)
            hv[:, : self.helper_variable.shape[1]] = self.helper_variable
            self.helper_variable = hv

        prev_dxy = torch.nan_to_num(self.helper_variable[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
        prev_z = torch.nan_to_num(self.helper_variable[:, 1], nan=0.0, posinf=0.0, neginf=0.0)
        prev_ready = torch.nan_to_num(self.helper_variable[:, 2], nan=0.0, posinf=0.0, neginf=0.0)

        self.helper_variable[:, 0] = d_xy
        self.helper_variable[:, 1] = obj_z
        self.helper_variable[:, 2] = g_ready_drop

        # Encourage reducing d_xy only when high (carry)
        dxy_prog = torch.clamp(prev_dxy - d_xy, -0.05, 0.05) / 0.05  # [-1,1]
        r_dxy_prog = torch.clamp(dxy_prog, -1.0, 1.0) * g_high

        # Encourage lifting up only when below carry
        dz = torch.clamp(obj_z - prev_z, -0.05, 0.05) / 0.05
        r_dz_prog = torch.clamp(dz, -1.0, 1.0) * torch.clamp(z_below / 0.20, 0.0, 1.0)

        # Encourage entering ready-to-drop
        r_ready_prog = torch.clamp(g_ready_drop - prev_ready, -0.1, 0.1)

        # -----------------------------
        # Strong anti-exploit penalties
        # -----------------------------
        # Absolute height ceiling penalty (key fix):
        # In these tasks, object z should stay near table/basket height (<~1m). Penalize beyond.
        z_abs_ceiling: float = 1.2
        z_abs_over = torch.clamp(obj_z - z_abs_ceiling, min=0.0)
        p_abs_too_high = torch.clamp((z_abs_over / 0.6) ** 2, 0.0, 25.0)

        # Relative too-high penalty (extra safety)
        z_rel_over = torch.clamp(obj_z - (z_carry + 0.25), min=0.0)
        p_rel_too_high = torch.clamp((z_rel_over / 0.4) ** 2, 0.0, 25.0)

        # Penalize being far from basket especially while high (prevents "high but far" loophole)
        p_far_xy = torch.clamp((d_xy / max(r_soft, eps)) ** 2, 0.0, 25.0)
        p_far_xy_high = p_far_xy * g_high

        # Penalize going low while not centered (bad drop)
        far01 = torch.clamp(d_xy / max(r_soft, eps), 0.0, 1.0)
        p_low_far = g_low * far01

        # Penalize slip (object far from hand) more when high (likely thrown)
        p_slip = torch.clamp(d_hand / 0.35, 0.0, 3.0) * g_high * (1.0 - success)

        # -----------------------------
        # Stage rewards (kept bounded)
        # -----------------------------
        rew_lift = 1.2 * r_lift + 0.1 * r_hold + 0.05 * r_ori
        rew_carry = 1.6 * r_xy + 0.7 * r_z_band + 0.05 * r_hold + 0.05 * r_ori
        rew_drop = 1.2 * r_xy + 1.4 * r_drop + 0.05 * r_ori

        shaped = w_lift * rew_lift + w_carry * rew_carry + w_drop * rew_drop

        # Success bonus: big and simple
        temp_success: float = 3.0
        r_success_bonus = torch.expm1(temp_success * success)  # 0 or expm1(3)=~19.09

        # Small living penalty
        time_penalty = torch.full((N,), 0.005, device=device)

        reward = (
            shaped
            + 1.0 * r_success_bonus
            + 0.03 * r_dxy_prog
            + 0.02 * r_dz_prog
            + 0.04 * r_ready_prog
            - 0.80 * p_far_xy_high
            - 0.20 * p_far_xy
            - 2.50 * p_abs_too_high
            - 0.80 * p_rel_too_high
            - 1.20 * p_low_far
            - 0.30 * p_slip
            - time_penalty
        )

        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        reward = torch.clamp(reward, -10.0, 10.0)
        assert torch.isfinite(reward).all()

        individual = {
            # state
            "d_xy": d_xy,
            "d_hand": d_hand,
            "obj_z": obj_z,
            "inside_site": inside_site.float(),
            "low_enough": low_enough.float(),
            "success": success,
            # gates / stage weights
            "g_high": g_high,
            "g_center": g_center,
            "g_ready_drop": g_ready_drop,
            "g_low": g_low,
            "w_lift": w_lift,
            "w_carry": w_carry,
            "w_drop": w_drop,
            # rewards
            "r_lift": r_lift,
            "r_xy": r_xy,
            "r_z_band": r_z_band,
            "r_drop": r_drop,
            "r_hold": r_hold,
            "r_ori": r_ori,
            "shaped": shaped,
            "success_bonus": r_success_bonus,
            # progress
            "r_dxy_prog": r_dxy_prog,
            "r_dz_prog": r_dz_prog,
            "r_ready_prog": r_ready_prog,
            # penalties
            "p_abs_too_high": p_abs_too_high,
            "p_rel_too_high": p_rel_too_high,
            "p_far_xy": p_far_xy,
            "p_far_xy_high": p_far_xy_high,
            "p_low_far": p_low_far,
            "p_slip": p_slip,
            "time_penalty": time_penalty,
            "reward": reward,
        }
        return reward


    def get_target_object_to_hand_pose(self, init_states: dict):
        # Firstly find out the relative transformation, and then adjust the object pose according to robot pose.
        # Assume that the scene has been initialized
        pass



    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        self._compute_robot_intermediate_values(env_ids)
        self._compute_target_object_corners(env_ids)
        


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