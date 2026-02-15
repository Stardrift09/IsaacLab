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
class LivingRoomScene1PickUpTheAlphabetSoupAndPutItInTheBasketCfg(DirectRLEnvCfg):
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
        num_envs=1920, env_spacing=3.0, replicate_physics=True, clone_in_fabric=True, 
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


class LivingRoomScene1PickUpTheAlphabetSoupAndPutItInTheBasket(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: LivingRoomScene1PickUpTheAlphabetSoupAndPutItInTheBasketCfg

    def __init__(self, cfg: LivingRoomScene1PickUpTheAlphabetSoupAndPutItInTheBasketCfg, render_mode: str | None = None, **kwargs):
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

        if self.debug:
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




        prim = stage.GetPrimAtPath(f"/World/envs/env_0/{self.target_site_name}")
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeLocalBound(prim)
        min_corner = bbox.GetRange().GetMin()  # Vec3
        max_corner = bbox.GetRange().GetMax()  # Vec3
        target_site_size = torch.tensor((max_corner - min_corner), device=self.device)
        obj_xy_extent = self.target_object_size[:2].min()
        site_xy_extent = target_site_size[:2].max()
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
        robot_local_pose_pos += torch.tensor([0, 0.0, 0.04], device=self.device)

        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1)) # 3
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1)) # 4

        self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("panda_rightfinger")[0][0]

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.helper_variable = torch.zeros((self.num_envs, 10), device=self.device)
        self.quat_desired = torch.tensor([0, 0, 0, 1], device=self.device).repeat(self.num_envs, 1)


    def _setup_scene(self):
        init_states = self.data['franka'][0]["init_state"] # this is from the first scene as set up. Init states of traj should be updated in reset_idx
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
        hand_pos = self._robot.data.body_pos_w[:, self.hand_link_idx]
        hand_rot = self._robot.data.body_quat_w[:, self.hand_link_idx]

        self.robot_grasp_rot, self.robot_grasp_pos = tf_combine(
            hand_rot, hand_pos, self.robot_local_grasp_rot, self.robot_local_grasp_pos
        )
        # update obj corners
        pos = self.target_object.data.root_pos_w
        quat= quat_conjugate(self.target_object.data.root_quat_w)

        self.corners_target_obj = transform_points(points=self.local_corners_init,
                                              pos=pos,
                                              quat=quat)
        self.local_centers = transform_points(points=self.local_centers_init,
                                              pos=pos,
                                              quat=quat)

        # condition for termination
        low_enough = self.target_object.data.root_pos_w[:, 2] <0.1
        obj_xy = self.target_object.data.root_pos_w[:, :2]
        site_pos = self.target_site.data.root_pos_w[:, :2]
        dist2 = ((obj_xy - site_pos)**2).sum(dim=-1)
        inside_site = dist2 < self.target_site_radius**2
        # terminated = inside_site & low_enough
        terminated = self.target_object.data.root_pos_w[:, 2] > 0.4
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated
    
    def _get_rewards(self) -> torch.Tensor:

        return self._compute_rewards(self.actions,self.cfg.action_penalty_scale,)


    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        # robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )
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

        hand_pos = self._robot.data.body_pos_w[env_ids, self.hand_link_idx]
        hand_rot = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]

        self.robot_grasp_rot[env_ids], self.robot_grasp_pos[env_ids] = tf_combine(
            hand_rot, hand_pos, self.robot_local_grasp_rot[env_ids], self.robot_local_grasp_pos[env_ids]
        )

        # update obj corners
        pos = self.target_object.data.root_pos_w[env_ids]
        quat= quat_conjugate(self.target_object.data.root_quat_w[env_ids])
        self.corners_target_obj[env_ids] = transform_points(points=self.local_corners_init[env_ids],
                                              pos=pos,
                                              quat=quat)
        self.local_centers[env_ids] = transform_points(points=self.local_centers_init[env_ids],
                                              pos=pos,
                                              quat=quat)
        

        # clear helper variable
        self.helper_variable[env_ids] = torch.zeros((len(env_ids), 10), device=self.device)


    def _get_observations(self) -> dict:
        # self.helper_variable = torch.zeros((self.num_envs, 10), device=self.device)
        # self.rigid_objects is a list with all RigidObject, you can use it to calculate AABBs.
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        self.target_to_hand_pos = self.target_object.data.body_pos_w.squeeze(1) - self.robot_grasp_pos # n, 3
        hand_quat = self._robot.data.body_quat_w[:, self.hand_link_idx]
        # Forcing the 0th and 3th element of hand_quat to be 0 to point downwards. You can also penalize the rotation of the target object

        self.corners_target_obj_to_hand_pos =  (self.corners_target_obj - self.robot_grasp_pos.unsqueeze(1)).reshape(self.num_envs, -1) # corners to robot dist described in world coordinate.
        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
                self.corners_target_obj_to_hand_pos, # this should be small
                self.target_to_hand_pos, # relative position from target object center to hand should be small
                hand_quat, # be close to inital orientation (pointing downwards is good, allows z axis rotation
            ),
            dim=-1,
        )


        print(self.target_object.data.root_pos_w[:,2])

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

        # rewards = torch.where(cream_cheese_pos[:,2] > 0.1, rewards + 1, rewards)
        return rewards



    def run_replay(self, log_dir:str, render:bool=False):
        #TODO: make sure number of envs are enough for replay
        print(f"REPLAY LOG DIR {log_dir}")
        num_episodes = len(self.episodes)
        env_ids = torch.arange(num_episodes, device=self.device, dtype=torch.long)
        self._reset_idx(env_ids)
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





    def _get_rewards_test(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        import torch

        device = self.device
        eps: float = 1e-6

        # ----------------------------
        # Read & sanitize state
        # ----------------------------
        obj_pos = torch.nan_to_num(self.target_object.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)  # (N,3)
        ee_pos = torch.nan_to_num(self.robot_grasp_pos, nan=0.0, posinf=0.0, neginf=0.0)  # (N,3)

        rel = torch.nan_to_num(obj_pos - ee_pos, nan=0.0, posinf=0.0, neginf=0.0)  # obj - ee
        rel_xy = torch.linalg.norm(rel[:, :2], dim=-1).clamp(min=0.0)
        rel_z = rel[:, 2].clamp(min=-2.0, max=2.0)
        dist = torch.linalg.norm(rel, dim=-1).clamp(min=0.0)

        z_obj = obj_pos[:, 2].clamp(min=-2.0, max=2.0)

        joint_vel = torch.nan_to_num(self._robot.data.joint_vel, nan=0.0, posinf=0.0, neginf=0.0)
        vel_l2 = torch.sqrt(torch.sum(joint_vel * joint_vel, dim=-1) + eps)

        # -------------------------------------------------------------------------
        # Analysis-driven rewrite:
        # - Your logs show many components are constant across training (reach/corners/orientation/above/etc.).
        #   That indicates those terms were either coming from a different reward function in your run
        #   or were effectively saturated/decoupled from action.
        # - Most critically: task_score/success/lift stayed 0. This means the agent never triggers lift.
        # - Also: rel_z is very negative (~ -4.5), and dist/rel_xy can get huge -> the agent often places
        #   the hand far above the object (EE z >> obj z) and still receives large dense rewards due to
        #   saturation / scaling issues in previous shaping.
        #
        # New reward design goals:
        #   (1) Keep ALL dense terms in [0, 1] with gentle temperatures (no >50 magnitudes).
        #   (2) Strongly encourage "hand slightly ABOVE object" first (safe approach).
        #   (3) Then encourage descending to a "grasp band" (EE slightly BELOW object center),
        #       but only after XY is aligned (prevents descending far away).
        #   (4) Encourage lift based on object height ALWAYS (not gated on unknown grasp signal),
        #       but scale it up when the EE is close to object (so lifting while not near gives less).
        #   (5) Add a strong sparse success bonus at z_obj > 0.4.
        # -------------------------------------------------------------------------

        # ----------------------------
        # Shaping: XY centering (dense)
        # ----------------------------
        # Temperatures chosen to avoid saturation too early, still providing gradient from far away.
        t_xy: float = 0.10
        r_xy = torch.exp(-rel_xy / (t_xy + eps)).clamp(0.0, 1.0)

        # 3D distance shaping (so it can't get reward by only matching XY while being meters away in Z)
        t_dist: float = 0.25
        r_dist = torch.exp(-dist / (t_dist + eps)).clamp(0.0, 1.0)

        # ----------------------------
        # Shaping: approach from above (EE above object by ~8cm)
        # rel_z = obj_z - ee_z. For EE above: rel_z negative.
        # Target rel_z ~= -0.08.
        # ----------------------------
        above_target: float = -0.08
        above_err = (rel_z - above_target).abs().clamp(0.0, 2.0)
        t_above: float = 0.10
        r_above = torch.exp(-above_err / (t_above + eps)).clamp(0.0, 1.0)

        # Gate for being reasonably centered before we reward descending/grasp band
        xy_gate_center: float = 0.06
        xy_gate_temp: float = 0.02
        g_center = torch.sigmoid((xy_gate_center - rel_xy) / (xy_gate_temp + eps)).clamp(0.0, 1.0)

        # ----------------------------
        # Shaping: "grasp band" (EE slightly below object center by ~2cm)
        # Target rel_z ~= +0.02.
        # Only reward when centered in XY (prevents diving elsewhere).
        # ----------------------------
        grasp_target: float = 0.02
        grasp_err = (rel_z - grasp_target).abs().clamp(0.0, 2.0)
        t_grasp: float = 0.06
        r_grasp_band = (torch.exp(-grasp_err / (t_grasp + eps)) * g_center).clamp(0.0, 1.0)

        # ----------------------------
        # Lift reward: based on object height, but boosted when EE is near object.
        # This avoids the "never lift because grasp not detected" failure mode.
        # ----------------------------
        z_success: float = 0.40
        z_start: float = 0.10
        lift_progress = ((z_obj - z_start) / (z_success - z_start + eps)).clamp(0.0, 1.0)

        # Proximity boost: if the hand stays near while object rises, likely actually grasping.
        close_dist: float = 0.08
        close_temp: float = 0.02
        g_close = torch.sigmoid((close_dist - dist) / (close_temp + eps)).clamp(0.0, 1.0)

        # Smooth but sharper near success to push crossing 0.4
        r_lift = (lift_progress**3).clamp(0.0, 1.0)

        # Combine: always some lift reward, but much larger when close
        lift_combined = (0.25 * r_lift + 0.75 * (r_lift * g_close)).clamp(0.0, 1.0)

        # ----------------------------
        # Success bonus
        # ----------------------------
        raw_success = (z_obj > z_success).to(torch.float32)
        # Mildly prefer being close at success, but don't over-gate (avoid missing reward on noisy contact)
        success = (raw_success * (0.5 + 0.5 * g_close)).clamp(0.0, 1.0)

        # ----------------------------
        # Penalties / regularization
        # ----------------------------
        vel_pen_scale: float = 0.005
        vel_pen = (-vel_pen_scale * vel_l2).clamp(-1.0, 0.0)

        # Penalize being extremely far (helps exploration not get stuck with huge distances)
        far_d: float = 0.8
        far_pen = (-torch.relu(dist - far_d)).clamp(-2.0, 0.0)

        # ----------------------------
        # Weighted sum (keep totals moderate)
        # ----------------------------
        w_xy: float = 1.5
        w_dist: float = 1.0
        w_above: float = 1.0
        w_grasp: float = 2.0
        w_lift: float = 6.0
        w_success: float = 12.0
        w_far: float = 0.2

        reward = (
            w_xy * r_xy
            + w_dist * r_dist
            + w_above * r_above
            + w_grasp * r_grasp_band
            + w_lift * lift_combined
            + w_success * success
            + vel_pen
            + w_far * far_pen
        )

        # Bound overall reward for stability
        t_total: float = 10.0
        reward = (t_total * torch.tanh(reward / t_total)).clamp(-t_total, t_total)
        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

        # Finite guard
        if not torch.isfinite(reward).all():
            reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

        individual_rewards = {
            "r_xy": r_xy,
            "r_dist": r_dist,
            "r_above": r_above,
            "g_center": g_center,
            "r_grasp_band": r_grasp_band,
            "lift_progress": lift_progress,
            "g_close": g_close,
            "lift": lift_combined,
            "raw_success": raw_success,
            "success": success,
            "vel_pen": vel_pen,
            "far_pen": far_pen,
            "dist": dist,
            "rel_xy": rel_xy,
            "rel_z": rel_z,
            "z_obj": z_obj,
        }
        return reward, individual_rewards