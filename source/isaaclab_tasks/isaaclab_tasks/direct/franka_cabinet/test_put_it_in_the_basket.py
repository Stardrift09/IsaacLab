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


@configclass
class TestPutItInTheBasketCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.65  # 519 timesteps
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
        self.left_finger_body_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_body_idx = self._robot.find_bodies("panda_rightfinger")[0][0]
        self.left_finger_joint_idx = self._robot.find_joints("panda_finger_joint2")[0][0]
        self.right_finger_joint_idx = self._robot.find_joints("panda_finger_joint1")[0][0]



        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.hand_quat = torch.zeros((self.num_envs, 4), device=self.device) 
        self.helper_variable = torch.zeros((self.num_envs, 10), device=self.device)
        self.manipulability = torch.zeros((self.num_envs), device=self.device)

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

        right_contact_sensor_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/alphabet_soup/object", update_period=0.0, history_length=10, 
            track_air_time=True
        )
        self._right_contact_sensors = ContactSensor(right_contact_sensor_cfg)
        self.scene.sensors["right_contact_sensor"] = self._right_contact_sensors

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
        tcp_vel = (self._robot.data.body_link_lin_vel_w[:,self.left_finger_body_idx] + self._robot.data.body_link_lin_vel_w[:,self.right_finger_body_idx])/2
        target_to_hand_vel = self.target_object.data.root_lin_vel_w - tcp_vel
        # add velocity

        self.site_to_target_pos = self.target_site.data.root_pos_w - self.target_object.data.root_pos_w
        site_to_target_vel = self.target_object.data.root_lin_vel_w - self.target_site.data.root_lin_vel_w
        # self.target_site_corners_world # you can read the three dim from the [2, 3] tensor storing min(first row) and max corner of the site in local env frame
        # you can check the direction where you enter with self.input_direction


        self.manipulability = self._compute_manipulability(self._robot._ALL_INDICES) # Always have reward on this to have correct robot motion
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
        
        # 5. Fill tensors
        with torch.inference_mode():
            for env_idx, ep in enumerate(self.episodes): # each episode
                new_ep = ep["states"][self.start_idx_in_episode:] # avoid changing in-place
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

                detect_grasp = False
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
                else:
                    raise NotImplementedError(
                        f"{self.__class__.__name__} must implement `_get_rewards_eureka()` "
                        "when using Eureka reward replay."
                    )

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



    def _get_rewards(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        import torch

        eps = 1e-6
        device = self.device

        # -----------------------------
        # Read / sanitize state
        # -----------------------------
        obj_pos = torch.nan_to_num(self.target_object.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)  # (N,3)
        site_pos = torch.nan_to_num(self.target_site.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)  # (N,3)
        hand_pos = torch.nan_to_num(self.robot_grasp_pos, nan=0.0, posinf=0.0, neginf=0.0)  # (N,3)

        obj_xy = obj_pos[:, :2]
        site_xy = site_pos[:, :2]
        xy_vec = obj_xy - site_xy
        dist2_xy = torch.clamp((xy_vec * xy_vec).sum(dim=-1), min=0.0)
        dist_xy = torch.sqrt(dist2_xy + eps)

        # Basket height model:
        # site z is bottom center; task wants clearance above rim by 10cm. We approximate rim at bottom + 0.10
        rim_z = site_pos[:, 2] + 0.10
        clearance = obj_pos[:, 2] - rim_z  # >0 means safely above rim+10cm

        # Provided success metric
        radius = float(self.target_site_radius)
        inside_site = dist2_xy < (radius * radius)
        low_enough = obj_pos[:, 2] < 0.1
        success = inside_site & low_enough

        # -----------------------------
        # Key issue in previous reward (from logs):
        # - Huge/near-constant exp rewards (r_lift_height, r_descend, r_grasp, r_joint_margin) -> reward hacking.
        # - XY shaping effectively off (r_xy_close ~0) because dist_xy huge and temp too small; also inside_site ~0.
        # - Orientation term collapsed to 0 because "init quat = zeros" invalid -> always bad.
        # New design:
        #   1) Use bounded, scale-stable shaping in [0,1] (mostly exp(-err/temp)).
        #   2) Make XY approach the primary dense objective early (bigger temp).
        #   3) Use soft stage weights but avoid multiplying by hard indicators that freeze gradients.
        #   4) Orientation: encourage "no rotation" by staying close to a *fixed reference* computed from current
        #      at reset is not available; use a robust "keep pointing down" proxy from hint: penalize (qx,qw).
        #   5) Add a strong "inside & low" sparse bonus, but keep dense terms informative.
        # -----------------------------

        # -----------------------------
        # Soft stage weights (no hard gates)
        # -----------------------------
        # w_clear ~ 0 when below clearance, -> 1 when above
        temp_stage_clear = 0.02
        w_clear = torch.sigmoid(torch.clamp(clearance / temp_stage_clear, -20.0, 20.0))

        # w_align_xy ~ 0 far, -> 1 near basket opening
        # Choose a scale around basket radius / few cm; use 4cm for smoothness.
        temp_stage_xy = 0.04
        w_align_xy = torch.exp(-dist_xy / temp_stage_xy)
        w_align_xy = torch.clamp(w_align_xy, 0.0, 1.0)

        # "ready to descend" weight: need both clearance and near-xy
        w_ready = w_clear * w_align_xy

        # -----------------------------
        # Dense objectives
        # -----------------------------
        # 1) XY approach (dominant): larger temp so it's not saturated at 0 when far
        temp_xy = 0.20
        r_xy = torch.exp(-dist_xy / temp_xy)
        r_xy = torch.clamp(r_xy, 0.0, 1.0)

        # 2) Clearance (lift) objective: only care until cleared; after that, don't keep rewarding "go higher"
        temp_clear = 0.04
        clear_def = torch.relu(-clearance)  # how much below clearance
        r_clear = torch.exp(-clear_def / temp_clear)
        r_clear = torch.clamp(r_clear, 0.0, 1.0)

        # 3) Z-hold during transfer: only matters when cleared (avoid constant large reward before lift)
        z_target = rim_z  # rim+10cm target height
        z_err = torch.abs(obj_pos[:, 2] - z_target)
        temp_z_hold = 0.06
        r_z_hold = torch.exp(-z_err / temp_z_hold)
        r_z_hold = torch.clamp(r_z_hold, 0.0, 1.0)

        # 4) Descend/place: reward being low *only when aligned and cleared* (prevents dropping elsewhere)
        temp_desc = 0.05
        z_above_goal = torch.relu(obj_pos[:, 2] - 0.1)
        r_low = torch.exp(-z_above_goal / temp_desc)  # 1 when low enough
        r_low = torch.clamp(r_low, 0.0, 1.0)

        # 5) Inside-site shaping: provide dense reward for getting within radius (helps when success is sparse)
        # Use normalized distance to radius; clamp to avoid huge grads
        norm_d = dist_xy / max(radius, 1e-3)
        norm_d = torch.clamp(norm_d, 0.0, 5.0)
        temp_inside = 0.5
        r_inside = torch.exp(-norm_d / temp_inside)
        r_inside = torch.clamp(r_inside, 0.0, 1.0)

        # -----------------------------
        # Constraints / regularizers (small, bounded)
        # -----------------------------
        # Grasp stability: keep object close to hand (non-slipping)
        rel = torch.nan_to_num(self.target_to_hand_pos, nan=0.0, posinf=0.0, neginf=0.0)
        rel_dist = torch.sqrt(torch.clamp((rel * rel).sum(dim=-1), min=0.0) + eps)
        temp_grasp = 0.06
        r_grasp = torch.exp(-rel_dist / temp_grasp)
        r_grasp = torch.clamp(r_grasp, 0.0, 1.0)

        # Orientation fixed: robust proxy from hint (qx and qw near 0 when pointing down)
        hand_quat = torch.nan_to_num(self.hand_quat, nan=0.0, posinf=0.0, neginf=0.0)
        ori_proxy = torch.sqrt(torch.clamp(hand_quat[:, 0] ** 2 + hand_quat[:, 3] ** 2, min=0.0) + eps)
        temp_ori = 0.10
        r_ori = torch.exp(-ori_proxy / temp_ori)
        r_ori = torch.clamp(r_ori, 0.0, 1.0)

        # Smoothness: discourage large joint velocities
        jvel = torch.nan_to_num(self._robot.data.joint_vel, nan=0.0, posinf=0.0, neginf=0.0)
        vmag = torch.sqrt(torch.clamp((jvel * jvel).sum(dim=-1), min=0.0) + eps)
        temp_vel = 6.0
        r_smooth = torch.exp(-vmag / temp_vel)
        r_smooth = torch.clamp(r_smooth, 0.0, 1.0)

        # Joint-limit margin: mild penalty only near limits
        qpos = torch.nan_to_num(self._robot.data.joint_pos, nan=0.0, posinf=0.0, neginf=0.0)
        lower = torch.nan_to_num(self.robot_dof_lower_limits, nan=0.0, posinf=0.0, neginf=0.0)
        upper = torch.nan_to_num(self.robot_dof_upper_limits, nan=0.0, posinf=0.0, neginf=0.0)
        span = torch.clamp(upper - lower, min=eps)
        margin = torch.minimum(qpos - lower, upper - qpos) / span
        margin_min = torch.clamp(margin.min(dim=-1).values, 0.0, 1.0)
        jl_thresh = 0.06
        jl_violation = torch.clamp((jl_thresh - margin_min) / jl_thresh, min=0.0, max=1.0)
        r_joint = 1.0 - jl_violation
        r_joint = torch.clamp(r_joint, 0.0, 1.0)

        # -----------------------------
        # Compose reward (all components ~O(1))
        # -----------------------------
        # Early: prioritize XY approach + clearance.
        # After clearance: prioritize XY + z_hold.
        # When ready (aligned+cleared): prioritize low + inside.
        r_lift_phase = (1.0 - w_clear) * (0.9 * r_clear + 0.4 * r_xy)
        r_transfer_phase = w_clear * (0.9 * r_xy + 0.5 * r_z_hold + 0.3 * r_inside)
        r_place_phase = w_ready * (1.2 * r_low + 1.0 * r_inside + 0.3 * r_xy)

        r_constraints = 0.25 * r_grasp + 0.20 * r_ori + 0.10 * r_smooth + 0.10 * r_joint

        success_bonus = success.float()

        reward = r_lift_phase + r_transfer_phase + r_place_phase + r_constraints + 8.0 * success_bonus

        # -----------------------------
        # Safety
        # -----------------------------
        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        reward = torch.clamp(reward, -2.0, 12.0)
        assert torch.isfinite(reward).all()

        individual_rewards = {
            "r_lift_phase": r_lift_phase,
            "r_transfer_phase": r_transfer_phase,
            "r_place_phase": r_place_phase,
            "r_xy": r_xy,
            "r_inside": r_inside,
            "r_clear": r_clear,
            "r_z_hold": r_z_hold,
            "r_low": r_low,
            "r_grasp": r_grasp,
            "r_ori": r_ori,
            "r_smooth": r_smooth,
            "r_joint": r_joint,
            "w_clear": w_clear,
            "w_align_xy": w_align_xy,
            "w_ready": w_ready,
            "success_bonus": success_bonus,
            "inside_site": inside_site.float(),
            "low_enough": low_enough.float(),
            "dist_xy": dist_xy,
            "clearance": clearance,
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
        pass

    def _get_rewards_test(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        import torch

        eps = 1e-6
        device = self.device

        # -----------------------------
        # Read / sanitize state
        # -----------------------------
        obj_pos = torch.nan_to_num(self.target_object.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)  # (N,3)
        site_pos = torch.nan_to_num(self.target_site.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)  # (N,3)
        hand_pos = torch.nan_to_num(self.robot_grasp_pos, nan=0.0, posinf=0.0, neginf=0.0)  # (N,3)

        obj_xy = obj_pos[:, :2]
        site_xy = site_pos[:, :2]
        xy_vec = obj_xy - site_xy
        dist2_xy = torch.clamp((xy_vec * xy_vec).sum(dim=-1), min=0.0)
        dist_xy = torch.sqrt(dist2_xy + eps)

        # Basket height model:
        # site z is bottom center; task wants clearance above rim by 10cm. We approximate rim at bottom + 0.10
        rim_z = site_pos[:, 2] + 0.10
        clearance = obj_pos[:, 2] - rim_z  # >0 means safely above rim+10cm

        # Provided success metric
        radius = float(self.target_site_radius)
        inside_site = dist2_xy < (radius * radius)
        low_enough = obj_pos[:, 2] < 0.1
        success = inside_site & low_enough

        # -----------------------------
        # Key issue in previous reward (from logs):
        # - Huge/near-constant exp rewards (r_lift_height, r_descend, r_grasp, r_joint_margin) -> reward hacking.
        # - XY shaping effectively off (r_xy_close ~0) because dist_xy huge and temp too small; also inside_site ~0.
        # - Orientation term collapsed to 0 because "init quat = zeros" invalid -> always bad.
        # New design:
        #   1) Use bounded, scale-stable shaping in [0,1] (mostly exp(-err/temp)).
        #   2) Make XY approach the primary dense objective early (bigger temp).
        #   3) Use soft stage weights but avoid multiplying by hard indicators that freeze gradients.
        #   4) Orientation: encourage "no rotation" by staying close to a *fixed reference* computed from current
        #      at reset is not available; use a robust "keep pointing down" proxy from hint: penalize (qx,qw).
        #   5) Add a strong "inside & low" sparse bonus, but keep dense terms informative.
        # -----------------------------

        # -----------------------------
        # Soft stage weights (no hard gates)
        # -----------------------------
        # w_clear ~ 0 when below clearance, -> 1 when above
        temp_stage_clear = 0.02
        w_clear = torch.sigmoid(torch.clamp(clearance / temp_stage_clear, -20.0, 20.0))

        # w_align_xy ~ 0 far, -> 1 near basket opening
        # Choose a scale around basket radius / few cm; use 4cm for smoothness.
        temp_stage_xy = 0.04
        w_align_xy = torch.exp(-dist_xy / temp_stage_xy)
        w_align_xy = torch.clamp(w_align_xy, 0.0, 1.0)

        # "ready to descend" weight: need both clearance and near-xy
        w_ready = w_clear * w_align_xy

        # -----------------------------
        # Dense objectives
        # -----------------------------
        # 1) XY approach (dominant): larger temp so it's not saturated at 0 when far
        temp_xy = 0.20
        r_xy = torch.exp(-dist_xy / temp_xy)
        r_xy = torch.clamp(r_xy, 0.0, 1.0)

        # 2) Clearance (lift) objective: only care until cleared; after that, don't keep rewarding "go higher"
        temp_clear = 0.04
        clear_def = torch.relu(-clearance)  # how much below clearance
        r_clear = torch.exp(-clear_def / temp_clear)
        r_clear = torch.clamp(r_clear, 0.0, 1.0)

        # 3) Z-hold during transfer: only matters when cleared (avoid constant large reward before lift)
        z_target = rim_z  # rim+10cm target height
        z_err = torch.abs(obj_pos[:, 2] - z_target)
        temp_z_hold = 0.06
        r_z_hold = torch.exp(-z_err / temp_z_hold)
        r_z_hold = torch.clamp(r_z_hold, 0.0, 1.0)

        # 4) Descend/place: reward being low *only when aligned and cleared* (prevents dropping elsewhere)
        temp_desc = 0.05
        z_above_goal = torch.relu(obj_pos[:, 2] - 0.1)
        r_low = torch.exp(-z_above_goal / temp_desc)  # 1 when low enough
        r_low = torch.clamp(r_low, 0.0, 1.0)

        # 5) Inside-site shaping: provide dense reward for getting within radius (helps when success is sparse)
        # Use normalized distance to radius; clamp to avoid huge grads
        norm_d = dist_xy / max(radius, 1e-3)
        norm_d = torch.clamp(norm_d, 0.0, 5.0)
        temp_inside = 0.5
        r_inside = torch.exp(-norm_d / temp_inside)
        r_inside = torch.clamp(r_inside, 0.0, 1.0)

        # -----------------------------
        # Constraints / regularizers (small, bounded)
        # -----------------------------
        # Grasp stability: keep object close to hand (non-slipping)
        rel = torch.nan_to_num(self.target_to_hand_pos, nan=0.0, posinf=0.0, neginf=0.0)
        rel_dist = torch.sqrt(torch.clamp((rel * rel).sum(dim=-1), min=0.0) + eps)
        temp_grasp = 0.06
        r_grasp = torch.exp(-rel_dist / temp_grasp)
        r_grasp = torch.clamp(r_grasp, 0.0, 1.0)

        # Orientation fixed: robust proxy from hint (qx and qw near 0 when pointing down)
        hand_quat = torch.nan_to_num(self.hand_quat, nan=0.0, posinf=0.0, neginf=0.0)
        ori_proxy = torch.sqrt(torch.clamp(hand_quat[:, 0] ** 2 + hand_quat[:, 3] ** 2, min=0.0) + eps)
        temp_ori = 0.10
        r_ori = torch.exp(-ori_proxy / temp_ori)
        r_ori = torch.clamp(r_ori, 0.0, 1.0)

        # Smoothness: discourage large joint velocities
        jvel = torch.nan_to_num(self._robot.data.joint_vel, nan=0.0, posinf=0.0, neginf=0.0)
        vmag = torch.sqrt(torch.clamp((jvel * jvel).sum(dim=-1), min=0.0) + eps)
        temp_vel = 6.0
        r_smooth = torch.exp(-vmag / temp_vel)
        r_smooth = torch.clamp(r_smooth, 0.0, 1.0)

        # Joint-limit margin: mild penalty only near limits
        qpos = torch.nan_to_num(self._robot.data.joint_pos, nan=0.0, posinf=0.0, neginf=0.0)
        lower = torch.nan_to_num(self.robot_dof_lower_limits, nan=0.0, posinf=0.0, neginf=0.0)
        upper = torch.nan_to_num(self.robot_dof_upper_limits, nan=0.0, posinf=0.0, neginf=0.0)
        span = torch.clamp(upper - lower, min=eps)
        margin = torch.minimum(qpos - lower, upper - qpos) / span
        margin_min = torch.clamp(margin.min(dim=-1).values, 0.0, 1.0)
        jl_thresh = 0.06
        jl_violation = torch.clamp((jl_thresh - margin_min) / jl_thresh, min=0.0, max=1.0)
        r_joint = 1.0 - jl_violation
        r_joint = torch.clamp(r_joint, 0.0, 1.0)

        # -----------------------------
        # Compose reward (all components ~O(1))
        # -----------------------------
        # Early: prioritize XY approach + clearance.
        # After clearance: prioritize XY + z_hold.
        # When ready (aligned+cleared): prioritize low + inside.
        r_lift_phase = (1.0 - w_clear) * (0.9 * r_clear + 0.4 * r_xy)
        r_transfer_phase = w_clear * (0.9 * r_xy + 0.5 * r_z_hold + 0.3 * r_inside)
        r_place_phase = w_ready * (1.2 * r_low + 1.0 * r_inside + 0.3 * r_xy)

        r_constraints = 0.25 * r_grasp + 0.20 * r_ori + 0.10 * r_smooth + 0.10 * r_joint

        success_bonus = success.float()

        reward = r_lift_phase + r_transfer_phase + r_place_phase + r_constraints + 8.0 * success_bonus

        # -----------------------------
        # Safety
        # -----------------------------
        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        reward = torch.clamp(reward, -2.0, 12.0)
        assert torch.isfinite(reward).all()

        individual_rewards = {
            "r_lift_phase": r_lift_phase,
            "r_transfer_phase": r_transfer_phase,
            "r_place_phase": r_place_phase,
            "r_xy": r_xy,
            "r_inside": r_inside,
            "r_clear": r_clear,
            "r_z_hold": r_z_hold,
            "r_low": r_low,
            "r_grasp": r_grasp,
            "r_ori": r_ori,
            "r_smooth": r_smooth,
            "r_joint": r_joint,
            "w_clear": w_clear,
            "w_align_xy": w_align_xy,
            "w_ready": w_ready,
            "success_bonus": success_bonus,
            "inside_site": inside_site.float(),
            "low_enough": low_enough.float(),
            "dist_xy": dist_xy,
            "clearance": clearance,
        }
        return reward, individual_rewards