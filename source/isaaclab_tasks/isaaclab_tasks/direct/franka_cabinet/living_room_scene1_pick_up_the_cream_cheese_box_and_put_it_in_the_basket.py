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
from isaaclab.utils.math import sample_uniform
# from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
@configclass
class LivingRoomScene1PickUpTheCreamCheeseAndPutItInTheBasketCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333  # 500 timesteps
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
        num_envs=1024, env_spacing=3.0, replicate_physics=True, clone_in_fabric=True
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
            usd_path=f"/home/shaotongchen/workspace_eureka/IsaacLabEureka/libero/COMMON/stable_hope_objects/ketchup/usd/ketchup.usd",
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
            usd_path=f"/home/shaotongchen/workspace_eureka/IsaacLabEureka/libero/COMMON/stable_hope_objects/basket/usd/basket.usd",
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
            usd_path=f"/home/shaotongchen/workspace_eureka/IsaacLabEureka/libero/COMMON/stable_hope_objects/alphabet_soup/usd/alphabet_soup.usd",
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
            usd_path=f"/home/shaotongchen/workspace_eureka/IsaacLabEureka/libero/COMMON/stable_hope_objects/cream_cheese/usd/cream_cheese.usd",
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
            usd_path=f"/home/shaotongchen/workspace_eureka/IsaacLabEureka/libero/COMMON/stable_hope_objects/tomato_sauce/usd/tomato_sauce.usd",
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


class LivingRoomScene1PickUpTheCreamCheeseAndPutItInTheBasket(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: LivingRoomScene1PickUpTheCreamCheeseAndPutItInTheBasketCfg

    def __init__(self, cfg: LivingRoomScene1PickUpTheCreamCheeseAndPutItInTheBasketCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.debug=False
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

        prim = stage.GetPrimAtPath("/World/envs/env_0/cream_cheese/grasp_pose")

        cream_cheese_local_grasp_pose = torch.tensor([0.2, 0.3, 0.03, 1, 0, 0, 0], device=self.device)
        self.cream_cheese_grasp_rot = cream_cheese_local_grasp_pose[3:7].repeat((self.num_envs, 1))



        
   
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(prim)
        min_corner = bbox.GetRange().GetMin()  # Vec3
        max_corner = bbox.GetRange().GetMax()  # Vec3
        cream_cheese_size = torch.tensor((max_corner - min_corner), device=self.device)
        if self.debug:
            print(f"cream_cheese_size{cream_cheese_size}")
        cream_cheese_local_center = torch.tensor((min_corner + max_corner)/2, device=self.device)
        self.cream_cheese_size = cream_cheese_size.repeat((self.num_envs, 1)) # 3
        self.single_size = cream_cheese_size
        
        # basket_pose = get_env_local_pose(
        #     self.scene.env_origins[0],
        #     UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/basket")),
        #     self.device,
        # )

        # basket_local_pos = basket_pose[0:3]
        # basket_local_rot = basket_pose[3:7] # to change
        # self.basket_local_pos = basket_local_pos.repeat((self.num_envs, 1)) # 3
        # self.basket_local_rot = basket_local_rot.repeat((self.num_envs, 1)) # 4


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
        robot_local_pose_pos += torch.tensor([0, 0.04, 0], device=self.device)

        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1)) # 3
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1)) # 4





        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.cream_cheese_inward_axis = torch.tensor([-1, 0, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.cream_cheese_up_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.basket_up_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )


        self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("panda_rightfinger")[0][0]
        # self.drawer_link_idx = self._cabinet.find_bodies("drawer_top")[0][0] # body index is different from joint index
        # self.drawer_top_idx = self._cabinet.find_joints("drawer_top_joint")[0] # joint index
        
        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # self.cream_cheese_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        # self._cream_cheese.data.root_pos_w = torch.zeros((self.num_envs, 3), device=self.device)


        # for termination calculation
        # self.basket_pos = torch.zeros((self.num_envs, 3), device=self.device)
        # self.basket_rot = torch.zeros((self.num_envs, 4), device=self.device)
        # self.object_to_container_relative_pos = torch.zeros((self.num_envs, 3), device=self.device)
        # self.cream_cheese_center = torch.zeros((self.num_envs, 3), device=self.device) # I don't need center?

        # self.helper_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        # self.helper_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)


    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._ketchup = RigidObject(self.cfg.ketchup)
        self._cream_cheese = RigidObject(self.cfg.cream_cheese)
        self._alphabet_soup = RigidObject(self.cfg.alphabet_soup)
        self._tomato_sauce = RigidObject(self.cfg.tomato_sauce)
        self._basket = RigidObject(self.cfg.basket)

        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["ketchup"] = self._ketchup
        self.scene.rigid_objects["cream_cheese"] = self._cream_cheese
        self.scene.rigid_objects["alphabet_soup"] = self._alphabet_soup
        self.scene.rigid_objects["tomato_sauce"] = self._tomato_sauce
        self.scene.rigid_objects["basket"] = self._basket

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


        terminated = self._cream_cheese.data.root_pos_w[:, 2] > 0.6
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    # def _get_terminated(self) -> torch.Tensor: 
        
    #     self.object_to_container_relative_pos = self._cream_cheese.data.root_pos_w -self.basket_pos # grasp pos should be identical for this case
    #     squared = self.object_to_container_relative_pos**2
    #     xy_check = (squared[:, 0] + squared[:, 1]) < (0.05)**2 # 5 cm for both
    #     obj_bottom_z = self._cream_cheese.data.root_pos_w[:, 2] - 0.5 * self.cream_cheese_size[:, 2]
    #     z_check = (obj_bottom_z - self.basket_pos[:, 2]) < 0.05
    #     return xy_check & z_check
    
    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        hand_pos = self._robot.data.body_pos_w[:, self.hand_link_idx]
        hand_rot = self._robot.data.body_quat_w[:, self.hand_link_idx]
        self.robot_grasp_rot , self.robot_grasp_pos = tf_combine(
            hand_rot,
            hand_pos,
            self.robot_local_grasp_rot,
            self.robot_local_grasp_pos,
            )
        
        robot_left_finger_pos = self._robot.data.body_pos_w[:, self.left_finger_link_idx]
        robot_right_finger_pos = self._robot.data.body_pos_w[:, self.right_finger_link_idx]
        

        return self._compute_rewards(
            self.actions,
            # self._cabinet.data.joint_pos,
            self.robot_grasp_pos,
            self._cream_cheese.data.root_pos_w,
            self.robot_grasp_rot,
            self.cream_cheese_grasp_rot,
            robot_left_finger_pos,
            robot_right_finger_pos,
            self._basket.data.root_pos_w, #
            self._basket.data.root_quat_w, #
            self._cream_cheese.data.root_pos_w, #
            self.cream_cheese_size, #
            self.gripper_forward_axis,
            self.cream_cheese_inward_axis,
            self.gripper_up_axis,
            self.cream_cheese_up_axis,
            self.basket_up_axis, #
            self.num_envs,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.open_reward_scale,
            self.cfg.action_penalty_scale,
            self.cfg.finger_reward_scale,
            self._robot.data.joint_pos,

        )

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

        # cabinet state
        # zeros = torch.zeros((len(env_ids), self._cabinet.num_joints), device=self.device)
        # self._cabinet.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)

        rigid_objects = [
            self._cream_cheese,
            self._basket,
            self._alphabet_soup,
            self._ketchup,
            self._tomato_sauce
        ]

        for object in rigid_objects:
            object_default_state = object.data.default_root_state.clone()[env_ids]
            # pos_noise = sample_uniform(-0.05, 0.05, (len(env_ids), 3), device=self.device) # what if goes in the ground
            # # global object positions
            object_default_state[:, 0:3] = (
                object_default_state[:, 0:3] + self.scene.env_origins[env_ids]
            )

            # rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)  # noise for X and Y rotation
            # object_default_state[:, 3:7] = randomize_rotation(
            #     rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
            # )

            object_default_state[:, 7:] = torch.zeros_like(object.data.default_root_state[env_ids, 7:])
            object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
            object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)



        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        hand_pos = self._robot.data.body_pos_w[env_ids, self.hand_link_idx]
        hand_rot = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]

        self.robot_grasp_rot[env_ids] , self.robot_grasp_pos[env_ids] = tf_combine(
            hand_rot,
            hand_pos,
            self.robot_local_grasp_rot[env_ids],
            self.robot_local_grasp_pos[env_ids]
            )
        
    def _get_observations(self) -> dict:
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        to_target = self._cream_cheese.data.root_pos_w - self.robot_grasp_pos

        # task hint: grasp from above without touching it beforehand, since grasping in other directions will push object away.
        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
                to_target,
                self._cream_cheese.data.root_pos_w - self.scene.env_origins,
                self._cream_cheese.data.root_link_vel_w,
                self._cream_cheese.data.root_quat_w, # 4
                self.cream_cheese_size, # 3
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # auxiliary methods


    def _compute_rewards(
        self,
        actions,
        # cabinet_dof_pos,
        franka_grasp_pos,
        cream_cheese_grasp_pos,
        franka_grasp_rot,
        cream_cheese_grasp_rot,
        franka_lfinger_pos,
        franka_rfinger_pos,
        basket_pos, 
        basket_rot,
        cream_cheese_pos,
        cream_cheese_size,
        gripper_forward_axis,
        cream_cheese_inward_axis,
        gripper_up_axis,
        cream_cheese_up_axis,
        basket_up_axis,
        num_envs,
        dist_reward_scale,
        rot_reward_scale,
        open_reward_scale,
        action_penalty_scale,
        finger_reward_scale,
        joint_positions,

    ):
        # distance from hand to the drawer
        # print(f"cream_pos{self._cream_cheese.data.root_pose_w}")
        # print(f"cream{cream_cheese_grasp_pos}") # check
        # print(franka_grasp_pos)


        if self.debug:
            print(f"franka_grasp_rot{franka_grasp_rot}\ncream_cheese_grasp_rot{cream_cheese_grasp_rot}")
        
        d = torch.norm(franka_grasp_pos - cream_cheese_grasp_pos, p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + d**2)
        dist_reward *= dist_reward
        dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

        axis1 = tf_vector(franka_grasp_rot, gripper_forward_axis)
        axis2 = tf_vector(cream_cheese_grasp_rot, cream_cheese_inward_axis)
        axis3 = tf_vector(franka_grasp_rot, gripper_up_axis)
        axis4 = tf_vector(cream_cheese_grasp_rot, cream_cheese_up_axis)

        dot1 = (
            torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of forward axis for gripper
        dot2 = (
            torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of up axis for gripper
        # reward for matching the orientation of the hand to the drawer (fingers wrapped)
        rot_reward = 0.5 * (torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2)

        # regularization on the actions (summed for each environment)
        action_penalty = torch.sum(actions**2, dim=-1)


        # penalty for distance of each finger from the drawer handle
        lfinger_dist = franka_lfinger_pos[:, 1] - cream_cheese_grasp_pos[:, 1]
        rfinger_dist = cream_cheese_grasp_pos[:, 1] - franka_rfinger_pos[:, 1]
        finger_dist_penalty = torch.zeros_like(lfinger_dist)
        finger_dist_penalty += torch.where(lfinger_dist < 0, lfinger_dist, torch.zeros_like(lfinger_dist))
        finger_dist_penalty += torch.where(rfinger_dist < 0, rfinger_dist, torch.zeros_like(rfinger_dist))
        threshold = torch.min(self.single_size, dim=-1).values + 0.02
        mask = d < threshold                   # shape: (num_envs,)
        penalty = torch.zeros_like(d)
        penalty[mask] = finger_dist_penalty[mask]                  # or any penalty function you want




        rewards = (
            dist_reward_scale * dist_reward
            # + rot_reward_scale * rot_reward
            + finger_reward_scale * finger_dist_penalty
            - action_penalty_scale * action_penalty
            # + lift_reward
        )

        self.extras["log"] = {
            "dist_reward": (dist_reward_scale * dist_reward).mean(),
            # "rot_reward": (rot_reward_scale * rot_reward).mean(),
            "action_penalty": (-action_penalty_scale * action_penalty).mean(),
            "left_finger_distance_reward": (finger_reward_scale * lfinger_dist).mean(),
            "right_finger_distance_reward": (finger_reward_scale * rfinger_dist).mean(),
            "finger_dist_penalty": (finger_reward_scale * finger_dist_penalty).mean(),
        }

        rewards = torch.where(self._cream_cheese.data.root_pos_w[:, 2] > 0.03, rewards + 0.25, rewards)
        rewards = torch.where(self._cream_cheese.data.root_pos_w[:, 2] > 0.2, rewards + 0.25, rewards)
        rewards = torch.where(self._cream_cheese.data.root_pos_w[:, 2] > 0.35, rewards + 0.25, rewards)
        rewards = torch.where(self._cream_cheese.data.root_pos_w[:, 2] > 0.5, rewards + 0.25, rewards)
        # rewards = torch.where(cream_cheese_pos[:,2] > 0.1, rewards + 1, rewards)
        return rewards






