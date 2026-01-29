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
from torch.utils.tensorboard import SummaryWriter
# from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

EUREKA_ROOT_DIR = "/home/shaotongchen/workspace_eureka/IsaacLabEureka"
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
    path = f"{EUREKA_ROOT_DIR}/libero/trajs/libero90/libero_90_living_room_scene1_pick_up_the_alphabet_soup_and_put_it_in_the_basket_traj_v2.pkl"
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
        num_envs=1024, env_spacing=3.0, replicate_physics=True, clone_in_fabric=True, 
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
        from isaaclab_eureka.utils import read_pkl
        self.data = read_pkl(self.path) 
        # [num_envs, step, states]
        # first loop over, add padding, then stack.
        self.episodes = self.data["franka"]
        self.target_object_name = "alphabet_soup"
        self.target_region = "basket"
        self.input_direction = 2 # z axis
        super().__init__(cfg, render_mode, **kwargs)
        self.log_dir = "/home/shaotongchen/workspace_eureka/IsaacLabEureka/logs/replay_test"
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

        prim = stage.GetPrimAtPath(f"/World/envs/env_0/{self.target_object_name}")

        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(prim)
        min_corner = bbox.GetRange().GetMin()  # Vec3
        max_corner = bbox.GetRange().GetMax()  # Vec3
        self.target_object_size = torch.tensor((max_corner - min_corner), device=self.device)


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

        self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("panda_rightfinger")[0][0]

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)


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
            cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/{k}",
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=init_states[k]["pos"],
                    rot=init_states[k]["rot"],
                ),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=f"{EUREKA_ROOT_DIR}/libero/COMMON/stable_hope_objects/{k}/usd/{k}.usd",
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

        terminated = self.target_object.data.root_pos_w[:, 2] > 0.6
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated
    
    def _get_rewards(self) -> torch.Tensor:

        return self._compute_rewards(self.actions,self.cfg.action_penalty_scale,)

    # def _get_rewards(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    #     # All tensors on the correct device
    #     device = self.device
    #     eps = 1e-6

    #     # ------------------------------------------------------------
    #     # Retrieve and sanitize useful state variables
    #     # ------------------------------------------------------------
    #     # End-effector / grasp pose
    #     ee_pos = torch.nan_to_num(self.robot_grasp_pos, nan=0.0, posinf=0.0, neginf=0.0)

    #     # Object pose (use the same computation as in observations)
    #     obj_root_pos = torch.nan_to_num(self.target_object.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
    #     env_origins = torch.nan_to_num(self.scene.env_origins, nan=0.0, posinf=0.0, neginf=0.0)

    #     # Compute object top position in environment frame (same as obs)
    #     target_object_root = obj_root_pos - env_origins
    #     target_object_root = torch.nan_to_num(target_object_root, nan=0.0, posinf=0.0, neginf=0.0)
    #     target_object_root[:, 2] += float(self.target_object_size[2])

    #     # Relative pose: vector from EE to object top
    #     to_target = torch.nan_to_num(target_object_root - ee_pos, nan=0.0, posinf=0.0, neginf=0.0)

    #     # Object quaternion
    #     obj_quat = torch.nan_to_num(self.target_object.data.root_quat_w, nan=0.0, posinf=0.0, neginf=0.0)

    #     # Joint info
    #     joint_pos = torch.nan_to_num(self._robot.data.joint_pos, nan=0.0, posinf=0.0, neginf=0.0)
    #     joint_vel = torch.nan_to_num(self._robot.data.joint_vel, nan=0.0, posinf=0.0, neginf=0.0)

    #     # Optionally: gripper (assuming last one or two DOFs are fingers; guard with slicing)
    #     # If not applicable, these terms will be very small and not dominate.
    #     finger_dofs = 2
    #     if joint_pos.shape[1] >= finger_dofs:
    #         finger_pos = joint_pos[:, -finger_dofs:]
    #     else:
    #         finger_pos = torch.zeros((self.num_envs, finger_dofs), device=device)

    #     # ------------------------------------------------------------
    #     # Helper norms
    #     # ------------------------------------------------------------
    #     dist_xy = torch.linalg.norm(to_target[:, :2], dim=-1)            # horizontal distance
    #     dist_z  = torch.abs(to_target[:, 2])                             # vertical offset

    #     # Distance from EE to object top in 3D
    #     dist_3d = torch.linalg.norm(to_target, dim=-1)

    #     # ------------------------------------------------------------
    #     # 1) Approach from above: encourage being above object and aligned in XY
    #     # ------------------------------------------------------------
    #     # Want EE directly above object: small XY distance, positive Z offset
    #     # XY reward (exponential, bounded in [0,1]):
    #     temp_xy = 5.0  # temperature for XY distance shaping
    #     rew_xy = torch.exp(-temp_xy * dist_xy)

    #     # Z-from-above reward: positive when EE is above object top, zero/negative otherwise
    #     # We shape only when EE is not too far in XY to avoid weird gradients.
    #     above_mask = (to_target[:, 2] > 0.0).float()
    #     temp_z_above = 10.0
    #     rew_above_z = above_mask * torch.exp(-temp_z_above * dist_z)

    #     # ------------------------------------------------------------
    #     # 2) Grasping / lifting proxy
    #     # ------------------------------------------------------------
    #     # Detect "lift": object COM height relative to environment origin
    #     # Use the original (non-offset) Z position of object root as lift indicator
    #     obj_root_pos_env = obj_root_pos - env_origins
    #     obj_height = torch.nan_to_num(obj_root_pos_env[:, 2], nan=0.0, posinf=0.0, neginf=0.0)

    #     # Assume initial/rest height is roughly constant; use a small margin above that.
    #     # If env tracks an initial height, we could use it; otherwise, just use a minimal threshold.
    #     # Here we use a fixed threshold.
    #     lift_threshold = 0.06  # meters above table plane (approx)
    #     lifted = (obj_height > lift_threshold).float()

    #     # Lift reward: strong bonus for being lifted
    #     rew_lift_raw = lifted

    #     # Optional smooth shaping: small reward as it starts to lift above table
    #     # Clamp height to a range to keep it stable
    #     h_clamped = torch.clamp(obj_height, 0.0, 0.10)
    #     temp_lift = 20.0
    #     rew_lift_smooth = torch.exp(-temp_lift * (0.10 - h_clamped))  # near 0.10m → ~1

    #     # Combine: strong sparse + smooth dense
    #     rew_lift = 2.0 * rew_lift_raw + 0.5 * rew_lift_smooth

    #     # ------------------------------------------------------------
    #     # 3) Gripper closure when close (encourage grasp)
    #     # ------------------------------------------------------------
    #     # Encourage closing fingers when EE is near the object
    #     # Define a proximity mask in 3D
    #     proximity_radius = 0.05  # m
    #     near_mask = (dist_3d < proximity_radius).float()

    #     # Assume smaller finger distance => better (i.e., joint_pos larger or smaller depending on robot;
    #     # we will simply encourage absolute position dev from mid-range to be small so it doesn't dominate).
    #     # Here we just use negative norm of finger velocities to avoid jitter and slightly reward being still.
    #     finger_vel = torch.nan_to_num(joint_vel[:, -finger_dofs:], nan=0.0, posinf=0.0, neginf=0.0)
    #     finger_vel_norm = torch.linalg.norm(finger_vel, dim=-1)

    #     temp_finger = 2.0
    #     rew_finger_still = near_mask * torch.exp(-temp_finger * finger_vel_norm)

    #     # ------------------------------------------------------------
    #     # 4) Motion smoothness (small joint velocity)
    #     # ------------------------------------------------------------
    #     joint_vel_norm = torch.linalg.norm(joint_vel, dim=-1)
    #     temp_smooth = 0.1
    #     rew_smooth = torch.exp(-temp_smooth * joint_vel_norm)

    #     # ------------------------------------------------------------
    #     # 5) Global distance-to-target shaping
    #     # ------------------------------------------------------------
    #     temp_dist = 5.0
    #     rew_dist = torch.exp(-temp_dist * dist_3d)

    #     # ------------------------------------------------------------
    #     # Weighted sum of reward components
    #     # ------------------------------------------------------------
    #     # Weights chosen heuristically to reach ~0.9 average task score when policy is good
    #     w_xy       = 0.6
    #     w_above_z  = 0.8
    #     w_dist     = 0.4
    #     w_lift     = 2.5
    #     w_finger   = 0.3
    #     w_smooth   = 0.1

    #     reward = (
    #         w_xy * rew_xy
    #         + w_above_z * rew_above_z
    #         + w_dist * rew_dist
    #         + w_lift * rew_lift
    #         + w_finger * rew_finger_still
    #         + w_smooth * rew_smooth
    #     )

    #     # ------------------------------------------------------------
    #     # Sanity: keep rewards finite and safe
    #     # ------------------------------------------------------------
    #     reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
    #     assert torch.isfinite(reward).all(), "Non-finite reward encountered"

    #     # Individual components for logging
    #     individual_rewards = {
    #         "rew_xy": torch.nan_to_num(rew_xy, nan=0.0, posinf=0.0, neginf=0.0),
    #         "rew_above_z": torch.nan_to_num(rew_above_z, nan=0.0, posinf=0.0, neginf=0.0),
    #         "rew_dist": torch.nan_to_num(rew_dist, nan=0.0, posinf=0.0, neginf=0.0),
    #         "rew_lift": torch.nan_to_num(rew_lift, nan=0.0, posinf=0.0, neginf=0.0),
    #         "rew_finger_still": torch.nan_to_num(rew_finger_still, nan=0.0, posinf=0.0, neginf=0.0),
    #         "rew_smooth": torch.nan_to_num(rew_smooth, nan=0.0, posinf=0.0, neginf=0.0),
    #     }

    #     return reward


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
    
    def _get_observations(self) -> dict:
        # object root is not bottom center of the object, the prim is at "/World/envs/env_0/{self.target_object_name}"
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        target_object_pos=self.target_object.data.root_pos_w -self.scene.env_origins
        target_object_pos[:,2] += self.target_object_size[2]
        to_target = target_object_pos - self.robot_grasp_pos
        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
                to_target,
                self.target_object.data.root_quat_w,
            ),
            dim=-1,
        )




        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # auxiliary methods
        # print(f"obs: {obs}")
        # print(f"target_object_pos{target_object_pos}")
        # print(f"robot_grasp_pos{self.robot_grasp_pos}")
        # obj_root_pos_env = self.target_object.data.root_pos_w -self.scene.env_origins
        # obj_height = torch.nan_to_num(obj_root_pos_env[:, 2], nan=0.0, posinf=0.0, neginf=0.0)
        # print(obj_height)

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



    
    def _get_rewards_test(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        import torch

        device = self.device
        num_envs = self.num_envs
        eps = 1e-6

        # ----------------------------------------------------------------------
        # DESIGN ANALYSIS (from previous run)
        # ----------------------------------------------------------------------
        # task_score      : always ~0 → policy never actually lifts / succeeds.
        # r_approach      : grows nicely, optimized.
        # r_xy_align      : grows nicely, optimized.
        # r_hover         : small but non-zero, optimized.
        # r_orient        : identically 0 → gating too strict / never active.
        # r_grasp_pose    : almost 0 → conditions too strict or unreachable.
        # r_close_gripper : ~0 → conditions too strict; policy never gets signal.
        # r_grasp_lift    : 0 → no meaningful object lifting/contact signal.
        # r_no_push       : huge (~60) and flat → dominates and is uninformative.
        # r_smooth        : huge (~60) and flat → dominates and is uninformative.
        # r_obj_stable    : tiny → irrelevant.
        #
        # Net effect: total reward dominated by r_no_push + r_smooth, which the
        # agent gets "for free" without interacting with the object. All real
        # task-related components are tiny or dead → no learning of grasp/lift.
        #
        # NEW STRATEGY:
        #  - Completely remove r_no_push and old r_smooth background bonuses.
        #  - Simplify and strengthen shaping towards top-down grasp and lift:
        #      1) Approach object center.
        #      2) Align EE above object (XY and Z hover).
        #      3) Encourage vertical (top-down) orientation using a smooth term.
        #      4) Encourage moving down to grasp height once above object.
        #      5) Encourage gripper closing when near object.
        #      6) Strong dense reward on object height (pick up).
        #  - Keep scales small (≈0–2 per component) and comparable.
        #  - Mild smoothness penalty only (no giant constant bonuses).
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # Fetch and sanitize state
        # ----------------------------------------------------------------------
        # End-effector pose: try dedicated ee_state if available
        if hasattr(self._robot.data, "ee_state_w"):
            ee_state = torch.nan_to_num(
                self._robot.data.ee_state_w, nan=0.0, posinf=0.0, neginf=0.0
            )
            ee_pos = ee_state[..., 0:3]
            ee_quat = ee_state[..., 3:7]
        elif hasattr(self._robot.data, "root_pos_w") and hasattr(self._robot.data, "root_quat_w"):
            ee_pos = torch.nan_to_num(
                self._robot.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0
            )
            ee_quat = torch.nan_to_num(
                self._robot.data.root_quat_w, nan=0.0, posinf=0.0, neginf=0.0
            )
        else:
            ee_pos = torch.zeros((num_envs, 3), device=device)
            ee_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(num_envs, 1)

        # Object pose / size / velocity
        obj_pos = torch.nan_to_num(
            self.target_object.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0
        )
        obj_quat = torch.nan_to_num(
            self.target_object.data.root_quat_w, nan=0.0, posinf=0.0, neginf=0.0
        )
        obj_linvel = torch.nan_to_num(
            self.target_object.data.root_lin_vel_w, nan=0.0, posinf=0.0, neginf=0.0
        )

        obj_size = torch.nan_to_num(
            self.target_object_size, nan=0.0, posinf=0.0, neginf=0.0
        )
        obj_half_height = 0.5 * obj_size[..., 2]

        # Robot joint velocities
        dof_vel = torch.nan_to_num(
            self._robot.data.joint_vel, nan=0.0, posinf=0.0, neginf=0.0
        )
        vel_scale = getattr(self.cfg, "dof_velocity_scale", 1.0)
        dof_vel_scaled = dof_vel * vel_scale

        # Gripper width if available (for closing encouragement)
        if hasattr(self._robot.data, "gripper_width"):
            gripper_width = torch.nan_to_num(
                self._robot.data.gripper_width, nan=0.0, posinf=0.0, neginf=0.0
            )
            if gripper_width.ndim == 0:
                gripper_width = gripper_width.expand(num_envs)
        else:
            gripper_width = torch.zeros((num_envs,), device=device)

        # Contact force on object as proxy for grasp contact (if available)
        if hasattr(self.target_object.data, "contact_force_body"):
            contact_force = torch.nan_to_num(
                self.target_object.data.contact_force_body, nan=0.0, posinf=0.0, neginf=0.0
            )
            obj_contact_mag = torch.linalg.norm(
                contact_force.view(num_envs, -1), dim=-1
            )
        else:
            obj_contact_mag = torch.zeros((num_envs,), device=device)

        # ----------------------------------------------------------------------
        # Geometric helpers
        # ----------------------------------------------------------------------
        ee_to_obj = ee_pos - obj_pos
        ee_to_obj = torch.nan_to_num(ee_to_obj, nan=0.0, posinf=0.0, neginf=0.0)

        ee_obj_dist = torch.linalg.norm(ee_to_obj, dim=-1)

        ee_to_obj_xy = ee_to_obj.clone()
        ee_to_obj_xy[..., 2] = 0.0
        ee_obj_xy_dist = torch.linalg.norm(ee_to_obj_xy, dim=-1)

        obj_height = obj_pos[..., 2]
        obj_speed = torch.linalg.norm(obj_linvel, dim=-1)

        dof_vel_norm = torch.linalg.norm(dof_vel_scaled, dim=-1)

        # Extract EE z-axis in world from quaternion
        qw, qx, qy, qz = ee_quat.unbind(-1)
        ee_z_world_x = 2.0 * (qx * qz + qw * qy)
        ee_z_world_y = 2.0 * (qy * qz - qw * qx)
        ee_z_world_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        ee_z_axis = torch.stack((ee_z_world_x, ee_z_world_y, ee_z_world_z), dim=-1)
        ee_z_axis = torch.nan_to_num(ee_z_axis, nan=0.0, posinf=0.0, neginf=0.0)
        ee_z_axis = ee_z_axis / (torch.linalg.norm(ee_z_axis, dim=-1, keepdim=True) + eps)

        world_down = torch.tensor([0.0, 0.0, -1.0], device=device).view(1, 3)
        cos_down = torch.sum(ee_z_axis * world_down, dim=-1).clamp(-1.0, 1.0)

        # Some useful heights
        table_height = torch.zeros((num_envs,), device=device)  # assume 0 as baseline
        obj_top = obj_height + obj_half_height

        # ----------------------------------------------------------------------
        # Reward components – all scaled to ~[0, 1.5] range
        # ----------------------------------------------------------------------

        # 1) Approach object center (global shaping)
        #    Encourage reducing full 3D distance independent of orientation.
        approach_temp = 3.0
        r_approach = torch.exp(-approach_temp * ee_obj_dist)
        r_approach = torch.nan_to_num(r_approach, nan=0.0, posinf=0.0, neginf=0.0)

        # 2) XY alignment above object (top-down strategy)
        #    Stronger shaping in XY once reasonably close in 3D.
        xy_temp = 40.0
        r_xy_align = torch.exp(-xy_temp * (ee_obj_xy_dist ** 2))
        r_xy_align = torch.nan_to_num(r_xy_align, nan=0.0, posinf=0.0, neginf=0.0)

        # 3) Hover above object: EE a bit above object top, *not* below (to avoid pushing)
        hover_margin = 0.05  # 5 cm above top
        desired_hover_z = obj_top + hover_margin
        ee_above_hover = ee_pos[..., 2] - desired_hover_z  # positive = above

        # Penalize being below desired hover height, but only when XY is close
        hover_xy_thresh = 0.10
        near_xy_for_hover = (ee_obj_xy_dist < hover_xy_thresh).float()

        hover_z_temp = 40.0
        hover_z_deficit = torch.clamp(-ee_above_hover, min=0.0)
        r_hover = torch.exp(-hover_z_temp * (hover_z_deficit ** 2)) * near_xy_for_hover
        r_hover = torch.nan_to_num(r_hover, nan=0.0, posinf=0.0, neginf=0.0)

        # 4) Top-down orientation: smoothly encourage EE z-axis to point down,
        #    especially when near object (no hard gate that kills gradient).
        orient_temp = 5.0
        near_orient_dist = 0.25
        near_obj_orient_w = torch.exp(
            -orient_temp * torch.clamp(ee_obj_dist - near_orient_dist, min=0.0)
        )
        near_obj_orient_w = torch.nan_to_num(
            near_obj_orient_w, nan=0.0, posinf=0.0, neginf=0.0
        )

        # Reward goes from ~0 at cos_down=-1 to 1 at cos_down=1
        orient_shape_temp = 3.0
        r_orient = torch.exp(orient_shape_temp * (cos_down - 1.0)) * near_obj_orient_w
        r_orient = torch.nan_to_num(r_orient, nan=0.0, posinf=0.0, neginf=0.0)

        # 5) Grasp pose: EE at object center height and very close in XY
        #    Once hovering, move down to around middle of box while staying aligned.
        grasp_z = obj_height + 0.5 * obj_half_height
        ee_z_error_grasp = ee_pos[..., 2] - grasp_z

        grasp_xy_thresh = 0.06
        near_xy_for_grasp = torch.exp(-80.0 * (ee_obj_xy_dist ** 2))
        near_xy_for_grasp = torch.nan_to_num(
            near_xy_for_grasp, nan=0.0, posinf=0.0, neginf=0.0
        )

        grasp_z_temp = 60.0
        r_grasp_pose = torch.exp(-grasp_z_temp * (ee_z_error_grasp ** 2)) * near_xy_for_grasp
        r_grasp_pose = torch.nan_to_num(r_grasp_pose, nan=0.0, posinf=0.0, neginf=0.0)

        # 6) Gripper closing when at grasp pose
        #    Encourage smaller width once close to object center.
        close_temp = 4.0
        # Rough scale: typical max opening maybe ~0.08 m
        width_norm = torch.clamp(gripper_width / (0.08 + eps), 0.0, 2.0)
        r_close_raw = 1.0 - torch.exp(-close_temp * (1.0 - width_norm).clamp(min=0.0))
        r_close_raw = torch.nan_to_num(r_close_raw, nan=0.0, posinf=0.0, neginf=0.0)

        # Gate by proximity in XY and Z
        close_gate_temp = 40.0
        close_xy_gate = torch.exp(-close_gate_temp * (ee_obj_xy_dist ** 2))
        close_z_gate = torch.exp(-close_gate_temp * (ee_z_error_grasp ** 2))
        close_gate = (close_xy_gate * close_z_gate).clamp(0.0, 1.0)
        r_close_gripper = r_close_raw * close_gate
        r_close_gripper = torch.nan_to_num(
            r_close_gripper, nan=0.0, posinf=0.0, neginf=0.0
        )

        # 7) Lift object: main success signal
        #    Dense reward on object height above table; saturates at a modest lift.
        lift_target = obj_half_height + 0.05  # center ~half-height+5cm above table
        height_above_table = torch.clamp(obj_height - table_height, min=0.0)
        lift_temp = 15.0
        # r_lift_height ~1 when obj_height >= lift_target; smooth below that.
        r_lift_height = torch.exp(
            -lift_temp * torch.clamp(lift_target - height_above_table, min=0.0) ** 2
        )
        r_lift_height = torch.nan_to_num(
            r_lift_height, nan=0.0, posinf=0.0, neginf=0.0
        )

        # Optional contact factor – do not gate too aggressively to keep gradient
        contact_temp = 0.02
        contact_factor = 1.0 - torch.exp(-contact_temp * obj_contact_mag)
        contact_factor = torch.nan_to_num(
            contact_factor, nan=0.0, posinf=0.0, neginf=0.0
        )

        # Softly favor lifts with contact, but still reward pure lifting signal
        r_grasp_lift = r_lift_height * (0.3 + 0.7 * contact_factor)
        r_grasp_lift = torch.nan_to_num(
            r_grasp_lift, nan=0.0, posinf=0.0, neginf=0.0
        )

        # 8) Object stability once lifted: prefer low speed when lifted
        stable_temp = 3.0
        is_lifted = (height_above_table > lift_target - 0.01).float()
        r_obj_stable = torch.exp(-stable_temp * obj_speed) * is_lifted
        r_obj_stable = torch.nan_to_num(
            r_obj_stable, nan=0.0, posinf=0.0, neginf=0.0
        )

        # 9) Smoothness: small penalty on joint velocity (no huge constant bonuses)
        smooth_temp = 0.03
        r_smooth = torch.exp(-smooth_temp * dof_vel_norm)
        r_smooth = torch.nan_to_num(r_smooth, nan=0.0, posinf=0.0, neginf=0.0)

        # ----------------------------------------------------------------------
        # Combine rewards with balanced weights
        # ----------------------------------------------------------------------
        # Weights chosen such that a good episode can reach per-step reward ~2–4
        w_approach = 0.4
        w_xy_align = 0.6
        w_hover = 0.5
        w_orient = 0.4
        w_grasp_pose = 0.7
        w_close_gripper = 0.6
        w_grasp_lift = 2.0
        w_obj_stable = 0.3
        w_smooth = 0.1

        reward = (
            w_approach * r_approach
            + w_xy_align * r_xy_align
            + w_hover * r_hover
            + w_orient * r_orient
            + w_grasp_pose * r_grasp_pose
            + w_close_gripper * r_close_gripper
            + w_grasp_lift * r_grasp_lift
            + w_obj_stable * r_obj_stable
            + w_smooth * r_smooth
        )

        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        reward = torch.clamp(reward, min=0.0)

        # Ensure reward is finite
        assert torch.isfinite(reward).all(), "Non-finite reward encountered"

        individual_rewards = {
            "r_approach": r_approach,
            "r_xy_align": r_xy_align,
            "r_hover": r_hover,
            "r_orient": r_orient,
            "r_grasp_pose": r_grasp_pose,
            "r_close_gripper": r_close_gripper,
            "r_grasp_lift": r_grasp_lift,
            "r_obj_stable": r_obj_stable,
            "r_smooth": r_smooth,
        }

        return reward, individual_rewards



    def run_replay(self):
        # [num_envs, step, states]
        # first loop over, add padding, then stack.
        writer = SummaryWriter(self.log_dir)
        num_episodes = len(self.episodes)
        env_ids = torch.arange(50, device=self.device, dtype=torch.long)
        eureka_episode_sums = dict()
        eureka_episode_sums["eureka_total_rewards"] = torch.zeros(num_episodes, device=self.device)
        eureka_episode_sums["oracle_total_rewards"] = torch.zeros(num_episodes, device=self.device)

        num_objects = len(self.object_names)

        # 2. Compute max episode length
        lengths = [len(ep["states"]) for ep in self.episodes]
        if self.debug:
            print(lengths)
        max_len = max(lengths)
        if self.debug:
            print(f"max_len {max_len}")
            print(f"num_envs: {self.num_envs}")
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

        padding_mask = torch.ones((self.num_envs, max_len), dtype=torch.bool)

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
                self.sim.step(render=True)
                if t%2 ==0:
                    self.sim.render()
                self.scene.update(dt=self.physics_dt)
                # import pdb
                # pdb.set_trace()

                # call reward function, should be from eureka
                if hasattr(self, "_get_rewards_eureka"):
                    # dict value size is equal to num of envs
                    rewards_oracle = self._get_rewards_oracle()
                    rewards_eureka, rewards_dict = self._get_rewards_eureka()
                    rewards_oracle_replay =rewards_oracle[:num_episodes]
                    rewards_eureka_replay = rewards_eureka[:num_episodes]
                    rewards_dict_replay = { k: v[:num_episodes] for k, v in rewards_dict.items() }
                    eureka_episode_sums["eureka_total_rewards"] += rewards_eureka_replay
                    eureka_episode_sums["oracle_total_rewards"] += rewards_oracle_replay
                    for key in rewards_dict_replay.keys():
                        if key not in eureka_episode_sums:
                            eureka_episode_sums[key] = torch.zeros(num_episodes, device=self.device)
                        eureka_episode_sums[key] += rewards_dict_replay[key]


                    for k in eureka_episode_sums.keys():
                        writer.add_scalar("Replay/"+k, eureka_episode_sums[k].mean().item(), t) 




        # reset the first num_envs environment, or all


            # env_ids = torch.arange(50, device=self.device, dtype=torch.long)
        self._reset_idx(env_ids)

