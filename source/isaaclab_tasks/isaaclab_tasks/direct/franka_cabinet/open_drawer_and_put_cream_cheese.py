# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector

from pxr import UsdGeom

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
from isaaclab.utils.math import sample_uniform, quat_conjugate
from isaaclab.sensors.contact_sensor.contact_sensor import ContactSensor
from isaaclab.sensors import ContactSensorCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from isaaclab_eureka.utils import eureka_root_dir


@configclass
class OpenDrawerAndPutCreamCheeseCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 12.0  # ~720 timesteps at dt=1/120, decimation=2
    decimation = 2
    action_space = 8
    observation_space = 31
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
        num_envs=1024, env_spacing=3.0, replicate_physics=True, clone_in_fabric=False
    )

    # robot — exactly as franka_cabinet_env.py, but activate_contact_sensors=True
    robot = ArticulationCfg(
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
            pos=(1.0, 0.0, 0.0),
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

    # cabinet — exactly as franka_cabinet_env.py, but activate_contact_sensors=True
    cabinet = ArticulationCfg(
        prim_path="/World/envs/env_.*/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0, 0.4),
            rot=(0.1, 0.0, 0.0, 0.0),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                effort_limit_sim=87.0,
                stiffness=0,
                damping=1.0,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit_sim=87.0,
                stiffness=10.0,
                damping=2.5,
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

    # cream_cheese object
    root = eureka_root_dir()
    cream_cheese = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cream_cheese",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[1.0, 0.4, 0.0], rot=[0.7071, 0.7071, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{root}/libero/COMMON/stable_hope_objects/cream_cheese/usd/cream_cheese.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
        ),
    )

    randomize_init: bool = True

    action_scale = 7.5
    dof_velocity_scale = 0.1

    # reward scales (Phase 1: open drawer — mirrors franka_cabinet_env)
    dist_reward_scale: float = 1.5      # hand → drawer handle
    rot_reward_scale: float = 1.5       # gripper orientation alignment
    open_reward_scale: float = 10.0     # drawer joint position
    finger_reward_scale: float = 2.0    # finger z-distance from handle
    action_penalty_scale: float = 0.05  # action regularisation

    # reward scales (Phase 2: pick cheese + place in drawer)
    cheese_dist_reward_scale: float = 1.5   # hand → cheese (when drawer open)
    placement_reward_scale: float = 10.0    # cheese → drawer interior

    # success + time
    success_bonus_scale: float = 5.0   # one-shot bonus when task succeeds (added after clamp)
    time_penalty_scale: float = 3.0    # penalizes slow success: grows with episode_length_buf

    # task params
    drawer_open_threshold: float = 0.30       # meters; drawer is "open" when joint_pos > this
    drawer_placement_tolerance: float = 0.15  # meters; cream_cheese is "inside" when closer than this


class OpenDrawerAndPutCreamCheese(DirectRLEnv):
    cfg: OpenDrawerAndPutCreamCheeseCfg

    def __init__(self, cfg: OpenDrawerAndPutCreamCheeseCfg, render_mode: str | None = None, **kwargs):
        self.debug_vis = True
        self.history_len = 2
        self.randomize_init = cfg.randomize_init
        super().__init__(cfg, render_mode, **kwargs)

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
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

        # DOF limits and speed scales
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint1")[0]] = 0.1
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint2")[0]] = 0.1
        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        stage = get_current_stage()

        # Robot local grasp pose — finger tip midpoint in hand-local frame
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
        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))

        # Drawer handle local grasp pose (handle is 0.3 m in front of the drawer body)
        drawer_local_grasp_pose = torch.tensor([0.3, 0.01, 0.0, 1.0, 0.0, 0.0, 0.0], device=self.device)
        self.drawer_local_grasp_pos = drawer_local_grasp_pose[0:3].repeat((self.num_envs, 1))
        self.drawer_local_grasp_rot = drawer_local_grasp_pose[3:7].repeat((self.num_envs, 1))

        # Drawer interior placement target in drawer-body-local frame
        self.drawer_interior_local_offset = torch.tensor([0.0, 0.0, 0.05], device=self.device).repeat((self.num_envs, 1))
        self.drawer_interior_local_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))

        # Axis tensors for drawer orientation reward
        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat((self.num_envs, 1))
        self.drawer_inward_axis   = torch.tensor([-1, 0, 0], device=self.device, dtype=torch.float32).repeat((self.num_envs, 1))
        self.gripper_up_axis      = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat((self.num_envs, 1))
        self.drawer_up_axis       = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat((self.num_envs, 1))

        # Body/joint indices
        self.hand_link_idx          = self._robot.find_bodies("panda_link7")[0][0]
        self.left_finger_body_idx   = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_body_idx  = self._robot.find_bodies("panda_rightfinger")[0][0]
        self.left_finger_joint_idx  = self._robot.find_joints("panda_finger_joint2")[0][0]
        self.right_finger_joint_idx = self._robot.find_joints("panda_finger_joint1")[0][0]
        self.drawer_link_idx        = self._cabinet.find_bodies("drawer_top")[0][0]
        self.drawer_top_joint_idx   = self._cabinet.find_joints("drawer_top_joint")[0][0]

        # State tensors
        self.robot_grasp_rot    = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos    = torch.zeros((self.num_envs, 3), device=self.device)
        self.drawer_grasp_rot   = torch.zeros((self.num_envs, 4), device=self.device)
        self.drawer_grasp_pos   = torch.zeros((self.num_envs, 3), device=self.device)
        self.drawer_interior_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # Grasp flags and distance history
        self.grasped_cheese   = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.grasped_drawer   = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.past_cheese_dist = torch.ones((self.num_envs, self.history_len), device=self.device)
        self.past_drawer_dist = torch.ones((self.num_envs, self.history_len), device=self.device)

        self.actions = torch.zeros((self.num_envs, cfg.action_space), device=self.device)

        if self.debug_vis:
            self.visualizer = self.define_markers()

    def _setup_scene(self):
        self._robot       = Articulation(self.cfg.robot)
        self._cabinet     = Articulation(self.cfg.cabinet)
        self._cream_cheese = RigidObject(self.cfg.cream_cheese)

        self.scene.articulations["robot"]         = self._robot
        self.scene.articulations["cabinet"]       = self._cabinet
        self.scene.rigid_objects["cream_cheese"]  = self._cream_cheese

        self.cfg.terrain.num_envs    = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Contact sensors: fingers ↔ cream_cheese/object
        left_cheese_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/panda_leftfinger",
            update_period=0.0,
            history_length=2 * self.history_len,
            track_air_time=True,
            filter_prim_paths_expr=["/World/envs/env_.*/cream_cheese/object"],
            track_contact_points=True,
            max_contact_data_count_per_prim=4,
        )
        right_cheese_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/panda_rightfinger",
            update_period=0.0,
            history_length=2 * self.history_len,
            track_air_time=True,
            filter_prim_paths_expr=["/World/envs/env_.*/cream_cheese/object"],
            track_contact_points=True,
            max_contact_data_count_per_prim=4,
        )
        # Contact sensors: fingers ↔ drawer_handle_top
        left_drawer_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/panda_leftfinger",
            update_period=0.0,
            history_length=2 * self.history_len,
            track_air_time=True,
            filter_prim_paths_expr=["/World/envs/env_.*/Cabinet/drawer_handle_top"],
            track_contact_points=True,
            max_contact_data_count_per_prim=4,
        )
        right_drawer_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/panda_rightfinger",
            update_period=0.0,
            history_length=2 * self.history_len,
            track_air_time=True,
            filter_prim_paths_expr=["/World/envs/env_.*/Cabinet/drawer_handle_top"],
            track_contact_points=True,
            max_contact_data_count_per_prim=4,
        )

        self.scene.sensors["left_cheese_sensor"]  = ContactSensor(left_cheese_cfg)
        self.scene.sensors["right_cheese_sensor"] = ContactSensor(right_cheese_cfg)
        self.scene.sensors["left_drawer_sensor"]  = ContactSensor(left_drawer_cfg)
        self.scene.sensors["right_drawer_sensor"] = ContactSensor(right_drawer_cfg)
        self.scene.sensors["left_cheese_sensor"].set_debug_vis(False)
        self.scene.sensors["right_cheese_sensor"].set_debug_vis(False)
        self.scene.sensors["left_drawer_sensor"].set_debug_vis(False)
        self.scene.sensors["right_drawer_sensor"].set_debug_vis(False)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # -------------------------------------------------------------------------
    # Pre/post physics callbacks
    # -------------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        arm_actions = actions[:, :-1]          # 7-dim
        grip_action = actions[:, -1]           # 1-dim

        # Arm: continuous position integration
        arm_targets = self.robot_dof_targets[:, :-2] + (
            self.robot_dof_speed_scales[:-2] * self.dt * arm_actions * self.cfg.action_scale
        )
        self.robot_dof_targets[:, :-2] = torch.clamp(
            arm_targets,
            self.robot_dof_lower_limits[:-2],
            self.robot_dof_upper_limits[:-2],
        )

        # Gripper: binary open/close
        open_mask = grip_action >= 0
        self.robot_dof_targets[open_mask, -2:]  = self.robot_dof_upper_limits[-1]
        self.robot_dof_targets[~open_mask, -2:] = self.robot_dof_lower_limits[-1]

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # -------------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        self.grasped_cheese = self._grasp_detection_generic(
            self.scene["left_cheese_sensor"],
            self.scene["right_cheese_sensor"],
            self.past_cheese_dist,
            dist_threshold=0.04,
        )
        self.grasped_drawer = self._grasp_detection_generic(
            self.scene["left_drawer_sensor"],
            self.scene["right_drawer_sensor"],
            self.past_drawer_dist,
            dist_threshold=0.06,
        )
        self._update_past_dists()

        if self.debug_vis:
            self.debug_vis_mine()

        drawer_joint_pos = self._cabinet.data.joint_pos[:, self.drawer_top_joint_idx]
        drawer_open = drawer_joint_pos > self.cfg.drawer_open_threshold

        cheese_pos = self._cream_cheese.data.root_pos_w
        d_cheese = torch.norm(cheese_pos - self.drawer_interior_pos, p=2, dim=-1)
        cheese_inside = d_cheese < self.cfg.drawer_placement_tolerance

        terminated = drawer_open & cheese_inside
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        rewards, log_dict = self._get_rewards_test()
        self.extras["log"] = log_dict
        return rewards

    def _get_rewards_test(self) -> tuple[torch.Tensor, dict]:
        N = self.num_envs
        cheese_pos       = self._cream_cheese.data.root_pos_w
        drawer_joint_pos = self._cabinet.data.joint_pos[:, self.drawer_top_joint_idx]

        # ── Phase 1: approach and open drawer (mirrors franka_cabinet_env) ──────

        # Distance: hand → drawer handle
        d_hand_drawer = torch.norm(self.robot_grasp_pos - self.drawer_grasp_pos, p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + d_hand_drawer ** 2)
        dist_reward = dist_reward * dist_reward
        dist_reward = torch.where(d_hand_drawer <= 0.02, dist_reward * 2, dist_reward)

        # Rotation: gripper forward ↔ drawer inward, gripper up ↔ drawer up
        axis1 = tf_vector(self.robot_grasp_rot, self.gripper_forward_axis)
        axis2 = tf_vector(self.drawer_grasp_rot, self.drawer_inward_axis)
        axis3 = tf_vector(self.robot_grasp_rot, self.gripper_up_axis)
        axis4 = tf_vector(self.drawer_grasp_rot, self.drawer_up_axis)
        dot1 = torch.bmm(axis1.view(N, 1, 3), axis2.view(N, 3, 1)).squeeze(-1).squeeze(-1)
        dot2 = torch.bmm(axis3.view(N, 1, 3), axis4.view(N, 3, 1)).squeeze(-1).squeeze(-1)
        rot_reward = 0.5 * (torch.sign(dot1) * dot1 ** 2 + torch.sign(dot2) * dot2 ** 2)

        # Open reward: raw drawer joint position
        open_reward = drawer_joint_pos

        # Finger z-distance penalty relative to drawer handle
        lfinger_pos = self._robot.data.body_pos_w[:, self.left_finger_body_idx]
        rfinger_pos = self._robot.data.body_pos_w[:, self.right_finger_body_idx]
        lfinger_dist = lfinger_pos[:, 2] - self.drawer_grasp_pos[:, 2]
        rfinger_dist = self.drawer_grasp_pos[:, 2] - rfinger_pos[:, 2]
        finger_dist_penalty = torch.zeros(N, device=self.device)
        finger_dist_penalty += torch.where(lfinger_dist < 0, lfinger_dist, torch.zeros_like(lfinger_dist))
        finger_dist_penalty += torch.where(rfinger_dist < 0, rfinger_dist, torch.zeros_like(rfinger_dist))

        # Action regularisation
        action_penalty = torch.sum(self.actions ** 2, dim=-1)

        # ── Phase 2: pick cheese and place in open drawer ────────────────────────

        # Distance: hand → cheese (rewarded once drawer starts opening)
        d_hand_cheese = torch.norm(self.robot_grasp_pos - cheese_pos, p=2, dim=-1)
        cheese_dist_reward_raw = 1.0 / (1.0 + d_hand_cheese ** 2)
        cheese_dist_reward_raw = cheese_dist_reward_raw * cheese_dist_reward_raw
        cheese_dist_reward = cheese_dist_reward_raw * torch.clamp(drawer_joint_pos / self.cfg.drawer_open_threshold, 0.0, 1.0)

        # Placement reward: cheese → drawer interior, weighted by drawer open fraction
        d_cheese_interior = torch.norm(cheese_pos - self.drawer_interior_pos, p=2, dim=-1)
        placement_reward_raw = 1.0 / (1.0 + d_cheese_interior ** 2)
        placement_reward_raw = placement_reward_raw * placement_reward_raw
        drawer_open_frac = torch.clamp(drawer_joint_pos / self.cfg.drawer_open_threshold, 0.0, 1.0)
        placement_reward = placement_reward_raw * drawer_open_frac

        # ── Combine ─────────────────────────────────────────────────────────────
        rewards = (
            self.cfg.dist_reward_scale      * dist_reward
            + self.cfg.rot_reward_scale     * rot_reward
            + self.cfg.open_reward_scale    * open_reward
            + self.cfg.finger_reward_scale  * finger_dist_penalty
            - self.cfg.action_penalty_scale * action_penalty
            + self.cfg.cheese_dist_reward_scale * cheese_dist_reward
            + self.cfg.placement_reward_scale   * placement_reward
        )

        # Milestone bonuses for opening drawer (same thresholds as franka_cabinet_env)
        rewards = torch.where(drawer_joint_pos > 0.01, rewards + 0.25, rewards)
        rewards = torch.where(drawer_joint_pos > 0.20, rewards + 0.25, rewards)
        rewards = torch.where(drawer_joint_pos > 0.35, rewards + 0.25, rewards)

        # Clamp shaped rewards for stability
        rewards = torch.clamp(rewards, -5.0, 5.0)

        # Success and time signals — added after clamp so they are not truncated
        d_cheese_to_interior = torch.norm(cheese_pos - self.drawer_interior_pos, p=2, dim=-1)
        success = (drawer_joint_pos > self.cfg.drawer_open_threshold) & (d_cheese_to_interior < self.cfg.drawer_placement_tolerance)

        rewards = rewards + self.cfg.success_bonus_scale * success.float()

        # Time penalty: penalizes slow episodes proportionally to success rate;
        # time_frac ∈ [0,1], penalty grows from 0→time_penalty_scale across episode
        time_frac = self.episode_length_buf.float() / self.max_episode_length
        rewards = rewards - self.cfg.time_penalty_scale * time_frac * success.float()

        log_dict = {
            "dist_reward":        (self.cfg.dist_reward_scale   * dist_reward).mean(),
            "rot_reward":         (self.cfg.rot_reward_scale    * rot_reward).mean(),
            "open_reward":        (self.cfg.open_reward_scale   * open_reward).mean(),
            "finger_dist_penalty":(self.cfg.finger_reward_scale * finger_dist_penalty).mean(),
            "action_penalty":     (-self.cfg.action_penalty_scale * action_penalty).mean(),
            "cheese_dist_reward": (self.cfg.cheese_dist_reward_scale * cheese_dist_reward).mean(),
            "placement_reward":   (self.cfg.placement_reward_scale  * placement_reward).mean(),
            "drawer_joint_pos":   drawer_joint_pos.mean(),
            "d_cheese_to_interior": d_cheese_to_interior.mean(),
            "success":            success.float().mean(),
            "time_penalty":       (self.cfg.time_penalty_scale * time_frac * success.float()).mean(),
        }
        return rewards, log_dict

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        # Robot: default joint pos (+ uniform noise when randomize_init)
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        if self.randomize_init:
            joint_pos += sample_uniform(-0.125, 0.125, (len(env_ids), self._robot.num_joints), self.device)
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Cabinet: fully closed
        zeros_cab = torch.zeros((len(env_ids), self._cabinet.num_joints), device=self.device)
        self._cabinet.write_joint_state_to_sim(zeros_cab, zeros_cab, env_ids=env_ids)

        # cream_cheese: default pos (+ small XY noise when randomize_init)
        cheese_state = self._cream_cheese.data.default_root_state.clone()[env_ids]
        cheese_state[:, 0:3] += self.scene.env_origins[env_ids]
        if self.randomize_init:
            cheese_state[:, 0:2] += sample_uniform(-0.03, 0.03, (len(env_ids), 2), self.device)
        cheese_state[:, 7:] = 0.0
        self._cream_cheese.write_root_pose_to_sim(cheese_state[:, :7], env_ids)
        self._cream_cheese.write_root_velocity_to_sim(cheese_state[:, 7:], env_ids)

        self._compute_intermediate_values(env_ids)
        self.grasped_cheese[env_ids]   = False
        self.grasped_drawer[env_ids]   = False
        self.past_cheese_dist[env_ids] = 1.0
        self.past_drawer_dist[env_ids] = 1.0
        self.actions[env_ids]          = 0.0

    def _get_observations(self) -> dict:
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        cheese_pos       = self._cream_cheese.data.root_pos_w
        drawer_joint_pos = self._cabinet.data.joint_pos[:, self.drawer_top_joint_idx].unsqueeze(-1)
        drawer_joint_vel = self._cabinet.data.joint_vel[:, self.drawer_top_joint_idx].unsqueeze(-1)

        obs = torch.cat(
            (
                dof_pos_scaled,                                        # 9
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,  # 9
                self.drawer_grasp_pos - self.robot_grasp_pos,          # 3  hand → drawer handle
                drawer_joint_pos,                                      # 1  drawer opening (m)
                drawer_joint_vel,                                      # 1
                cheese_pos - self.robot_grasp_pos,                     # 3  hand → cheese
                self.drawer_interior_pos - cheese_pos,                 # 3  cheese → placement target
                self.grasped_cheese.float().unsqueeze(-1),             # 1
                self.grasped_drawer.float().unsqueeze(-1),             # 1
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # -------------------------------------------------------------------------
    # Auxiliary methods
    # -------------------------------------------------------------------------

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        hand_pos  = self._robot.data.body_pos_w[env_ids, self.hand_link_idx]
        hand_rot  = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]
        drawer_pos = self._cabinet.data.body_pos_w[env_ids, self.drawer_link_idx]
        drawer_rot = self._cabinet.data.body_quat_w[env_ids, self.drawer_link_idx]

        # Robot TCP grasp pose
        self.robot_grasp_rot[env_ids], self.robot_grasp_pos[env_ids] = tf_combine(
            hand_rot, hand_pos,
            self.robot_local_grasp_rot[env_ids],
            self.robot_local_grasp_pos[env_ids],
        )

        # Drawer handle grasp pose
        self.drawer_grasp_rot[env_ids], self.drawer_grasp_pos[env_ids] = tf_combine(
            drawer_rot, drawer_pos,
            self.drawer_local_grasp_rot[env_ids],
            self.drawer_local_grasp_pos[env_ids],
        )

        # Drawer interior placement target (at drawer body center + local offset)
        _, self.drawer_interior_pos[env_ids] = tf_combine(
            drawer_rot, drawer_pos,
            self.drawer_interior_local_rot[env_ids],
            self.drawer_interior_local_offset[env_ids],
        )

    def _update_past_dists(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        new_cheese_dist = torch.linalg.norm(
            self._cream_cheese.data.root_pos_w[env_ids] - self.robot_grasp_pos[env_ids], dim=-1
        )
        self.past_cheese_dist[env_ids] = torch.cat(
            [new_cheese_dist.unsqueeze(1), self.past_cheese_dist[env_ids, :-1]], dim=1
        )

        new_drawer_dist = torch.linalg.norm(
            self.drawer_grasp_pos[env_ids] - self.robot_grasp_pos[env_ids], dim=-1
        )
        self.past_drawer_dist[env_ids] = torch.cat(
            [new_drawer_dist.unsqueeze(1), self.past_drawer_dist[env_ids, :-1]], dim=1
        )

    def _grasp_detection_generic(
        self,
        left_sensor: ContactSensor,
        right_sensor: ContactSensor,
        past_dist: torch.Tensor,
        dist_threshold: float = 0.04,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        always_small = torch.all(past_dist[env_ids] < dist_threshold, dim=1)

        left_forces_hist  = left_sensor.data.force_matrix_w_history[env_ids]
        right_forces_hist = right_sensor.data.force_matrix_w_history[env_ids]

        left_force_mag  = torch.linalg.norm(left_forces_hist,  dim=-1)
        right_force_mag = torch.linalg.norm(right_forces_hist, dim=-1)

        left_contact  = left_force_mag  > 0
        right_contact = right_force_mag > 0

        stable_left_contact  = torch.all(left_contact,  dim=tuple(range(1, left_contact.ndim)))
        stable_right_contact = torch.all(right_contact, dim=tuple(range(1, right_contact.ndim)))

        eps = 1e-8
        left_dir  = left_forces_hist  / (left_force_mag.unsqueeze(-1)  + eps)
        right_dir = right_forces_hist / (right_force_mag.unsqueeze(-1) + eps)
        dir_dot = torch.sum(left_dir * right_dir, dim=-1)

        mutual_contact   = left_contact & right_contact
        opposing_contact = mutual_contact & (dir_dot < 0.0)
        stable_opposing  = torch.all(opposing_contact, dim=tuple(range(1, opposing_contact.ndim)))

        grasped = stable_left_contact & stable_right_contact & stable_opposing & always_small
        return grasped

    # -------------------------------------------------------------------------
    # Debug visualisation
    # -------------------------------------------------------------------------

    def define_markers(self) -> VisualizationMarkers:
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
        # Show: robot TCP (blue), drawer handle target (green), interior placement target (red)
        translations = torch.cat(
            [self.robot_grasp_pos, self.drawer_grasp_pos, self.drawer_interior_pos], dim=0
        )
        orientations = torch.cat(
            [self.robot_grasp_rot, self.drawer_grasp_rot, self.drawer_grasp_rot], dim=0
        )
        self.visualizer.visualize(translations=translations, orientations=orientations)
