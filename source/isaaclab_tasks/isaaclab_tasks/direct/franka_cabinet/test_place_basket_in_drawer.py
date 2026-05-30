# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Place-basket-in-open-drawer task.

Scene
-----
* Franka Panda at local (0.4, 0, 0)
* Sektion cabinet at local (0.8, 0, 0.4) with TOP drawer pre-opened (0.39 m)
* Basket rigid object – initially on the table, position randomised each episode

Goal
----
Pick up the basket and place it inside the open top drawer.

Inheritance
-----------
Inherits robot / grasp-detection / action / visualisation infrastructure from
TestPickItUp. Overrides scene setup, reset, dones, observations, and reward.
DirectRLEnv.__init__ is called directly to skip the demo-trajectory loading
that TestPickItUp.__init__ performs.
"""

from __future__ import annotations

import os

import torch
from pxr import UsdGeom, Usd

from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim import SimulationCfg
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass, convert_dict_to_backend
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.math import (
    transform_points, quat_conjugate, quat_apply, quat_mul,
)
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sensors.contact_sensor.contact_sensor import ContactSensor
from isaaclab.scene import InteractiveSceneCfg

from isaaclab_eureka.utils import eureka_root_dir

from .test_pick_it_up import TestPickItUp, TestPickItUpCfg


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@configclass
class TestPlaceCreamCheeseInDrawerCfg(TestPickItUpCfg):
    """Configuration for the place-basket-in-drawer task."""

    episode_length_s: float = 12.0
    observation_space: int = 71   # matches actual obs vector (see _get_observations)

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024, env_spacing=3.0, replicate_physics=True, clone_in_fabric=False,
    )

    # Nullify parent object configs – only cream_cheese + cabinet used in this task.
    robot         = None
    ketchup       = None
    cream_cheese  = None   # re-spawned in _setup_scene with the correct USD path
    alphabet_soup = None
    tomato_sauce  = None
    basket        = None

    # Cabinet – same position as franka_cabinet_env.py (robot at 1.0, cabinet at 0.0 → 1 m gap)
    cabinet_pos: tuple = (-0.2, 0.0, 0.4)
    drawer_open_pos: float = 0.39   # metres – joint position at episode start

    # Zero-centred noise applied on top of cream_cheese init_state in _setup_scene.
    # (x=1.0, y=0.4 base position defined there; these are uniform half-widths.)
    cream_cheese_pos_noise: tuple = (0.10, 0.10)

    # Randomisation
    arm_joint_noise: float = 0.08   # std (rad) for all 7 arm joints at reset


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class TestPlaceCreamCheeseInDrawer(TestPickItUp):
    """
    Pick up the basket and place it inside the pre-opened top drawer.

    Inherits grasp detection, action application, markers, and helper
    computation from TestPickItUp.  Demo-trajectory loading is bypassed
    by calling DirectRLEnv.__init__ directly.
    """

    cfg: TestPlaceCreamCheeseInDrawerCfg

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def __init__(self, cfg: TestPlaceCreamCheeseInDrawerCfg, render_mode=None, **kwargs):
        seed = cfg.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

        # Flags needed BEFORE DirectRLEnv.__init__ (which calls _setup_scene)
        self.discrete_action = True
        self.debug_vis = True
        self.show_robot_grasp = True
        self.show_target_object = True
        self.show_target_grasp_pose = False
        self.camera_sensor_record = cfg.camera_sensor_record
        self.log_mine = False
        self.start_in_air = False
        self.randomize_init = cfg.randomize_init
        self.root = eureka_root_dir()
        self.target_object_name = "cream_cheese"
        self.target_site_name = "drawer"      # logical name – tracked from articulation
        self.input_direction = 2
        self.history_len = 2
        self.num_stages = 5
        self.debug = False

        # No demo data – fixed episode length handled by cfg
        self.start_idx_in_episode = 0

        # Call DirectRLEnv.__init__ → triggers _setup_scene (overridden below)
        DirectRLEnv.__init__(self, cfg, render_mode, **kwargs)

        # ------------------------------------------------------------------
        # Post-scene setup
        # ------------------------------------------------------------------
        offset = torch.tensor([0.0, 0.0, 0.04], device=self.device)  # cream_cheese grasp offset

        # Robot joint limits / speed scales
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint1")[0]] = 0.1
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint2")[0]] = 0.1
        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        # Drawer body index in cabinet articulation (body "drawer_bottom")
        drawer_body_ids, _ = self._cabinet.find_bodies("drawer_bottom")
        self.drawer_body_idx = drawer_body_ids[0]

        # Drawer joint index (prismatic) — for detecting drawer displacement.
        self.drawer_joint_idx = self._cabinet.find_joints("drawer_bottom_joint")[0][0]

        # Drawer interior half-sizes (Sektion cabinet top drawer, approximate).
        # X = depth direction (drawer pulls out in +X), Y = width, Z = height.
        # Tune if basket success detection is off.
        self.drawer_half_x = 0.18   # depth half
        self.drawer_half_y = 0.17   # width half
        self.drawer_half_z = 0.06   # height half

        # Interior centre relative to drawer body origin (body centre ≈ drawer centre).
        # The handle end is at +0.3 m in local X, so interior is slightly inward.
        self.drawer_interior_local = torch.tensor([-0.05, 0.0, 0.0], device=self.device)

        # ------------------------------------------------------------------
        # Basket bounding-box (computed once from env_0)
        # ------------------------------------------------------------------
        stage = get_current_stage()
        prim = stage.GetPrimAtPath(f"/World/envs/env_0/{self.target_object_name}/object")
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(prim)
        lo = bbox.GetRange().GetMin()
        hi = bbox.GetRange().GetMax()
        corners_world0 = torch.tensor(
            [[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
             [lo[0], hi[1], lo[2]], [lo[0], lo[1], hi[2]],
             [hi[0], hi[1], lo[2]], [hi[0], lo[1], hi[2]],
             [lo[0], hi[1], hi[2]], [hi[0], hi[1], hi[2]]],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0)  # [1, 8, 3]
        pos0  = self.target_object.data.root_pos_w[0]
        quat0 = self.target_object.data.root_quat_w[0]
        qi    = quat_conjugate(quat0)
        inv_p = -quat_apply(qi, pos0)
        local_c0 = transform_points(points=corners_world0, pos=inv_p, quat=qi).squeeze(0)  # [8,3]
        self.local_corners_init   = local_c0.unsqueeze(0).repeat(self.num_envs, 1, 1)
        lmin = self.local_corners_init.min(dim=1).values
        lmax = self.local_corners_init.max(dim=1).values
        self.target_object_size   = lmax - lmin
        self.local_centers_init   = ((lmin + lmax) / 2.0).unsqueeze(1)
        self.local_centers        = torch.zeros_like(self.local_centers_init)
        self.corners_target_obj   = torch.zeros_like(self.local_corners_init)
        self.corners_target_obj_to_hand_pos = torch.zeros((self.num_envs, 24), device=self.device)

        # ------------------------------------------------------------------
        # Robot grasp frame (identical computation to TestPickItUp)
        # ------------------------------------------------------------------
        def _env_local_pose(env_pos, xformable, device):
            wt  = xformable.ComputeLocalToWorldTransform(0)
            wp  = wt.ExtractTranslation()
            wq  = wt.ExtractRotationQuat()
            return torch.tensor(
                [wp[0]-env_pos[0], wp[1]-env_pos[1], wp[2]-env_pos[2],
                 wq.real, wq.imaginary[0], wq.imaginary[1], wq.imaginary[2]],
                device=device,
            )

        e0 = self.scene.env_origins[0]
        hand_pose = _env_local_pose(e0, UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_link7")), self.device)
        lf_pose   = _env_local_pose(e0, UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_leftfinger")), self.device)
        rf_pose   = _env_local_pose(e0, UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_rightfinger")), self.device)
        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[:3] = (lf_pose[:3] + rf_pose[:3]) / 2.0
        finger_pose[3:] = lf_pose[3:]
        hi_rot, hi_pos = tf_inverse(hand_pose[3:], hand_pose[:3])
        grasp_rot, grasp_pos = tf_combine(hi_rot, hi_pos, finger_pose[3:], finger_pose[:3])
        grasp_pos += offset

        self.robot_local_grasp_pos = grasp_pos.repeat(self.num_envs, 1)
        self.robot_local_grasp_rot = grasp_rot.repeat(self.num_envs, 1)

        self.hand_link_idx         = self._robot.find_bodies("panda_link7")[0][0]
        self.left_finger_body_idx  = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_body_idx = self._robot.find_bodies("panda_rightfinger")[0][0]
        self.left_finger_joint_idx  = self._robot.find_joints("panda_finger_joint2")[0][0]
        self.right_finger_joint_idx = self._robot.find_joints("panda_finger_joint1")[0][0]

        # ------------------------------------------------------------------
        # State tensors
        # ------------------------------------------------------------------
        self.robot_grasp_rot  = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos  = torch.zeros((self.num_envs, 3), device=self.device)
        self.hand_quat        = torch.zeros((self.num_envs, 4), device=self.device)
        self.helper_variable  = torch.zeros((self.num_envs, 10), device=self.device)
        self.manipulability   = torch.zeros(self.num_envs, device=self.device)
        self.to_desired_rot   = torch.zeros((self.num_envs, 4), device=self.device)
        # Desired grasp orientation (same magic quaternion as TestPickItUp)
        self.q_rel = torch.tensor([0.0014, 0.9270, 0.3749, 0.0036], device=self.device).repeat(self.num_envs, 1)
        self.past_relative_dist   = torch.ones((self.num_envs, self.history_len), device=self.device)
        self.grasped              = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.stage                = torch.zeros((self.num_envs, self.num_stages), device=self.device, dtype=torch.bool)
        self.prev_actions         = torch.zeros((self.num_envs, cfg.action_space), device=self.device)
        self.high_enough          = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.small_rotation       = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.low_enough           = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.inside_site          = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.grasped_and_lifted   = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.high_enough_for_basket = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Tracked drawer position (world frame)
        self.drawer_interior_pos_w = torch.zeros((self.num_envs, 3), device=self.device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self.update_rate = 0

        if self.debug_vis:
            self.visualizer = self.define_markers()

    # ------------------------------------------------------------------
    # Scene setup (no demo data, no libero objects)
    # ------------------------------------------------------------------

    def _setup_scene(self):
        """Spawn robot, basket, cabinet with top drawer held open."""

        # --- Robot ---
        robot_cfg = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False, max_depenetration_velocity=5.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=12,
                    solver_velocity_iteration_count=4,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "panda_joint1": 1.157, "panda_joint2": -1.066,
                    "panda_joint3": -0.155, "panda_joint4": -2.239,
                    "panda_joint5": -1.841, "panda_joint6": 1.003,
                    "panda_joint7": 0.469,
                    "panda_finger_joint.*": 0.035,
                },
                pos=(1.0, 0.0, 0.0),   # matches franka_cabinet_env.py: 1 m from cabinet at (0,0,0.4)
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
            actuators={
                "panda_shoulder": ImplicitActuatorCfg(
                    joint_names_expr=["panda_joint[1-4]"],
                    effort_limit_sim=200.0, stiffness=600.0, damping=80.0,
                ),
                "panda_forearm": ImplicitActuatorCfg(
                    joint_names_expr=["panda_joint[5-7]"],
                    effort_limit_sim=120.0, stiffness=500.0, damping=70.0,
                ),
                "panda_hand": ImplicitActuatorCfg(
                    joint_names_expr=["panda_finger_joint.*"],
                    effort_limit_sim=200.0, stiffness=2e3, damping=1e2,
                ),
            },
        )
        self._robot = Articulation(robot_cfg)
        self.scene.articulations["robot"] = self._robot

        # --- Cream cheese (target object to pick and place in drawer) ---
        cream_cheese_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/cream_cheese",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[1.0, 0.4, 0.0],
                rot=[0.7071, 0.7071, 0.0, 0.0],
            ),
            debug_vis=True,
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{self.root}/libero/COMMON/stable_hope_objects/cream_cheese/usd/cream_cheese.usd",
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
            ),
        )
        self._cream_cheese = RigidObject(cream_cheese_cfg)
        self.scene.rigid_objects["cream_cheese"] = self._cream_cheese

        # --- Sektion cabinet – drawer is passive (stiffness=0); only damping ---
        cabinet_cfg = ArticulationCfg(
            prim_path="/World/envs/env_.*/Cabinet",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
                activate_contact_sensors=False,
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=self.cfg.cabinet_pos,
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos={
                    "door_left_joint":    0.0,
                    "door_right_joint":   0.0,
                    "drawer_bottom_joint": self.cfg.drawer_open_pos,
                    "drawer_top_joint":   0.0,
                },
            ),
            actuators={
                # Zero stiffness → drawer free to move; damping for stability.
                "drawers": ImplicitActuatorCfg(
                    joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                    effort_limit_sim=87.0,
                    stiffness=0.0,
                    damping=10.0,
                ),
                "doors": ImplicitActuatorCfg(
                    joint_names_expr=["door_left_joint", "door_right_joint"],
                    effort_limit_sim=87.0,
                    stiffness=10.0,
                    damping=2.5,
                ),
            },
        )
        self._cabinet = Articulation(cabinet_cfg)
        self.scene.articulations["cabinet"] = self._cabinet

        # Alias required by inherited helpers
        self.rigid_objects      = {"cream_cheese": self._cream_cheese}
        self.object_names       = ["cream_cheese"]
        self.object_indices     = {"cream_cheese": 0}
        self.target_object      = self._cream_cheese

        # --- Terrain ---
        self.cfg.terrain.num_envs    = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Camera for VLM feedback; MUST be defined before scene.clone_environments.
        if getattr(self.cfg, "camera_sensor_record", False):
            self.define_camera()

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # --- Finger contact sensors (for grasp detection) ---
        filter_path = f"/World/envs/env_.*/cream_cheese/object"
        self._left_contact_sensors = ContactSensor(ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/panda_leftfinger",
            update_period=0.0,
            history_length=2 * self.history_len,
            track_air_time=True,
            filter_prim_paths_expr=[filter_path],
            track_contact_points=True,
            max_contact_data_count_per_prim=4,
        ))
        self._right_contact_sensors = ContactSensor(ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/panda_rightfinger",
            update_period=0.0,
            history_length=2 * self.history_len,
            track_air_time=True,
            filter_prim_paths_expr=[filter_path],
            track_contact_points=True,
            max_contact_data_count_per_prim=4,
        ))
        self.scene.sensors["left_contact_sensor"]  = self._left_contact_sensors
        self.scene.sensors["right_contact_sensor"] = self._right_contact_sensors
        self._left_contact_sensors.set_debug_vis(False)
        self._right_contact_sensors.set_debug_vis(False)

        # Lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Intermediate values
    # ------------------------------------------------------------------

    def _compute_intermediate_values(self, env_ids=None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        self._compute_robot_intermediate_values(env_ids)
        self._compute_target_object_corners(env_ids)   # basket corners
        self._compute_drawer_interior(env_ids)          # drawer world position

    def _compute_drawer_interior(self, env_ids=None):
        """Track drawer interior centre in world frame."""
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        body_pos_w  = self._cabinet.data.body_pos_w[env_ids, self.drawer_body_idx]   # (n,3)
        body_quat_w = self._cabinet.data.body_quat_w[env_ids, self.drawer_body_idx]  # (n,4)
        n = len(env_ids)
        offset = self.drawer_interior_local.unsqueeze(0).expand(n, -1)               # (n,3)
        self.drawer_interior_pos_w[env_ids] = body_pos_w + quat_apply(body_quat_w, offset)

    # ------------------------------------------------------------------
    # Dones
    # ------------------------------------------------------------------

    def _get_dones(self):
        self._compute_intermediate_values()
        self.to_desired_rot = quat_mul(quat_conjugate(self.robot_grasp_rot), self.q_rel)
        self.manipulability = self._compute_manipulability()
        self.grasped = self._grasp_detection()
        self._update_past_relative_dist()

        basket_pos = self.target_object.data.root_pos_w   # (N,3)
        drawer_pos = self.drawer_interior_pos_w            # (N,3)

        # Inside drawer volume (axis-aligned box around interior centre)
        diff = basket_pos - drawer_pos
        inside_x = diff[:, 0].abs() < self.drawer_half_x
        inside_y = diff[:, 1].abs() < self.drawer_half_y
        inside_z = diff[:, 2].abs() < (self.drawer_half_z + 0.04)   # extra tolerance in Z

        self.inside_site          = inside_x & inside_y & inside_z
        self.low_enough           = basket_pos[:, 2] < (drawer_pos[:, 2] + self.drawer_half_z)
        self.high_enough_for_basket = basket_pos[:, 2] > (drawer_pos[:, 2] - self.drawer_half_z - 0.04)
        # "high enough" = basket lifted above the drawer opening plane (for transport)
        self.high_enough = basket_pos[:, 2] > drawer_pos[:, 2]

        self.grasped_and_lifted |= self.grasped & self.high_enough
        terminated = self.inside_site & self.grasped_and_lifted
        truncated  = self.episode_length_buf >= self.max_episode_length - 1

        self.stage = self._current_stage_detection()
        if self.debug_vis:
            self.debug_vis_mine()
        return terminated, truncated

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        self.corners_target_obj_to_hand_pos = (
            self.corners_target_obj - self.robot_grasp_pos.unsqueeze(1)
        ).reshape(self.num_envs, -1)
        self.target_to_hand_pos = self.target_object.data.root_pos_w - self.robot_grasp_pos  # (N,3)

        tcp_vel = (
            self._robot.data.body_link_lin_vel_w[:, self.left_finger_body_idx]
            + self._robot.data.body_link_lin_vel_w[:, self.right_finger_body_idx]
        ) / 2.0
        target_to_hand_vel = self.target_object.data.root_lin_vel_w - tcp_vel

        # Drawer is essentially static – vel ≈ 0
        drawer_to_target_pos = self.drawer_interior_pos_w - self.target_object.data.root_pos_w
        drawer_to_target_vel = torch.zeros_like(drawer_to_target_pos)

        obs = torch.cat(
            (
                dof_pos_scaled,                                      # 9
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,  # 9
                self.corners_target_obj_to_hand_pos,                 # 24
                self.target_to_hand_pos,                             # 3
                self.to_desired_rot,                                 # 4
                drawer_to_target_pos,                                # 3
                target_to_hand_vel,                                  # 3
                drawer_to_target_vel,                                # 3
                self.stage.float(),                                  # 5
                self.prev_actions - self.actions,                    # 8
            ),
            dim=-1,
        )  # total = 71
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset without demo data: random robot joints + random cream_cheese Y position."""
        self.update_rate += 1
        DirectRLEnv._reset_idx(self, env_ids)   # resets scene (robot/cream_cheese/cabinet to init_state)
        n = len(env_ids)

        # --- Robot: default joint pos + Gaussian noise on all 7 arm joints ---
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        if self.cfg.arm_joint_noise > 0:
            joint_pos[:, :7] += torch.randn(n, 7, device=self.device) * self.cfg.arm_joint_noise
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # --- Cream cheese: zero-centred noise on top of init_state from _setup_scene ---
        xn, yn = self.cfg.cream_cheese_pos_noise
        cc_local = self._cream_cheese.data.default_root_state[env_ids, :3].clone()
        cc_local[:, 0] += (torch.rand(n, device=self.device) * 2 - 1) * xn
        cc_local[:, 1] += (torch.rand(n, device=self.device) * 2 - 1) * yn
        cc_world_pos = cc_local + self.scene.env_origins[env_ids]
        cc_quat = torch.zeros(n, 4, device=self.device)
        cc_quat[:, 0] = 0.7071   # 90° around X – matches spawn orientation
        cc_quat[:, 1] = 0.7071
        self._cream_cheese.write_root_pose_to_sim(
            torch.cat([cc_world_pos, cc_quat], dim=-1), env_ids,
        )
        self._cream_cheese.write_root_velocity_to_sim(
            torch.zeros(n, 6, device=self.device), env_ids,
        )

        # Cabinet: reset joint positions to init_state; no position target set so
        # the drawer is free to move under contact forces.
        cab_pos = self._cabinet.data.default_joint_pos[env_ids].clone()
        cab_vel = torch.zeros_like(cab_pos)
        self._cabinet.write_joint_state_to_sim(cab_pos, cab_vel, env_ids=env_ids)

        self._compute_intermediate_values(env_ids=env_ids)
        self.to_desired_rot[env_ids] = quat_mul(
            quat_conjugate(self.robot_grasp_rot[env_ids]), self.q_rel[env_ids]
        )
        self.helper_variable[env_ids].zero_()
        self.grasped[env_ids].fill_(False)
        self.grasped_and_lifted[env_ids].fill_(False)
        self.stage[env_ids] = 0
        self.stage[env_ids, 0] = 1
        self.past_relative_dist[env_ids].zero_()
        self.prev_actions[env_ids].zero_()

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        rewards, info = self._get_rewards_place_in_drawer()
        self.extras["log"] = info
        return rewards

    def _get_rewards_place_in_drawer(self):
        """Stage-conditioned reward: approach → grasp → lift → transport → drop."""
        eps   = 1e-6
        dtype = self.target_object.data.root_pos_w.dtype

        # Sanitised state
        eef_pos    = torch.nan_to_num(self.robot_grasp_pos)
        obj_pos    = torch.nan_to_num(self.target_object.data.root_pos_w)
        drawer_pos = torch.nan_to_num(self.drawer_interior_pos_w)
        actions    = torch.nan_to_num(self.actions)
        prev_act   = torch.nan_to_num(self.prev_actions)
        gripper    = actions[:, -1]

        grasped_f       = self.grasped.float()
        high_enough_f   = self.high_enough.float()
        s               = self.stage.float()   # [N, 5]

        obj_z  = obj_pos[:, 2]
        obj_xy = obj_pos[:, :2]
        eef_xy = eef_pos[:, :2]
        drw_xy = drawer_pos[:, :2]
        drw_z  = drawer_pos[:, 2]

        # Gripper intent
        close_score = torch.sigmoid((-gripper / 0.20).clamp(-20, 20))
        open_score  = torch.sigmoid(( gripper / 0.20).clamp(-20, 20))

        # Distances
        eef_to_obj_dist   = torch.linalg.norm(eef_pos - obj_pos, dim=-1)
        eef_to_obj_xy     = torch.linalg.norm(eef_xy - obj_xy, dim=-1)
        obj_to_drw_dist   = torch.linalg.norm(obj_pos - drawer_pos, dim=-1)
        obj_to_drw_xy     = torch.linalg.norm(obj_xy - drw_xy, dim=-1)

        # -- Stage 0: approach basket from above --
        approach_xy = torch.exp(-(eef_to_obj_xy / 0.12).clamp(0, 50))
        approach_z  = torch.exp(-(torch.abs(eef_pos[:, 2] - (obj_z + 0.08)) / 0.08).clamp(0, 50))
        approach    = 0.6 * approach_xy + 0.4 * approach_z

        # -- Stage 1: grasp basket --
        grasp_xy    = torch.exp(-(eef_to_obj_xy / 0.05).clamp(0, 50))
        grasp_z     = torch.exp(-(torch.abs(eef_pos[:, 2] - obj_z) / 0.04).clamp(0, 50))
        grasp_prog  = 0.50 * grasp_xy + 0.30 * grasp_z + 0.20 * close_score

        # -- Stage 2: lift above drawer opening --
        lift_remaining   = torch.relu(drw_z + 0.05 - obj_z)
        lift_score       = grasped_f * torch.exp(-(lift_remaining / 0.12).clamp(0, 50))

        # -- Stage 3: transport over drawer (XY alignment) --
        over_drw_broad = torch.exp(-(obj_to_drw_xy / 0.30).clamp(0, 50))
        over_drw_fine  = torch.exp(-(obj_to_drw_xy / 0.12).clamp(0, 50))
        transport      = grasped_f * high_enough_f * (0.5 * over_drw_broad + 0.5 * over_drw_fine)

        # -- Stage 4: lower into drawer & release --
        in_drw_xy  = torch.exp(-(obj_to_drw_xy / 0.10).clamp(0, 50))
        in_drw_z   = torch.exp(-(torch.abs(obj_z - drw_z) / 0.06).clamp(0, 50))
        drop_score = in_drw_xy * (
            0.35 * in_drw_z
            + 0.35 * open_score
            + 0.30 * torch.sigmoid(((drw_z - obj_z) / 0.06).clamp(-20, 20))
        )

        # Stage-conditioned progress reward
        stage_bonus = 0.012*s[:,1] + 0.030*s[:,2] + 0.070*s[:,3] + 0.120*s[:,4]
        stage_rew   = (
            s[:,0] * 0.06 * approach
            + s[:,1] * 0.18 * grasp_prog
            + s[:,2] * 0.20 * lift_score
            + s[:,3] * 0.38 * transport
            + s[:,4] * 0.40 * drop_score
        )

        # Milestone bonuses
        grasp_milestone   = 0.04 * grasped_f * (s[:,1] + s[:,2])
        lift_milestone    = 0.04 * grasped_f * high_enough_f * s[:,3]
        inside_milestone  = 0.12 * self.inside_site.float() * grasped_f * high_enough_f

        # Regularisation
        joint_pos = torch.nan_to_num(self._robot.data.joint_pos)
        joint_vel = torch.nan_to_num(self._robot.data.joint_vel)
        dof_lo    = self.robot_dof_lower_limits
        dof_hi    = self.robot_dof_upper_limits
        dof_rng   = (dof_hi - dof_lo).clamp(min=eps)
        dof_sc    = 2.0 * (joint_pos - dof_lo) / dof_rng - 1.0
        lim_pen   = torch.relu(dof_sc.abs() - 0.82).clamp(0, 1).mean(dim=-1)
        ar_pen    = torch.square(actions - prev_act).mean(dim=-1).clamp(0, 1)
        vel_pen   = (torch.square(joint_vel).mean(dim=-1) / 25.0).clamp(0, 1)
        reg_pen   = 0.04*lim_pen + 0.008*ar_pen + 0.003*vel_pen + 0.004   # + time penalty

        non_success = (
            stage_bonus + stage_rew + grasp_milestone + lift_milestone + inside_milestone - reg_pen
        )
        non_success = torch.nan_to_num(non_success).clamp(-1.0, 0.48)

        success = self.inside_site & self.grasped_and_lifted
        success_rew = torch.full((self.num_envs,), 5.0, device=self.device, dtype=dtype)
        reward = torch.where(success, success_rew, non_success).clamp(-5.0, 5.0)

        info = {
            "success":        success.float(),
            "inside_drawer":  self.inside_site.float(),
            "grasped":        grasped_f,
            "high_enough":    high_enough_f,
            "approach":       approach,
            "grasp_progress": grasp_prog,
            "lift_score":     lift_score,
            "transport":      transport,
            "drop_score":     drop_score,
            "stage_reward":   stage_rew,
            "reg_penalty":    reg_pen,
            "obj_to_drw_xy":  obj_to_drw_xy,
            "obj_to_drw_dist": obj_to_drw_dist,
        }
        return reward, info

    # ------------------------------------------------------------------
    # Camera + VLM feedback
    # ------------------------------------------------------------------

    def define_camera(self):
        """Wide-FOV pinhole camera framing robot, cabinet drawer, and cream_cheese.

        Overrides the inherited tight 24 mm / 224 px camera with a 14 mm / 336 px
        setup so the whole pick-and-place workspace fits in frame.
        """
        sim_utils.create_prim("/World/Origin_00", "Xform")
        camera_cfg = CameraCfg(
            prim_path="/World/Origin_.*/CameraSensor",
            update_period=0,
            height=336,
            width=336,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=14.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
        )
        self.camera = Camera(cfg=camera_cfg)

    def run_single_traj_and_get_vlm_feedback(self, policy_nn, policy, output_dir: str) -> str:
        """Roll out one episode in env 0, dump RGB frames, call the VLM, return its text.

        Overrides TestPickItUp.run_single_traj_and_get_vlm_feedback because this
        subclass does NOT load demo trajectories, so self.data is not available.
        Camera pose is fixed to a side-overhead view that contains the robot,
        the cabinet (with the pre-opened bottom drawer), and the cream_cheese
        spawn region.
        """
        import omni.replicator.core as rep

        if not hasattr(self, "camera"):
            raise RuntimeError(
                "Camera not initialised. Set cfg.camera_sensor_record = True before "
                "constructing the env so define_camera() runs inside _setup_scene."
            )

        os.makedirs(output_dir, exist_ok=True)
        camera = self.camera

        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._reset_idx(env_ids=env_ids)

        rep_writer = rep.BasicWriter(output_dir=output_dir, frame_padding=0)

        env_origin = self.scene.env_origins[0]
        # Workspace bounds (per _setup_scene init poses): robot at (1.0, 0, 0..1),
        # cabinet at (-0.2, 0, 0.4) with drawer pulled out in +X, cream_cheese
        # spawn at (~1.0, 0.4, 0). Side-overhead view frames all three.
        camera_positions = torch.tensor([[1.0, 1.4, 1.0]], device=self.device) + env_origin
        camera_targets   = torch.tensor([[0.3, 0.0, 0.45]], device=self.device) + env_origin
        camera.set_world_poses_from_view(camera_positions, camera_targets)

        camera_index = 0
        obs = self._get_observations()
        for _ in range(self.max_episode_length):
            actions = policy(obs)
            obs, _rew, terminated, time_outs, _extras = self.step(actions)
            dones = terminated | time_outs
            policy_nn.reset(dones)

            camera.update(dt=self.sim.get_physics_dt())

            single_cam_data = convert_dict_to_backend(
                {k: v[camera_index] for k, v in camera.data.output.items()},
                backend="numpy",
            )
            single_cam_info = camera.data.info[camera_index]

            rep_output = {"annotators": {}}
            for key, data, info in zip(
                single_cam_data.keys(), single_cam_data.values(), single_cam_info.values()
            ):
                if info is not None:
                    rep_output["annotators"][key] = {"render_product": {"data": data, **info}}
                else:
                    rep_output["annotators"][key] = {"render_product": {"data": data}}
            rep_output["trigger_outputs"] = {"on_time": camera.frame[camera_index]}
            rep_writer.write(rep_output)

            if dones[0]:
                print("[run_single_traj] env 0 finished episode")
                break

        return self.get_feedback_from_vlm(output_dir=output_dir)
