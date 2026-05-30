# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenTheMicrowave: Franka arm opens a microwave door by pulling the handle."""

from __future__ import annotations

import os

import numpy as np
import torch
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab_eureka.utils import eureka_root_dir, read_pkl


@configclass
class OpenTheMicrowaveCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.65
    decimation = 2
    action_space = 8
    # 9 dof_pos + 9 dof_vel + 3 hand_to_handle + 1 door_angle + 8 prev_actions-actions
    observation_space = 30
    state_space = 0
    seed = 42

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

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048, env_spacing=3.0, replicate_physics=True, clone_in_fabric=False,
    )

    randomize_init: bool = True

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
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
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
                effort_limit_sim=80.0,
                stiffness=1500.0,
                damping=200.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit_sim=80.0,
                stiffness=1200.0,
                damping=180.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit_sim=200.0,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

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

    action_scale = 7.5
    dof_velocity_scale = 0.1
    # microjoint range: -2.094 to 0 rad (from MJCF); success when door open past 70 deg
    door_success_threshold = -1.222

    camera_sensor_record: bool = False


class OpenTheMicrowave(DirectRLEnv):
    cfg: OpenTheMicrowaveCfg

    def __init__(self, cfg: OpenTheMicrowaveCfg, render_mode: str | None = None, **kwargs):

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

        seed = cfg.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

        self.discrete_action = True
        self.randomize_init = cfg.randomize_init
        self.camera_sensor_record = cfg.camera_sensor_record
        self.root = eureka_root_dir()
        self.path = (
            f"{self.root}/libero/trajs/libero90/"
            "libero_90_kitchen_scene7_open_the_microwave_traj_v2.pkl"
        )

        self.data = read_pkl(self.path)
        self.episodes = self.data["franka"]
        lengths = [len(ep["states"]) for ep in self.episodes]
        max_len = max(lengths)
        frequency_ratio = 3
        episode_length = (max_len - 1) * frequency_ratio + 1
        cfg.episode_length_s = episode_length * cfg.sim.dt * cfg.decimation

        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        grasp_offset = torch.tensor([-0.01, 0.0046, 0.0738], device=self.device)

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint1")[0]] = 0.1
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint2")[0]] = 0.1

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        stage = get_current_stage()
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
        robot_local_pose_pos += grasp_offset

        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1))  # 3
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))  # 4

        self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
        self.left_finger_body_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_body_idx = self._robot.find_bodies("panda_rightfinger")[0][0]

        # MjcfConverter drops the "micro" prefix: microdoorroot → doorroot, microjoint → first joint
        print(f"[OpenTheMicrowave] microwave joint_names: {self._microwave.joint_names}")
        print(f"[OpenTheMicrowave] microwave body_names: {self._microwave.body_names}")
        self.door_joint_idx = self._microwave.find_joints("microjoint")[0][0]
        # doorroot is the door body in USD (MJCF microdoorroot → doorroot after conversion)
        self.door_body_idx = self._microwave.find_bodies("microdoorroot")[0][0]

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.handle_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.hand_to_handle_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, cfg.action_space), device=self.device)

        self.debug_vis = True
        self.visualizer = self.define_markers()

    def _setup_scene(self):
        init_states = self.data["franka"][0]["init_state"]
        robot_data = init_states["franka"]
        robot_joint_pos = robot_data["dof_pos"]
        robot_joint_pos["panda_finger_joint2"] = robot_joint_pos["panda_finger_joint1"]
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
                    enabled_self_collisions=False,
                    solver_position_iteration_count=12,
                    solver_velocity_iteration_count=4,
                ),
            ),
            debug_vis=False,
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
            },
        )
        self._robot = Articulation(robot_cfg)
        self.scene.articulations["robot"] = self._robot

        microwave_data = init_states["microwave"]
        microwave_cfg = ArticulationCfg(
            prim_path="/World/envs/env_.*/Microwave",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{self.root}/libero/COMMON/articulated_objects/microwave/usd1/microwave.usd",
                activate_contact_sensors=False,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    fix_root_link=True,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=microwave_data["pos"],
                rot=microwave_data["rot"],
                joint_pos={"microjoint": 0.0},
            ),
            actuators={
                "door": ImplicitActuatorCfg(
                    joint_names_expr=["microjoint"],
                    effort_limit_sim=1000.0,
                    stiffness=0.0,
                    damping=1.0,
                ),
            },
        )
        self._microwave = Articulation(microwave_cfg)
        self.scene.articulations["microwave"] = self._microwave

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        if self.camera_sensor_record:
            self.define_camera()

    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone().clamp(-1.0, 1.0)

        arm_actions = actions[:, :-1]
        grip_action = actions[:, -1]

        arm_targets = self.robot_dof_targets[:, :-2] + (
            self.robot_dof_speed_scales[:-2] * self.dt * arm_actions * self.cfg.action_scale
        )
        self.robot_dof_targets[:, :-2] = torch.clamp(
            arm_targets,
            self.robot_dof_lower_limits[:-2],
            self.robot_dof_upper_limits[:-2],
        )
        open_mask = grip_action >= 0
        self.robot_dof_targets[open_mask, -2:] = self.robot_dof_upper_limits[-1]
        self.robot_dof_targets[~open_mask, -2:] = self.robot_dof_lower_limits[-1]

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        if self.debug_vis:
            self.debug_vis_mine()
        door_angle = self._microwave.data.joint_pos[:, self.door_joint_idx]
        terminated = door_angle < self.cfg.door_success_threshold
        print(terminated.float().mean())
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        rewards, individual_rewards = self._get_rewards_test()
        self.extras["log"] = individual_rewards
        return rewards

    def _get_rewards_test(self):
        """Placeholder replaced by Eureka LLM-generated reward."""
        rewards = torch.zeros(self.num_envs, device=self.device)
        individual_rewards: dict = {}
        return rewards, individual_rewards

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        rand_episode_idx = torch.randint(
            low=0, high=min(50, len(self.episodes)), size=(1,), device=self.device
        ).item()
        rand_int = torch.randint(low=0, high=21, size=(1,), device=self.device).item()
        rand_int = min(rand_int, len(self.episodes[rand_episode_idx]["states"]) - 1)
        init_states = self.episodes[rand_episode_idx]["states"][rand_int]

        robot_data = init_states["franka"]
        robot_joint_pos = robot_data["dof_pos"]
        robot_joint_pos["panda_finger_joint2"] = robot_joint_pos["panda_finger_joint1"]
        robot_joint_pos = {k: v.item() for k, v in robot_joint_pos.items()}
        joint_values = torch.tensor(list(robot_joint_pos.values()), device=self.device)
        self._robot.data.default_joint_pos[env_ids] = joint_values

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)

        if self.randomize_init:
            base_quat = torch.tensor(
                robot_data["rot"], dtype=torch.float32, device=self.device
            ).unsqueeze(0).expand(len(env_ids), -1)
            demo_base_pos_local = torch.tensor(
                robot_data["pos"], dtype=torch.float32, device=self.device
            ).unsqueeze(0).expand(len(env_ids), -1)
            self._robot.write_root_pose_to_sim(
                torch.cat([demo_base_pos_local + self.scene.env_origins[env_ids], base_quat], dim=-1),
                env_ids=env_ids,
            )
            noise = torch.randn(len(env_ids), 7, device=self.device) * 0.05
            joint_pos[:, :7] += noise
            joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Root is fixed (fix_root_link=True) — no pose reset needed, same as stove
        # Reset door to closed
        microwave_joint_pos = self._microwave.data.default_joint_pos[env_ids].clone()
        microwave_joint_pos[:, self.door_joint_idx] = 0.0
        microwave_joint_vel = torch.zeros_like(microwave_joint_pos)
        self._microwave.write_joint_state_to_sim(microwave_joint_pos, microwave_joint_vel, env_ids=env_ids)

        self.prev_actions[env_ids].zero_()
        self._compute_intermediate_values(env_ids=env_ids)

    def _get_observations(self) -> dict:
        """
        Task: rotate microjoint from 0 to < -1.0 rad to open the microwave.

        Key attributes for reward shaping:
        - self.robot_grasp_pos: TCP world position [num_envs, 3]
        - self.hand_to_handle_pos: vector from TCP to door handle center [num_envs, 3]
        - self.handle_pos_w: handle world position [num_envs, 3]
        - self._microwave.data.joint_pos[:, self.door_joint_idx]: door angle [num_envs]
        - self.cfg.door_success_threshold: -1.222 rad (70 degrees)
        """
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        door_angle = self._microwave.data.joint_pos[:, self.door_joint_idx].unsqueeze(-1)
        obs = torch.cat(
            (
                dof_pos_scaled,                                              # 9
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,   # 9
                self.hand_to_handle_pos,                                     # 3
                door_angle,                                                  # 1
                self.prev_actions - self.actions,                            # 8
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    def _compute_intermediate_values(self, env_ids=None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        hand_pos = self._robot.data.body_pos_w[env_ids, self.hand_link_idx]
        hand_rot = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]
        self.robot_grasp_rot[env_ids], self.robot_grasp_pos[env_ids] = tf_combine(
            hand_rot, hand_pos, self.robot_local_grasp_rot[env_ids], self.robot_local_grasp_pos[env_ids]
        )

        self.handle_pos_w[env_ids] = self._microwave.data.body_pos_w[env_ids, self.door_body_idx]
        self.hand_to_handle_pos[env_ids] = self.handle_pos_w[env_ids] - self.robot_grasp_pos[env_ids]

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
        # Show robot TCP and microwave door handle, following test_pick_it_up.py pattern
        translations = torch.cat([self.robot_grasp_pos, self.handle_pos_w], dim=0)
        orientations = torch.cat([
            self.robot_grasp_rot,
            self._microwave.data.body_quat_w[:, self.door_body_idx],
        ], dim=0)
        self.visualizer.visualize(translations=translations, orientations=orientations)

    # --- Stubs: not needed for this task ---

    def run_replay(self, log_dir: str, render: bool = False, detect_grasp: bool = False):
        import numpy as np
        from torch.utils.tensorboard import SummaryWriter

        libero_dt = 1 / 20
        num_episodes = len(self.episodes)
        if num_episodes > self.num_envs:
            raise ValueError(
                f"num_episodes ({num_episodes}) exceeds self.num_envs ({self.num_envs})"
            )

        env_ids = torch.arange(num_episodes, device=self.device, dtype=torch.long)
        self._reset_idx(env_ids)

        writer = SummaryWriter(log_dir)

        lengths = [len(ep["states"]) for ep in self.episodes]
        max_len = max(lengths)
        frequency_ratio = libero_dt / (self.cfg.decimation * self.cfg.sim.dt)
        rest_episode_length = int((max_len - 1) * frequency_ratio + 1)

        robot_dof_dim = len(self.episodes[0]["states"][0]["franka"]["dof_pos"])
        robot_dof = torch.zeros(
            (self.num_envs, rest_episode_length, robot_dof_dim),
            device=self.device, dtype=torch.float32,
        )

        def lerp(a, b, u):
            return (1.0 - u) * a + u * b

        def slerp(q0, q1, u, eps=1e-8):
            q0 = np.asarray(q0, dtype=np.float64)
            q1 = np.asarray(q1, dtype=np.float64)
            q0 = q0 / (np.linalg.norm(q0) + eps)
            q1 = q1 / (np.linalg.norm(q1) + eps)
            dot = np.dot(q0, q1)
            if dot < 0.0:
                q1, dot = -q1, -dot
            if dot > 0.9995:
                q = lerp(q0, q1, u)
                return (q / (np.linalg.norm(q) + eps)).astype(np.float64)
            theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
            sin_theta_0 = np.sin(theta_0)
            s0 = np.sin(theta_0 * (1 - u)) / (sin_theta_0 + eps)
            s1 = np.sin(theta_0 * u) / (sin_theta_0 + eps)
            return (s0 * q0 + s1 * q1).astype(np.float64)

        def interp_state(s0, s1, u):
            out = {"franka": {"dof_pos": {}}}
            for k in s0["franka"]["dof_pos"].keys():
                v0 = float(np.array(s0["franka"]["dof_pos"][k]).reshape(-1)[0])
                v1 = float(np.array(s1["franka"]["dof_pos"][k]).reshape(-1)[0])
                out["franka"]["dof_pos"][k] = np.array([lerp(v0, v1, u)], dtype=np.float64)
            return out

        def insert_two_between(states):
            new_states = []
            for t in range(len(states) - 1):
                s0, s1 = states[t], states[t + 1]
                new_states.append(s0)
                new_states.append(interp_state(s0, s1, 1 / 3))
                new_states.append(interp_state(s0, s1, 2 / 3))
            new_states.append(states[-1])
            return new_states

        with torch.inference_mode():
            for env_idx, ep in enumerate(self.episodes):
                new_ep = insert_two_between(ep["states"])
                for t, state in enumerate(new_ep):
                    dof_vals = [v[0] for v in state["franka"]["dof_pos"].values()]
                    dof_vals[-1] = -dof_vals[-1]
                    robot_dof[env_idx, t] = torch.tensor(dof_vals)

            for t in range(rest_episode_length):
                joint_pos = robot_dof[:, t, :]
                self._robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))

                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                if render and t % 2 == 0:
                    self.sim.render()
                self.scene.update(dt=self.physics_dt)

                self._compute_intermediate_values()
                diff = self.hand_to_handle_pos[0]
                dist = torch.norm(self.hand_to_handle_pos, dim=-1)
                print(f"t={t:4d} | gripper-to-handle dx={diff[0].item():.4f} dy={diff[1].item():.4f} dz={diff[2].item():.4f} |d|={dist[0].item():.4f} m")
                writer.add_scalar("Replay/gripper_to_handle_dist_mean", dist.mean().item(), t)

        writer.close()

    def define_camera(self):
        from isaaclab.sensors.camera import Camera, CameraCfg
        camera_cfg = CameraCfg(
            prim_path="/World/Origin_.*/CameraSensor",
            update_period=0,
            height=224,
            width=224,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
        )
        self.camera = Camera(cfg=camera_cfg)

    def run_single_traj_and_get_vlm_feedback(self, policy_nn, policy, output_dir: str) -> str:
        import omni.replicator.core as rep
        from isaaclab.utils.math import convert_dict_to_backend

        os.makedirs(output_dir, exist_ok=True)
        camera = self.camera
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._reset_idx(env_ids=env_ids)
        rep_writer = rep.BasicWriter(output_dir=output_dir, frame_padding=0)

        env_origin = self.scene.env_origins[0]
        camera_positions = torch.tensor([[0.8, 0.6, 1.0]], device=self.device) + env_origin
        camera_targets = self.handle_pos_w[0:1].clone()
        camera.set_world_poses_from_view(camera_positions, camera_targets)

        camera_index = 0
        obs = self._get_observations()
        for i in range(self.max_episode_length):
            actions = policy(obs)
            obs, rewards, terminated, time_outs, extras = self.step(actions)
            dones = terminated | time_outs
            policy_nn.reset(dones)
            camera.update(dt=self.sim.get_physics_dt())
            single_cam_data = convert_dict_to_backend(
                {k: v[camera_index] for k, v in camera.data.output.items()}, backend="numpy"
            )
            single_cam_info = camera.data.info[camera_index]
            rep_output = {"annotators": {}}
            for key, data, info in zip(
                single_cam_data.keys(), single_cam_data.values(), single_cam_info.values()
            ):
                rep_output["annotators"][key] = {
                    "render_product": {"data": data, **(info if info is not None else {})}
                }
            rep_output["trigger_outputs"] = {"on_time": camera.frame[camera_index]}
            rep_writer.write(rep_output)
            if dones[0]:
                break
        return self.get_feedback_from_vlm(output_dir=output_dir)

    def get_feedback_from_vlm(self, output_dir: str = "", **kwargs) -> str:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "prithivMLmods/DeepCaption-VLA-7B", torch_dtype="auto", device_map="cpu"
        )
        processor = AutoProcessor.from_pretrained("prithivMLmods/DeepCaption-VLA-7B")

        all_files = sorted(
            [f for f in os.listdir(output_dir) if f.endswith(".png") and f.startswith("rgb_")],
            key=lambda f: int(f.split("_")[1]),
        )
        num_samples = 30
        if len(all_files) > num_samples:
            indices = np.linspace(0, len(all_files) - 1, num_samples, dtype=int)
            all_files = [all_files[i] for i in indices]

        images_content = [{"type": "image", "image": os.path.join(output_dir, f)} for f in all_files]
        messages = [
            {
                "role": "user",
                "content": [
                    *images_content,
                    {
                        "type": "text",
                        "text": (
                            "Analyze this image sequence showing a robot arm near a microwave.\n\n"
                            "Answer each question with ONLY: yes / no / unsure\n\n"
                            "Q1: Does the robot gripper make contact with or get very close to the microwave door handle?\n"
                            "A1: <yes/no/unsure>\n\n"
                            "Q2: Does the microwave door visibly open during the sequence?\n"
                            "A2: <yes/no/unsure>\n\n"
                            "Q3: In the final frame, does the microwave door appear to be open significantly?\n"
                            "A3: <yes/no/unsure>\n\n"
                            "Reasoning: <brief explanation>"
                        ),
                    },
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )
        generated_ids = model.generate(
            **inputs, max_new_tokens=300, repetition_penalty=1.3, no_repeat_ngram_size=3
        )
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, generated_ids)]
        return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def _grasp_detection(self, env_ids=None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        return torch.zeros(len(env_ids), device=self.device, dtype=torch.bool)

    def _current_stage_detection(self, env_ids=None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        return torch.zeros((len(env_ids), 1), device=self.device)
