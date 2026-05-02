# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TurnOnTheStove: Franka arm turns on a stove by rotating the button knob."""

from __future__ import annotations

import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab_eureka.utils import eureka_root_dir, read_pkl


@configclass
class TurnOnTheStoveCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.65
    decimation = 2
    action_space = 8
    # 9 dof_pos + 9 dof_vel + 3 hand_to_button + 1 button_angle + 8 prev_actions-actions
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
    # button_joint range: 0 to 2.1 rad (from URDF); success at 1.5 rad
    button_success_threshold = 1.5


class TurnOnTheStove(DirectRLEnv):
    cfg: TurnOnTheStoveCfg

    def __init__(self, cfg: TurnOnTheStoveCfg, render_mode: str | None = None, **kwargs):
        seed = cfg.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

        self.discrete_action = True
        self.randomize_init = cfg.randomize_init
        self.root = eureka_root_dir()
        self.path = (
            f"{self.root}/libero/trajs/libero90/"
            "libero_90_kitchen_scene3_turn_on_the_stove_traj_v2.pkl"
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

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint1")[0]] = 0.1
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint2")[0]] = 0.1

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        self.left_finger_body_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_body_idx = self._robot.find_bodies("panda_rightfinger")[0][0]
        self.button_joint_idx = self._stove.find_joints("button_joint")[0][0]
        # button link for world-frame position tracking
        # pdb.set_trace()
        self.button_body_idx = self._stove.find_bodies("button")[0][0]

        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.button_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.hand_to_button_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, cfg.action_space), device=self.device)

        self.debug_vis = True
        self.visualizer = self.define_markers()

    def _setup_scene(self):
        init_states = self.data["franka"][0]["states"][0]
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

        stove_cfg = ArticulationCfg(
            prim_path="/World/envs/env_.*/Stove",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{self.root}/libero/COMMON/articulated_objects/flat_stove/usd/flat_stove_urdf.usd",
                activate_contact_sensors=False,
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.4),
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos={"button_joint": 0.0},
            ),
            actuators={
                "knob": ImplicitActuatorCfg(
                    joint_names_expr=["button_joint"],
                    effort_limit_sim=1000.0,
                    stiffness=0.0,
                    damping=1.0,
                ),
            },
        )
        self._stove = Articulation(stove_cfg)
        self.scene.articulations["stove"] = self._stove

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

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
        button_angle = self._stove.data.joint_pos[:, self.button_joint_idx]
        terminated = button_angle > self.cfg.button_success_threshold
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

        # Reset stove button to 0; root is fixed so no pose reset needed
        stove_joint_pos = self._stove.data.default_joint_pos[env_ids].clone()
        stove_joint_pos[:, self.button_joint_idx] = 0.0
        stove_joint_vel = torch.zeros_like(stove_joint_pos)
        self._stove.write_joint_state_to_sim(stove_joint_pos, stove_joint_vel, env_ids=env_ids)

        self.prev_actions[env_ids].zero_()
        self._compute_intermediate_values(env_ids=env_ids)

    def _get_observations(self) -> dict:
        """
        Task: rotate button_joint from 0 to > 1.5 rad to turn on the stove.

        Key attributes for reward shaping:
        - self.robot_grasp_pos: TCP world position [num_envs, 3]
        - self.hand_to_button_pos: vector from TCP to button center [num_envs, 3]
        - self.button_pos_w: button world position [num_envs, 3]
        - self._stove.data.joint_pos[:, self.button_joint_idx]: button angle [num_envs]
        - self.cfg.button_success_threshold: 1.5 rad
        """
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        button_angle = self._stove.data.joint_pos[:, self.button_joint_idx].unsqueeze(-1)
        obs = torch.cat(
            (
                dof_pos_scaled,                                              # 9
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,   # 9
                self.hand_to_button_pos,                                     # 3
                button_angle,                                                # 1
                self.prev_actions - self.actions,                            # 8
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    def _compute_intermediate_values(self, env_ids=None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        left_finger_pos = self._robot.data.body_pos_w[env_ids, self.left_finger_body_idx]
        right_finger_pos = self._robot.data.body_pos_w[env_ids, self.right_finger_body_idx]
        self.robot_grasp_pos[env_ids] = (left_finger_pos + right_finger_pos) / 2.0

        self.button_pos_w[env_ids] = self._stove.data.body_pos_w[env_ids, self.button_body_idx]
        self.hand_to_button_pos[env_ids] = self.button_pos_w[env_ids] - self.robot_grasp_pos[env_ids]

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
        translations = self.button_pos_w
        orientations = self._stove.data.body_quat_w[:, self.button_body_idx]
        self.visualizer.visualize(translations=translations, orientations=orientations)

    # --- Stubs: not needed for this task ---

    def run_replay(self, *args, **kwargs):
        pass

    def get_feedback_from_vlm(self, *args, **kwargs):
        return ""

    def _grasp_detection(self, env_ids=None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        return torch.zeros(len(env_ids), device=self.device, dtype=torch.bool)

    def _current_stage_detection(self, env_ids=None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        return torch.zeros((len(env_ids), 1), device=self.device)
