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
from isaaclab.utils import configclass, convert_dict_to_backend
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform, quat_inv, quat_mul, transform_points, quat_conjugate, quat_apply
from torch.utils.tensorboard import SummaryWriter
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sensors.contact_sensor.contact_sensor import ContactSensor
from isaaclab.sensors import ContactSensorCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

import pdb
import os
import numpy as np
from isaaclab_eureka.utils import eureka_root_dir, read_pkl

@configclass
class TestCollideAndPlaceCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.65  # 519 timesteps
    decimation = 2
    action_space = 8
    observation_space = 66  # 9+9+24+3+3+3+3+8+3+1
    state_space = 0
    seed=42
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
        num_envs=1024, env_spacing=3.0, replicate_physics=True, clone_in_fabric=False, # See if 2048 works
    )

    camera_sensor_record: bool = False
    start_in_air: bool = False
    randomize_init: bool = True
    randomize_rotation: bool = False  # if False, skip joint6/7 noise (no EEF rotation perturbation)


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
            }
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

    action_scale = 7.5 # 7.5 originally
    dof_velocity_scale = 0.1

    # reward scales
    dist_reward_scale = 1.5
    rot_reward_scale = 1.5
    open_reward_scale = 10.0
    action_penalty_scale = 0.05
    finger_reward_scale = 2.0

    collision_object_name: str = "tomato_sauce"
    collision_displacement_threshold: float = 0.05  # metres to count as knocked
    basket_drift_threshold: float = 0.03            # metres, basket considered untouched below this


class TestCollideAndPlace(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: TestCollideAndPlaceCfg

    def __init__(self, cfg: TestCollideAndPlaceCfg, render_mode: str | None = None, **kwargs):
        seed = cfg.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        self.discrete_action = True
        self.debug_vis = True
        # Only when debug_vis is true:
        self.show_robot_grasp=True
        self.show_target_object=False

        self.camera_sensor_record = cfg.camera_sensor_record

        self.log_mine = False
        self.start_in_air = cfg.start_in_air
        self.randomize_init = cfg.randomize_init
        self.root = eureka_root_dir()
        self.target_object_name = "alphabet_soup"
        self.target_site_name = "basket"
        self.collision_object_name = cfg.collision_object_name
        self.input_direction = 2 # z axis
        if self.input_direction != 2:
            raise NotImplementedError("Change the _get_dones method and other calculations for deciding the entry size")
        self.path = f"{self.root}/libero/trajs/libero90/libero_90_living_room_scene1_pick_up_the_{self.target_object_name}_and_put_it_in_the_basket_traj_v2.pkl"
        if self.log_mine:
            import os
            log_dir = os.path.join(self.root, "logs", "replay_test")
            self.summary_writer_mine = SummaryWriter(log_dir)
        start_idx_in_episode_dict = {
            "alphabet_soup":80,
            "cream_cheese":80,
            "ketchup":80, # 80
            "tomato_sauce":90,
        }
        if not self.start_in_air:
            self.start_idx_in_episode = 0
        else:
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

        # Real-time basket corners — same pattern as target object corners
        basket_corners_world0 = torch.tensor(
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
        basket_quat0_conj = quat_conjugate(self.target_site.data.root_quat_w[0])
        basket_inv_pos    = -quat_apply(basket_quat0_conj, self.target_site.data.root_pos_w[0])
        local_basket_corners0 = transform_points(
            points=basket_corners_world0,
            pos=basket_inv_pos,
            quat=basket_quat0_conj,
        ).squeeze(0)  # [8, 3]
        self.local_basket_corners_init = local_basket_corners0.unsqueeze(0).repeat(self.num_envs, 1, 1)  # [num_envs, 8, 3]
        self.basket_corners_world = torch.zeros_like(self.local_basket_corners_init)  # [num_envs, 8, 3]

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
        self.prev_actions = torch.zeros((self.num_envs, cfg.action_space), device=self.device)

        self.low_enough = torch.zeros((self.num_envs), device=self.device, dtype=bool)
        self.inside_site = torch.zeros((self.num_envs), device=self.device, dtype=bool)

        self.collision_object: RigidObject = self.rigid_objects[self.collision_object_name]
        self.collision_obj_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.collision_triggered = torch.zeros((self.num_envs,), device=self.device, dtype=bool)
        self.collision_object_init_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.basket_init_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # # Specific to in the air task, randomize the initialization
        self.update_rate = 0
        # this is probably from height 0.1
        self.base = torch.tensor([ 91,  66,  67, 110,  66,  84, 106,  81,  54,  83,  72,  63,  68,  63,
                    57,  58,  72,  84,  76,  72,  62,  63,  85,  70,  72,  60,  68,  70,
                    64,  69,  85,  89,  77,  71,  74,  69,  64,  65,  70, 101,  72,  80,
                    61,  58,  74,  72,  61,  84,  60,  67], device=self.device)
        # This is from height 0.055
        self.base_0055 = torch.tensor([ 55,  59,  60, 101,  59,  74,  52,  53,  49,  74,  62,  56,  61,  56,
                    49,  52,  63,  75,  65,  66,  45,  53,  56,  56,  66,  50,  57,  62,
                    53,  61,  45,  80,  67,  62,  67,  62,  55,  59,  61,  48,  66,  58,
                    54,  50,  68,  61,  53,  52,  50,  39], device=self.device)

        # Adding a visualizer
        if self.debug_vis:
            self.visualizer = self.define_markers()
            print("Debug visualizer initialized")

    def _setup_scene(self):
        init_states = self.data['franka'][0]["states"][self.start_idx_in_episode]
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
                    enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=4
                ),
            ),
            debug_vis=True,
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
                }
        )
        self._robot = Articulation(robot_cfg)
        self.scene.articulations["robot"] = self._robot

        #TODO: for the rest: consider if the task has articulation
        self.rigid_objects = {}
        keys = list(init_states.keys())
        keys.remove("franka")   # remove the one you don't want

        for k in keys:
            if k == self.target_object_name:
                print(f"target object {k} initial z height: {init_states[k]['pos'][2]}")
                activate_contact_sensors=True
                debug_vis=True
            else:
                activate_contact_sensors = False
                debug_vis=False
            cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/{k}",
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=init_states[k]["pos"],
                    rot=init_states[k]["rot"],
                ),
                debug_vis=debug_vis,
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

        if self.camera_sensor_record:
            self.define_camera() # Single camera, should not be added to the scene

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
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone().clamp(-1.0, 1.0)
        
        
        if self.discrete_action:
            arm_actions = actions[:, :-1]
            grip_action = actions[:, -1]

            # Arm: continuous
            arm_targets = self.robot_dof_targets[:, :-2] + (
                self.robot_dof_speed_scales[:-2] * self.dt * arm_actions * self.cfg.action_scale
            )
            self.robot_dof_targets[:, :-2] = torch.clamp(
                arm_targets,
                self.robot_dof_lower_limits[:-2],
                self.robot_dof_upper_limits[:-2]
            )

            # Gripper: binary
            open_mask = grip_action >= 0
            self.robot_dof_targets[open_mask, -2:] = self.robot_dof_upper_limits[-1]
            self.robot_dof_targets[~open_mask, -2:] = self.robot_dof_lower_limits[-1]
        else:
            actions = torch.cat([actions,actions[:,-1].unsqueeze(1)], dim=1) # duplicate the gripper motion
            targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * actions * self.cfg.action_scale
            self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)


    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        # condition for termination: for dropping the object in the basket
        obj_z = self.target_object.data.root_pos_w[:, 2]
        basket_z        = self.basket_corners_world[:, :, 2]      # [num_envs, 8]
        basket_bottom_z = basket_z.min(dim=1).values              # [num_envs]
        basket_top_z    = basket_z.max(dim=1).values              # [num_envs]
        self.low_enough  = obj_z < basket_top_z                   # object center below basket rim
        self.high_enough_for_basket = obj_z > basket_bottom_z     # object center above basket floor
        obj_xy = self.target_object.data.root_pos_w[:, :2]
        site_pos = self.target_site.data.root_pos_w[:, :2]
        dist2 = ((obj_xy - site_pos)**2).sum(dim=-1)
        self.inside_site = dist2 < self.target_site_radius**2
        # terminated = inside_site & low_enough
        # self.high_enough = object_lowest_point > site_height

        # Collision object must have been knocked > threshold from its spawn
        if not self.collision_triggered.all():
            collision_displacement = torch.norm(self.collision_obj_pos - self.collision_object_init_pos, dim=-1)
            self.collision_triggered |= collision_displacement > self.cfg.collision_displacement_threshold

        # Basket must not have drifted beyond threshold
        basket_displacement = torch.norm(self.target_site.data.root_pos_w - self.basket_init_pos, dim=-1)
        basket_untouched = basket_displacement < self.cfg.basket_drift_threshold

        terminated = self.collision_triggered & self.inside_site & self.low_enough & self.high_enough_for_basket & basket_untouched
        
        # print(f"self.inside_site{self.inside_site}")
        # print(f"self.high_enough_for_basket{self.high_enough_for_basket}")
        # print(f"self.low_enough{self.low_enough}")
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        # terminated_count = high_enough.sum().item()
        # truncated_count = truncated.sum().item()

        # ratio = terminated_count / (truncated_count + 1e-8)
        # print(ratio)

        if self.debug_vis:
            self.debug_vis_mine()

        if self.log_mine:
            for i in range(20):
                pass
                # target_object_current_pose = self.target_object.data.root_quat_w        # shape (N, 4)
                # target_object_desired_pose = object_default_state[:,3:7]         # shape (N, 4) 
                # # inverse of current
                # target_object_current_pose_inv = quat_conjugate(target_object_current_pose)
                # # relative rotation
                # q_error = quat_mul(target_object_desired_pose, target_object_current_pose_inv)
                # q_error = q_error / torch.norm(q_error, dim=-1, keepdim=True).clamp_min(1e-9)
                # q_error = torch.where(q_error[:, 0:1] < 0, -q_error, q_error)
                # angle_error = 2 * torch.acos(torch.clamp(q_error[:, 0], -1.0, 1.0))
                # self.summary_writer_mine.add_scalars(
                #     f"q_error/env_{i}",
                #     {
                #         "grasped": angle_error[i].item(),
                #     },
                #     self._sim_step_counter
                # )

        
        return terminated, truncated
    
    def _get_rewards(self) -> torch.Tensor:
            

        # rewards_pick, individual_rewards = self._get_rewards_test_pick()
        # rewards_pick /=10 # around 2
        # rewards_put, _ = self._get_rewards_test_put()   
        # rewards_put += 2
        # grasped_mask = self.grasped.bool()  # shape: (n,)
        # rewards = torch.where(grasped_mask, rewards_put, rewards_pick)
        # rewards = rewards_pick


        rewards, individual_rewards = self._get_rewards_test()


        if self.log_mine:
            for k,v in individual_rewards.items():
                for i in range(20):
                    self.summary_writer_mine.add_scalar(k, v[i].item(), self._sim_step_counter) 

        self.extras["log"] = individual_rewards
        return rewards


    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        self.update_rate += 1
        # if self.update_rate%2 == 0:
        # update default root states for random initialization
        rand_episode_idx = torch.randint(low=0, high=50, size=(1,), device=self.device).item()
        if self.start_in_air:
            start_idx_in_episode = self.base_0055[rand_episode_idx]
        else:
            start_idx_in_episode = 0
        rand_int = torch.randint(low=0, high=21, size=(1,), device=self.device).item()
        idx = start_idx_in_episode + rand_int
        # print(idx)
        # print(len(self.data['franka'][rand_episode_idx]["states"]))
        init_states = self.data['franka'][rand_episode_idx]["states"][idx]
        robot_data = init_states['franka'] # seems that joint pos for isaaclab is always positive
        robot_joint_pos = robot_data["dof_pos"]
        # print(f"robot_joint_pos:{robot_joint_pos}")
        robot_joint_pos['panda_finger_joint2'] = robot_joint_pos['panda_finger_joint1']
        # print(f"robot_joint_pos:{robot_joint_pos}")
        robot_joint_pos = {k: v.item() for k, v in robot_joint_pos.items()}
        joint_values = torch.tensor(list(robot_joint_pos.values()), device=self.device)
        self._robot.data.default_joint_pos[env_ids] = joint_values

        for object_name in self.object_names:
            object = self.rigid_objects[object_name]
            pos = torch.tensor(init_states[object_name]["pos"], device=self.device)
            rot = torch.tensor(init_states[object_name]["rot"], device=self.device)

            object.data.default_root_state[env_ids, 0:3] = pos
            object.data.default_root_state[env_ids, 3:7] = rot

        # robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids] # TODO: recover randomization and test
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        if self.randomize_init:
            base_quat = torch.tensor(
                robot_data["rot"], dtype=torch.float32, device=self.device
            ).unsqueeze(0).expand(len(env_ids), -1)
            demo_base_pos_local = torch.tensor(
                robot_data["pos"], dtype=torch.float32, device=self.device
            ).unsqueeze(0).expand(len(env_ids), -1)

            # Ensure robot base is at demo position (no XY noise)
            self._robot.write_root_pose_to_sim(
                torch.cat([demo_base_pos_local + self.scene.env_origins[env_ids], base_quat], dim=-1),
                env_ids=env_ids,
            )

            joint_pos = joint_pos.clone()

            if self.start_in_air:
                # ----------------------------------------------------------------
                # FK at demo joint state → capture demo grasp pose
                # ----------------------------------------------------------------
                self.scene.update(dt=self.physics_dt)
                hand_pos_d  = self._robot.data.body_pos_w[env_ids, self.hand_link_idx]
                hand_quat_d = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]
                grasp_rot_d, grasp_pos_d = tf_combine(
                    hand_quat_d, hand_pos_d,
                    self.robot_local_grasp_rot[env_ids], self.robot_local_grasp_pos[env_ids],
                )

                # Demo object pose in world frame
                obj_pos_d = (
                    torch.tensor(
                        init_states[self.target_object_name]["pos"], dtype=torch.float32, device=self.device
                    ).unsqueeze(0).expand(len(env_ids), -1)
                    + self.scene.env_origins[env_ids]
                )
                obj_rot_d = torch.tensor(
                    init_states[self.target_object_name]["rot"], dtype=torch.float32, device=self.device
                ).unsqueeze(0).expand(len(env_ids), -1)

                # Express object in grasp frame (rigid body offset, stays constant through randomization)
                grasp_rot_d_inv, grasp_pos_d_inv = tf_inverse(grasp_rot_d, grasp_pos_d)
                obj_rot_in_grasp, obj_pos_in_grasp = tf_combine(
                    grasp_rot_d_inv, grasp_pos_d_inv, obj_rot_d, obj_pos_d
                )

                # ----------------------------------------------------------------
                # Add arm joint noise; object will follow EEF
                # ----------------------------------------------------------------
                noise = torch.randn(len(env_ids), 7, device=self.device) * 0.1
                if not self.cfg.randomize_rotation:
                    noise[:, 5:7] = 0.0  # keep wrist joints fixed
                joint_pos[:, :7] += noise
                joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

                # FK at randomized joints → new grasp pose
                self._robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
                self.scene.update(dt=self.physics_dt)
                hand_pos_new  = self._robot.data.body_pos_w[env_ids, self.hand_link_idx]
                hand_quat_new = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]
                grasp_rot_new, grasp_pos_new = tf_combine(
                    hand_quat_new, hand_pos_new,
                    self.robot_local_grasp_rot[env_ids], self.robot_local_grasp_pos[env_ids],
                )

                # Move object to maintain grasp-relative pose
                obj_rot_new, obj_pos_new = tf_combine(
                    grasp_rot_new, grasp_pos_new, obj_rot_in_grasp, obj_pos_in_grasp
                )

                # Store in default_root_state (local frame); the object write loop adds env_origins
                self.target_object.data.default_root_state[env_ids, 0:3] = (
                    obj_pos_new - self.scene.env_origins[env_ids]
                )
                self.target_object.data.default_root_state[env_ids, 3:7] = obj_rot_new

            else:
                # ----------------------------------------------------------------
                # start_in_air=False: object on table, optionally add rotation noise
                # ----------------------------------------------------------------
                if self.cfg.randomize_rotation:
                    joint_pos[:, 5] += torch.randn(len(env_ids), device=self.device) * 0.15
                    joint_pos[:, 6] += torch.randn(len(env_ids), device=self.device) * 0.15
                joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

            self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
            self._robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)

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
        self.prev_actions[env_ids].zero_()
        collision_pos_local = torch.tensor(init_states[self.collision_object_name]["pos"], device=self.device)
        self.collision_object_init_pos[env_ids] = collision_pos_local.unsqueeze(0) + self.scene.env_origins[env_ids]
        basket_pos_local = torch.tensor(init_states[self.target_site_name]["pos"], device=self.device)
        self.basket_init_pos[env_ids] = basket_pos_local.unsqueeze(0) + self.scene.env_origins[env_ids]
        self.collision_triggered[env_ids] = False


    def _get_observations(self) -> dict:
        """
        Observation hints for Eureka reward design:

        self.rigid_objects: dict of all RigidObjects (target, site, collision obj, distractors)
        Action space: 8-dim; last dim > 0 = open gripper, < 0 = close gripper.
        self.target_site_corners_world: [2, 3] tensor, row 0 = min corner, row 1 = max corner (env-local).
        self.collision_object_name: name of the object to collide with ("tomato_sauce").
        self.collision_triggered: [num_envs] bool, True once collision object displaced > 5 cm.
        self.collision_object_init_pos: [num_envs, 3] world-frame initial position of collision object.
        self.inside_site, self.low_enough, self.high_enough_for_basket: termination sub-conditions.
        obs includes collision_obj_to_target [N,3] and collision_triggered_flag [N,1].
        """

        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        self.corners_target_obj_to_hand_pos =  (self.corners_target_obj - self.robot_grasp_pos.unsqueeze(1)).reshape(self.num_envs, -1) # distance from corners of target objects to robot, described in world coordinate.
        self.target_to_hand_pos = self.target_object.data.root_pos_w - self.robot_grasp_pos # [self.num_envs, 3]
        tcp_vel = (self._robot.data.body_link_lin_vel_w[:,self.left_finger_body_idx] + self._robot.data.body_link_lin_vel_w[:,self.right_finger_body_idx])/2
        target_to_hand_vel = self.target_object.data.root_lin_vel_w - tcp_vel

        self.site_to_target_pos = self.target_site.data.root_pos_w - self.target_object.data.root_pos_w
        site_to_target_vel = self.target_object.data.root_lin_vel_w - self.target_site.data.root_lin_vel_w

        # Vector from held object to collision target; tells agent where to steer for the knock
        collision_obj_to_target = self.collision_obj_pos - self.target_object.data.root_pos_w  # [N, 3]
        collision_triggered_flag = self.collision_triggered.float().unsqueeze(-1)          # [N, 1]

        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
                self.corners_target_obj_to_hand_pos, # this should be small
                self.target_to_hand_pos, # relative position from target object center to hand should be small
                self.site_to_target_pos, # relative position from target site to target object
                target_to_hand_vel,
                site_to_target_vel,
                self.prev_actions - self.actions, # add penalty on delta action
                collision_obj_to_target,           # [N, 3] vector from held obj to collision target
                collision_triggered_flag,          # [N, 1] binary: collision already happened
            ),
            dim=-1,
        )

        # if self.cfg.randomize_init:
        #     obs = obs + torch.randn_like(obs) * 0.002

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



    def run_replay(self, log_dir:str, render:bool=False, detect_grasp:bool=False):
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
                    _ = self._get_observations()
                    # for i in range(50):
                    #     writer.add_scalar("Grasp/"+str(i), self.grasped[i], t) 
                    diff = self.target_object.data.body_pos_w.squeeze(1) - self.robot_grasp_pos
                    # print(diff[0]) # play the first episode to check the constant object-hand distance
                    diff_norm = torch.norm(diff, dim=-1)  # shape: (n,)
                    # 0.09 ketchup/ 0.03 cream_cheese 0.1 alphabet_soup

                    if t > 20:
                        obj_z = self.target_object.data.root_pos_w[env_ids, 2]  # (num_envs,)
                        in_air = obj_z > 0.055                             # (num_envs,) bool
                        new_air_envs = torch.nonzero(in_air & (in_the_air_matrix == -1), as_tuple=False).squeeze(-1)
                        in_the_air_matrix[new_air_envs] = t


                    # print("logging")
                    # for k in range(50):
                    #     writer.add_scalars("Position_error/"+str(k), 
                    #                     {"x": diff[k,0], 
                    #                     "y": diff[k,1], 
                    #                     "z": diff[k,2], },
                    #                     t)
                    # print(f"timestep {t}: object in the air")

                        
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

        print("Replay finished.")

    def run_single_traj_and_get_vlm_feedback(self, policy_nn, policy, output_dir: str) -> str:
        import omni.replicator.core as rep
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)  # creates the directory (and any parent directories if needed)

        # camera on the 0th env
        camera = self.camera
        # Firstly set up the camera
        # Then reset the environment and step all environment together
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._reset_idx(env_ids=env_ids)
        rep_writer = rep.BasicWriter(
            output_dir=output_dir,
            frame_padding=0,

        )
        offset = torch.tensor(self.data['franka'][0]["states"][self.start_idx_in_episode]['franka']["pos"], device=self.device)
        env_origin = self.scene.env_origins[0]

        camera_positions = torch.tensor([[1.24047, 0.0246736, 0.74118]], device=self.device) + env_origin
        camera_targets = torch.tensor([[0.0, 0.0, 0.0]], device=self.device) + env_origin + offset
        camera_orientations = torch.tensor(  # noqa: F841
        [[0.648518, 0.626944, -0.286825, -0.322639]], device=self.device
        ) # This is not being used
        camera.set_world_poses_from_view(camera_positions, camera_targets)
        camera_index = 0 # I only use one camera so this is the only index
        obs = self._get_observations()
        for i in range(self.max_episode_length):
            actions = policy(obs)
            obs, rewards, terminated, time_outs , extras = self.step(actions)
            # reset recurrent states for episodes that have terminated
            dones = terminated | time_outs
            policy_nn.reset(dones)
            camera.update(dt=self.sim.get_physics_dt())
            # if "rgb" in camera.data.output.keys():
            #     print("Received shape of rgb image        : ", camera.data.output["rgb"].shape)
            # Save images from camera at camera_index
            # note: BasicWriter only supports saving data in numpy format, so we need to convert the data to numpy.
            single_cam_data = convert_dict_to_backend(
                {k: v[camera_index] for k, v in camera.data.output.items()}, backend="numpy"
            )

            # Extract the other information
            single_cam_info = camera.data.info[camera_index]

            # Pack data back into replicator format to save them using its writer
            rep_output = {"annotators": {}}
            for key, data, info in zip(single_cam_data.keys(), single_cam_data.values(), single_cam_info.values()):
                if info is not None:
                    rep_output["annotators"][key] = {"render_product": {"data": data, **info}}
                else:
                    rep_output["annotators"][key] = {"render_product": {"data": data}}
            # Save images
            # Note: We need to provide On-time data for Replicator to save the images.
            rep_output["trigger_outputs"] = {"on_time": camera.frame[camera_index]}
            rep_writer.write(rep_output)
            if dones[0]:
                print("env has finished one episode")
                break
        output_text = self.get_feedback_from_vlm(output_dir=output_dir)
        return output_text
    
    def define_camera(self) -> Camera:
        """Defines the camera sensor to add to the scene."""
        # Setup camera sensor
        # In contrast to the ray-cast camera, we spawn the prim at these locations.
        # This means the camera sensor will be attached to these prims.
        sim_utils.create_prim("/World/Origin_00", "Xform")
        camera_cfg = CameraCfg(
            prim_path="/World/Origin_.*/CameraSensor",
            update_period=0,
            height=224,
            width=224,
            data_types=[
                "rgb",
                # "distance_to_image_plane",
                # "normals",
                # "semantic_segmentation",
                # "instance_segmentation_fast",
                # "instance_id_segmentation_fast",
            ],
            # colorize_semantic_segmentation=True,
            # colorize_instance_id_segmentation=True,
            # colorize_instance_segmentation=True,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
        )
        # Create camera
        self.camera = Camera(cfg=camera_cfg)


    def get_feedback_from_vlm(self, output_dir) -> str:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
        from qwen_vl_utils import process_vision_info
        use_cpu = True
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "prithivMLmods/DeepCaption-VLA-7B", torch_dtype="auto", device_map="cpu" if use_cpu else "auto"
        )
        processor = AutoProcessor.from_pretrained("prithivMLmods/DeepCaption-VLA-7B")
        num_samples = 30
        all_files = sorted(
            [f for f in os.listdir(output_dir) if f.endswith(".png") and f.startswith("rgb_")],
            key=lambda f: int(f.split("_")[1]),
        )
        if len(all_files) > num_samples:
            indices = np.linspace(0, len(all_files) - 1, num_samples, dtype=int)
            sampled_files = [all_files[i] for i in indices]
        else:
            sampled_files = all_files
        # Prepare image messages
        images_content = [{"type": "image", "image": os.path.join(output_dir, f)} for f in sampled_files]

        messages = [
            {
                "role": "user",
                "content": [
                    *images_content,
                    {
                        "type": "text",
                        "text": """
                        Analyze the sequence of images.

                        There is one coordinate frame attached to the object.

                        Answer the following questions. For each question:
                        - Repeat the question and answer using ONLY: (yes / no / unsure)

                        Questions:

                        Q1: Is the object grasped at any point in the sequence?
                        A1: <yes/no/unsure>

                        Q2: Is the object positioned above the basket at any point?
                        A2: <yes/no/unsure>

                        Q3: In the final frame, is the origin of the object coordinate frame located at the center of the gripper (which indicates that the object is not dropped)?
                        A3: <yes/no/unsure>

                        Reasoning: <brief explanation based on visible alignment and motion of the coordinate frames>
                        """
                    },
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        if not use_cpu:
            inputs = inputs.to("cuda")

        generated_ids = model.generate(
            **inputs, max_new_tokens=300, repetition_penalty=1.3, no_repeat_ngram_size=3
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]



    def get_target_object_to_hand_pose(self, init_states: dict):
        # Firstly find out the relative transformation, and then adjust the object pose according to robot pose.
        # Assume that the scene has been initialized
        pass



    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        self._compute_robot_intermediate_values(env_ids)
        self._compute_target_object_corners(env_ids)
        self._compute_basket_corners(env_ids)
        self.collision_obj_pos[env_ids] = self.collision_object.data.root_pos_w[env_ids]


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

    def _compute_basket_corners(self, env_ids: torch.Tensor):
        pos  = self.target_site.data.root_pos_w[env_ids]
        quat = quat_conjugate(self.target_site.data.root_quat_w[env_ids])
        self.basket_corners_world[env_ids] = transform_points(
            points=self.local_basket_corners_init[env_ids],
            pos=pos,
            quat=quat,
        )

    def _get_target_object_lowest_points(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        # shape: (N, num_corners, 3)
        corners = self.corners_target_obj[env_ids]
        # Extract z coordinates
        z_coords = corners[..., 2]
        # Get minimum z per environment
        lowest_z, _ = torch.min(z_coords, dim=1) # num_corners
        return lowest_z
    

    def _get_inside_site(self, env_ids: torch.Tensor):
        obj_xy = self.target_object.data.root_pos_w[env_ids, :2]
        site_pos = self.target_site.data.root_pos_w[env_ids, :2]
        dist2 = ((obj_xy - site_pos)**2).sum(dim=-1)
        inside_site = dist2 < self.target_site_radius**2
        return inside_site
    

    def define_markers(self) -> VisualizationMarkers:
        """Define markers with various different shapes."""
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
        translations_list = [
            x for flag, x in [
                (self.show_robot_grasp, self.robot_grasp_pos),
                (self.show_target_object, self.target_object.data.root_pos_w),
            ] if flag
        ]
        translations = torch.cat(translations_list, dim=0) if translations_list else None
        
        orientations_list = [
            x for flag, x in [
                (self.show_robot_grasp, self.robot_grasp_rot),
                (self.show_target_object, self.target_object.data.root_quat_w),
            ] if flag
        ]
        orientations = torch.cat(orientations_list, dim=0) if orientations_list else None
        self.visualizer.visualize(translations=translations,
                                orientations=orientations,
                                )

    def _get_rewards_test_manipulability_UNUSED(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        import torch

        eps = 1e-8

        # ------------------------------------------------------------------
        # Inputs
        # ------------------------------------------------------------------
        tcp_pos = torch.nan_to_num(self.robot_grasp_pos)
        target_pos = torch.nan_to_num(self.target_object.data.root_pos_w.clone())
        target_pos[:, 2] += 0.2

        manipulability = torch.nan_to_num(self.manipulability).clamp(min=0.0)

        # ------------------------------------------------------------------
        # Position tracking
        # ------------------------------------------------------------------
        pos_delta = tcp_pos - target_pos
        pos_error = torch.linalg.norm(pos_delta, dim=-1)

        # Gaussian-style reward: smoother and more standard than exp(-d / c)
        pos_sigma = 0.10
        r_position = torch.exp(-(pos_error ** 2) / (2 * pos_sigma ** 2))

        # Optional: success-style bonus when very close
        pos_close_thresh = 0.03
        r_pos_bonus = (pos_error < pos_close_thresh).float()

        # ------------------------------------------------------------------
        # Orientation tracking
        # Assumes self.to_desired_rot is the relative quaternion from current to desired
        # Quaternion format assumed: [w, x, y, z]
        # ------------------------------------------------------------------
        q_error = torch.nan_to_num(self.to_desired_rot)
        q_error = q_error / torch.linalg.norm(q_error, dim=-1, keepdim=True).clamp_min(eps)

        # Resolve quaternion sign ambiguity
        q_error = torch.where(q_error[:, 0:1] < 0.0, -q_error, q_error)

        # Angle in [0, pi]
        qw = torch.clamp(q_error[:, 0], -1.0 + eps, 1.0 - eps)
        angle_error = 2.0 * torch.acos(qw)
        # print(angle_error)
        # print(self.robot_grasp_rot)
        ori_sigma = 0.30
        r_orientation_raw = torch.exp(-(angle_error ** 2) / (2 * ori_sigma ** 2))

        # Gate orientation reward by position proximity
        # This prevents the policy from trying to perfectly orient while still far away
        orientation_gate = torch.exp(-(pos_error ** 2) / (2 * (0.15 ** 2)))
        r_orientation = orientation_gate * r_orientation_raw

        # Optional orientation success bonus
        ori_close_thresh = 0.15  # radians
        r_ori_bonus = ((angle_error < ori_close_thresh) & (pos_error < 0.05)).float()

        # ------------------------------------------------------------------
        # Manipulability reward
        # Keep small so it does not dominate tracking
        # ------------------------------------------------------------------
        r_manipulability = torch.tanh(manipulability)

        # ------------------------------------------------------------------
        # Final reward
        # ------------------------------------------------------------------
        reward = (
            2.5 * r_position +
            1.5 * r_orientation +
            0.1 * r_manipulability
        )

        individual_rewards = {
            # "position_tracking": r_position,
            # "position_bonus": r_pos_bonus,
            # "orientation_tracking": r_orientation,
            # "orientation_tracking_raw": r_orientation_raw,
            # "orientation_gate": orientation_gate,
            # "orientation_bonus": r_ori_bonus,
            # "manipulability": r_manipulability,
            "pos_error": pos_error,
            "angle_error": angle_error,
        }

        return reward, individual_rewards


    def _get_rewards_test(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        import torch

        eps = 1.0e-6

        # ------------------------------------------------------------------
        # Sanitize inputs
        # ------------------------------------------------------------------
        tcp_pos = torch.nan_to_num(self.robot_grasp_pos)
        obj_pos = torch.nan_to_num(self.target_object.data.root_pos_w)
        obj_vel = torch.nan_to_num(self.target_object.data.root_lin_vel_w)
        collision_obj_pos = torch.nan_to_num(self.collision_obj_pos)
        site_pos = torch.nan_to_num(self.target_site.data.root_pos_w)
        basket_corners = torch.nan_to_num(self.basket_corners_world)
        basket_init_pos = torch.nan_to_num(self.basket_init_pos)

        joint_pos = torch.nan_to_num(self._robot.data.joint_pos)
        joint_vel = torch.nan_to_num(self._robot.data.joint_vel)
        actions = torch.nan_to_num(self.actions)
        prev_actions = torch.nan_to_num(self.prev_actions)
        dof_lower = torch.nan_to_num(self.robot_dof_lower_limits)
        dof_upper = torch.nan_to_num(self.robot_dof_upper_limits)

        collision_triggered = torch.nan_to_num(self.collision_triggered.float()).bool()

        # ------------------------------------------------------------------
        # Geometry
        # ------------------------------------------------------------------
        target_to_hand = obj_pos - tcp_pos
        target_to_hand_dist = torch.norm(target_to_hand, dim=-1)

        obj_to_collision = collision_obj_pos - obj_pos
        obj_to_collision_dist = torch.norm(obj_to_collision, dim=-1)

        site_to_obj = site_pos - obj_pos
        site_to_obj_dist = torch.norm(site_to_obj, dim=-1)
        site_to_obj_xy = site_to_obj[:, :2]
        site_to_obj_xy_dist = torch.norm(site_to_obj_xy, dim=-1)

        basket_z = basket_corners[:, :, 2]
        basket_bottom_z = basket_z.min(dim=1).values
        basket_top_z = basket_z.max(dim=1).values
        basket_center_z = 0.5 * (basket_bottom_z + basket_top_z)

        obj_z = obj_pos[:, 2]
        tcp_z = tcp_pos[:, 2]

        target_site_radius = float(self.target_site_radius)
        basket_drift_threshold = float(self.cfg.basket_drift_threshold)

        inside_site = (site_to_obj_xy_dist ** 2) < (target_site_radius ** 2)
        low_enough = obj_z < basket_top_z
        high_enough_for_basket = obj_z > basket_bottom_z
        inside_height_band = low_enough & high_enough_for_basket

        basket_displacement = torch.norm(site_pos - basket_init_pos, dim=-1)
        basket_untouched = basket_displacement < basket_drift_threshold

        success = collision_triggered & inside_site & low_enough & high_enough_for_basket & basket_untouched

        pre_collision = (~collision_triggered).float()
        post_collision = collision_triggered.float()

        gripper_open_cmd = torch.clamp(actions[:, -1], min=0.0, max=1.0)
        gripper_close_cmd = torch.clamp(-actions[:, -1], min=0.0, max=1.0)

        # ------------------------------------------------------------------
        # Analysis:
        # Policy already solves approach/grasp/collision.
        # Main failure is post-collision placement:
        # - release-related rewards are ~0, so gripper almost never opens.
        # - exact in-basket rewards are too sparse.
        # - current reward still pays enough for holding object near basket.
        #
        # Fix:
        # 1) Keep strong early-stage guidance.
        # 2) Make basket transport/placement denser with broad->tight XY shaping.
        # 3) Add a release-opportunity reward that is achievable before exact success.
        # 4) Penalize keeping gripper closed when aligned over basket.
        # 5) Make success bonus dominant.
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Stage 1: approach and grasp
        # ------------------------------------------------------------------
        approach_temp = 0.06
        approach_target = torch.exp(-torch.clamp(target_to_hand_dist, 0.0, 5.0) / approach_temp)

        close_loose = (target_to_hand_dist < 0.08).float()
        close_mid = (target_to_hand_dist < 0.05).float()
        close_tight = (target_to_hand_dist < 0.03).float()

        grasp_close_reward = gripper_close_cmd * close_mid
        secure_grasp_reward = gripper_close_cmd * close_tight

        lift_amount = torch.clamp(obj_z - (basket_top_z + 0.02), min=0.0, max=0.25)
        lift_temp = 0.05
        lift_shaping = 1.0 - torch.exp(-lift_amount / lift_temp)

        controlled_pick = close_mid * gripper_close_cmd * (0.5 + 0.5 * lift_shaping)
        secure_pick = close_tight * gripper_close_cmd * (0.5 + 0.5 * lift_shaping)

        # ------------------------------------------------------------------
        # Stage 2: collision
        # ------------------------------------------------------------------
        collision_approach_temp = 0.08
        move_to_collision = torch.exp(-torch.clamp(obj_to_collision_dist, 0.0, 5.0) / collision_approach_temp)
        pre_collision_guidance = move_to_collision * torch.maximum(controlled_pick, secure_pick) * pre_collision
        collision_bonus = collision_triggered.float()

        # ------------------------------------------------------------------
        # Stage 3: dense basket transport and placement
        # ------------------------------------------------------------------
        basket_xy_broad_temp = 0.22
        basket_xy_mid_temp = 0.12
        basket_xy_tight_temp = 0.06
        basket_xyz_temp = 0.18

        basket_xy_broad = torch.exp(-torch.clamp(site_to_obj_xy_dist, 0.0, 5.0) / basket_xy_broad_temp) * post_collision
        basket_xy_mid = torch.exp(-torch.clamp(site_to_obj_xy_dist, 0.0, 5.0) / basket_xy_mid_temp) * post_collision
        basket_xy_tight = torch.exp(-torch.clamp(site_to_obj_xy_dist, 0.0, 5.0) / basket_xy_tight_temp) * post_collision
        basket_xyz_align = torch.exp(-torch.clamp(site_to_obj_dist, 0.0, 5.0) / basket_xyz_temp) * post_collision

        above_basket_xy = (site_to_obj_xy_dist < 1.50 * target_site_radius).float() * post_collision
        near_basket_xy = (site_to_obj_xy_dist < 1.00 * target_site_radius).float() * post_collision
        centered_xy = (site_to_obj_xy_dist < 0.60 * target_site_radius).float() * post_collision

        carry_target_z = basket_top_z + 0.06
        carry_height_err = torch.abs(obj_z - carry_target_z)
        carry_height_temp = 0.06
        carry_height_reward = torch.exp(-torch.clamp(carry_height_err, 0.0, 5.0) / carry_height_temp) * post_collision

        top_height_err = torch.abs(obj_z - basket_top_z)
        top_height_temp = 0.04
        top_height_align = torch.exp(-torch.clamp(top_height_err, 0.0, 5.0) / top_height_temp) * basket_xy_mid

        center_height_err = torch.abs(obj_z - basket_center_z)
        center_height_temp = 0.05
        center_height_align = torch.exp(-torch.clamp(center_height_err, 0.0, 5.0) / center_height_temp) * basket_xy_tight

        # Soft placement terms
        xy_inside_soft = torch.clamp(1.0 - site_to_obj_xy_dist / max(target_site_radius, eps), min=0.0, max=1.0)
        vertical_margin = 0.03
        vertical_clearance = torch.minimum(obj_z - basket_bottom_z, basket_top_z - obj_z)
        inside_height_soft = torch.clamp(vertical_clearance / vertical_margin, min=0.0, max=1.0)

        in_basket_soft = xy_inside_soft * inside_height_soft * basket_untouched.float() * post_collision
        in_basket_partial = inside_site.float() * inside_height_band.float() * basket_untouched.float() * post_collision

        descend_progress = torch.clamp(
            0.20 * basket_xy_mid
            + 0.25 * top_height_align
            + 0.25 * center_height_align
            + 0.30 * in_basket_soft,
            min=0.0,
            max=1.0,
        ) * basket_untouched.float() * post_collision

        placement_ready = torch.clamp(
            0.30 * basket_xy_mid
            + 0.20 * basket_xy_tight
            + 0.20 * top_height_align
            + 0.10 * center_height_align
            + 0.20 * in_basket_soft,
            min=0.0,
            max=1.0,
        ) * basket_untouched.float() * post_collision

        # ------------------------------------------------------------------
        # Release shaping
        # Key redesign: make opening reward achievable when aligned over basket,
        # not only when already exactly inside basket.
        # ------------------------------------------------------------------
        release_zone = torch.clamp(
            0.35 * basket_xy_mid
            + 0.20 * basket_xy_tight
            + 0.20 * top_height_align
            + 0.10 * center_height_align
            + 0.15 * in_basket_soft,
            min=0.0,
            max=1.0,
        ) * basket_untouched.float() * post_collision

        open_over_basket = gripper_open_cmd * release_zone
        open_for_drop = gripper_open_cmd * placement_ready
        open_in_basket = gripper_open_cmd * in_basket_partial

        keep_hold_transport = gripper_close_cmd * basket_xy_broad * post_collision
        hold_closed_release_zone = gripper_close_cmd * release_zone
        hold_closed_placement = gripper_close_cmd * placement_ready

        premature_release = gripper_open_cmd * post_collision * (1.0 - release_zone)

        # Penalize hovering high above basket while closed
        height_above_top = torch.clamp(obj_z - basket_top_z, min=0.0, max=0.30)
        hover_high_over_basket = basket_xy_tight * gripper_close_cmd * torch.clamp(height_above_top / 0.07, 0.0, 1.0)

        # Encourage TCP slightly above rim during release
        tcp_release_target_z = basket_top_z + 0.08
        tcp_release_err = torch.abs(tcp_z - tcp_release_target_z)
        tcp_release_temp = 0.06
        tcp_release_posture = torch.exp(-torch.clamp(tcp_release_err, 0.0, 5.0) / tcp_release_temp) * open_over_basket

        # ------------------------------------------------------------------
        # Basket protection
        # ------------------------------------------------------------------
        basket_stable_temp = 0.01
        basket_stable = torch.exp(-torch.clamp(basket_displacement, 0.0, 1.0) / basket_stable_temp)
        basket_drift_excess = torch.clamp(basket_displacement - basket_drift_threshold, min=0.0)
        basket_drift_ratio = torch.clamp(
            basket_displacement / max(basket_drift_threshold, eps),
            min=0.0,
            max=3.0,
        )

        # ------------------------------------------------------------------
        # Regularization
        # ------------------------------------------------------------------
        joint_speed_penalty = torch.norm(joint_vel, dim=-1)
        action_rate_penalty = torch.norm(actions - prev_actions, dim=-1)
        obj_speed_penalty = torch.norm(obj_vel, dim=-1)

        joint_range = torch.clamp(dof_upper - dof_lower, min=1.0e-3)
        dist_lower = (joint_pos - dof_lower) / joint_range
        dist_upper = (dof_upper - joint_pos) / joint_range
        min_limit_dist = torch.minimum(dist_lower, dist_upper)

        joint_limit_margin = 0.10
        joint_limit_penalty = torch.clamp(
            (joint_limit_margin - min_limit_dist) / joint_limit_margin,
            min=0.0,
            max=1.0,
        ).sum(dim=-1)

        # ------------------------------------------------------------------
        # Total reward
        # Success bonus is made dominant versus best shaping.
        # Non-success shaping is kept moderate to avoid rewarding hover/hold.
        # ------------------------------------------------------------------
        reward = (
            # Stage 1
            0.16 * approach_target
            + 0.04 * close_loose
            + 0.07 * close_mid
            + 0.10 * close_tight
            + 0.12 * grasp_close_reward
            + 0.14 * secure_grasp_reward
            + 0.16 * controlled_pick
            + 0.18 * secure_pick
            # Stage 2
            + 0.28 * pre_collision_guidance
            + 0.75 * collision_bonus
            # Stage 3 transport
            + 0.18 * basket_xy_broad
            + 0.22 * basket_xy_mid
            + 0.08 * basket_xyz_align
            + 0.10 * above_basket_xy
            + 0.14 * near_basket_xy
            + 0.12 * centered_xy
            + 0.10 * carry_height_reward
            + 0.05 * keep_hold_transport
            # Stage 3 placement
            + 0.35 * basket_xy_tight
            + 0.45 * top_height_align
            + 0.55 * center_height_align
            + 0.80 * in_basket_soft
            + 1.20 * in_basket_partial
            + 0.70 * descend_progress
            + 0.55 * placement_ready
            # Release
            + 1.10 * open_over_basket
            + 1.60 * open_for_drop
            + 2.20 * open_in_basket
            + 0.15 * tcp_release_posture
            # Basket protection
            + 0.05 * basket_stable
            # Success
            + 20.0 * success.float()
            # Penalties
            - 0.45 * premature_release
            - 0.55 * hold_closed_release_zone
            - 0.95 * hold_closed_placement
            - 0.70 * hover_high_over_basket
            - 3.50 * basket_drift_excess
            - 0.05 * basket_drift_ratio
            - 0.0015 * joint_speed_penalty
            - 0.0030 * action_rate_penalty
            - 0.0200 * joint_limit_penalty
            - 0.0006 * obj_speed_penalty
        )

        reward = torch.nan_to_num(reward, nan=-5.0, posinf=5.0, neginf=-5.0)
        reward = torch.clamp(reward, -5.0, 5.0)

        individual_rewards_dict = {
            "approach_target": torch.nan_to_num(approach_target),
            "close_loose": torch.nan_to_num(close_loose),
            "close_mid": torch.nan_to_num(close_mid),
            "close_tight": torch.nan_to_num(close_tight),
            "grasp_close_reward": torch.nan_to_num(grasp_close_reward),
            "secure_grasp_reward": torch.nan_to_num(secure_grasp_reward),
            "lift_shaping": torch.nan_to_num(lift_shaping),
            "controlled_pick": torch.nan_to_num(controlled_pick),
            "secure_pick": torch.nan_to_num(secure_pick),
            "move_to_collision": torch.nan_to_num(move_to_collision),
            "pre_collision_guidance": torch.nan_to_num(pre_collision_guidance),
            "collision_bonus": torch.nan_to_num(collision_bonus),
            "basket_xy_broad": torch.nan_to_num(basket_xy_broad),
            "basket_xy_mid": torch.nan_to_num(basket_xy_mid),
            "basket_xy_tight": torch.nan_to_num(basket_xy_tight),
            "basket_xyz_align": torch.nan_to_num(basket_xyz_align),
            "above_basket_xy": torch.nan_to_num(above_basket_xy),
            "near_basket_xy": torch.nan_to_num(near_basket_xy),
            "centered_xy": torch.nan_to_num(centered_xy),
            "carry_height_reward": torch.nan_to_num(carry_height_reward),
            "top_height_align": torch.nan_to_num(top_height_align),
            "center_height_align": torch.nan_to_num(center_height_align),
            "xy_inside_soft": torch.nan_to_num(xy_inside_soft),
            "inside_height_soft": torch.nan_to_num(inside_height_soft),
            "in_basket_soft": torch.nan_to_num(in_basket_soft),
            "in_basket_partial": torch.nan_to_num(in_basket_partial),
            "descend_progress": torch.nan_to_num(descend_progress),
            "placement_ready": torch.nan_to_num(placement_ready),
            "release_zone": torch.nan_to_num(release_zone),
            "open_over_basket": torch.nan_to_num(open_over_basket),
            "open_for_drop": torch.nan_to_num(open_for_drop),
            "open_in_basket": torch.nan_to_num(open_in_basket),
            "premature_release": torch.nan_to_num(premature_release),
            "keep_hold_transport": torch.nan_to_num(keep_hold_transport),
            "hold_closed_release_zone": torch.nan_to_num(hold_closed_release_zone),
            "hold_closed_placement": torch.nan_to_num(hold_closed_placement),
            "hover_high_over_basket": torch.nan_to_num(hover_high_over_basket),
            "tcp_release_posture": torch.nan_to_num(tcp_release_posture),
            "basket_stable": torch.nan_to_num(basket_stable),
            "basket_drift_excess": torch.nan_to_num(basket_drift_excess),
            "basket_drift_ratio": torch.nan_to_num(basket_drift_ratio),
            "joint_speed_penalty": torch.nan_to_num(joint_speed_penalty),
            "action_rate_penalty": torch.nan_to_num(action_rate_penalty),
            "joint_limit_penalty": torch.nan_to_num(joint_limit_penalty),
            "obj_speed_penalty": torch.nan_to_num(obj_speed_penalty),
            "collision_triggered": collision_triggered.float(),
            "inside_site": inside_site.float(),
            "inside_height_band": inside_height_band.float(),
            "basket_untouched": basket_untouched.float(),
            "success_bonus": success.float(),
            "success": success.float(),
        }

        assert torch.isfinite(reward).all(), "Non-finite reward detected in _get_rewards_eureka"
        for k, v in individual_rewards_dict.items():
            assert torch.isfinite(v).all(), f"Non-finite reward component detected: {k}"

        return reward, individual_rewards_dict


@configclass
class TestSlightCollideAndPlaceCfg(TestCollideAndPlaceCfg):
    collision_max_displacement: float = 0.2  # metres; collision object must not be pushed beyond this


class TestSlightCollideAndPlace(TestCollideAndPlace):
    """Full task: slight collision (≤ 0.2 m) + place in basket."""

    cfg: TestSlightCollideAndPlaceCfg

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, truncated = super()._get_dones()
        collision_displacement = torch.norm(
            self.collision_obj_pos - self.collision_object_init_pos, dim=-1
        )
        collision_not_too_hard = collision_displacement <= self.cfg.collision_max_displacement
        return terminated & collision_not_too_hard, truncated


@configclass
class TestSlightCollideCfg(TestSlightCollideAndPlaceCfg):
    collision_stop_vel_threshold: float = 0.02  # m/s; collision object must be nearly stopped


class TestSlightCollide(TestSlightCollideAndPlace):
    """Stage 1: terminate when a gentle collision is triggered and the collision object has come to rest."""

    cfg: TestSlightCollideCfg

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        _, truncated = TestCollideAndPlace._get_dones(self)
        collision_displacement = torch.norm(
            self.collision_obj_pos - self.collision_object_init_pos, dim=-1
        )
        collision_not_too_hard = collision_displacement <= self.cfg.collision_max_displacement
        collision_obj_vel = torch.norm(self.collision_object.data.root_lin_vel_w, dim=-1)
        collision_obj_stopped = collision_obj_vel < self.cfg.collision_stop_vel_threshold
        terminated = self.collision_triggered & collision_not_too_hard & collision_obj_stopped
        return terminated, truncated
