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
class ReplayLivingRoomScene1PickUpTheAlphabetSoupAndPutItInTheBasketCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2
    action_space = 9
    observation_space = 25 # to modify
    state_space = 0
    path = "libero/trajs/libero90/libero_90_living_room_scene1_pick_up_the_alphabet_soup_and_put_it_in_the_basket_traj_v2.pkl"

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
        num_envs=50, env_spacing=3.0, replicate_physics=True, clone_in_fabric=True
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

    alphabet_soup = RigidObjectCfg(
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

    cream_cheese = RigidObjectCfg(
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


class ReplayLivingRoomScene1PickUpTheAlphabetSoupAndPutItInTheBasket(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: ReplayLivingRoomScene1PickUpTheAlphabetSoupAndPutItInTheBasketCfg

    def __init__(self, cfg: ReplayLivingRoomScene1PickUpTheAlphabetSoupAndPutItInTheBasketCfg, render_mode: str | None = None, **kwargs):
        self.debug = False  
        self.path = cfg.path # scene init is called in super
        from isaaclab_eureka.utils import read_pkl
        self.data = read_pkl(self.path) 
        lengths = [len(ep["states"]) for ep in self.data["franka"]]
        self.max_len = max(lengths)
        # [num_envs, step, states]
        # first loop over, add padding, then stack.

        
        episodes = self.data["franka"]
        num_envs = len(episodes) # overwrite num_envs
        cfg.scene = InteractiveSceneCfg(
            num_envs=num_envs, env_spacing=3.0, replicate_physics=True, clone_in_fabric=True
        )


        super().__init__(cfg, render_mode, **kwargs)



        sample = episodes[0]["states"][0].copy()
        sample.pop("franka", None)
        
        # 1. Determine object names and ordering
        self.object_names = sorted(list(sample.keys()))
        if self.debug:
            print(self.object_names)
        self.object_index = {name: i for i, name in enumerate(self.object_names)}

        num_objects = len(self.object_names)
        # 2. Compute max episode length
        lengths = [len(ep["states"]) for ep in episodes]
        if self.debug:
            print(lengths)
        max_len = max(lengths)
        if self.debug:
            print(f"max_len {max_len}")
        # 3. Determine state dimensions
        # robot
        robot_pos_dim = len(episodes[0]["states"][0]["franka"]["pos"])      # 3

        # actually I am not using this..
        robot_rot_dim = len(episodes[0]["states"][0]["franka"]["rot"])      # 4
        robot_dof_dim = len(episodes[0]["states"][0]["franka"]["dof_pos"])  # num_joints

        # rigid objects (all share same structure)
        one_obj = next(iter({k: v for k, v in sample.items() if k != "franka"}.values()))
        object_pos_dim = len(one_obj["pos"])              # 3
        object_rot_dim = len(one_obj["rot"])              # 4

        # 4. Preallocate tensors
        # self.robot_pos = torch.zeros((self.num_envs, max_len, robot_pos_dim))
        # self.robot_rot = torch.zeros((self.num_envs, max_len, robot_rot_dim))
        self.robot_dof = torch.zeros((self.num_envs, max_len, robot_dof_dim),device=self.device,dtype=torch.float32)

        self.object_pos = torch.zeros((self.num_envs, max_len, num_objects, object_pos_dim),device=self.device,dtype=torch.float32)
        self.object_rot = torch.zeros((self.num_envs, max_len, num_objects, object_rot_dim),device=self.device,dtype=torch.float32)

        self.padding_mask = torch.ones((self.num_envs, max_len), dtype=torch.bool)

        # 5. Fill tensors
        for env_idx, ep in enumerate(episodes): # each episode
            ep_len = lengths[env_idx]

            for t, state in enumerate(ep["states"]):
                # robot
                # self.robot_pos[env_idx, t] = torch.tensor(state["franka"]["pos"])
                # self.robot_rot[env_idx, t] = torch.tensor(state["franka"]["rot"])

                dof_vals = [v[0] for v in state["franka"]["dof_pos"].values()]
                self.robot_dof[env_idx, t] = torch.tensor(dof_vals)

                # objects
                for obj_idx, obj_name in enumerate(self.object_names):
                    obj = state[obj_name]
                    self.object_pos[env_idx, t, obj_idx] = torch.tensor(obj["pos"])
                    self.object_rot[env_idx, t, obj_idx] = torch.tensor(obj["rot"])

            # padding mask
            self.padding_mask[env_idx, :ep_len] = False

        # should be different for interaction and placement predicate, here placement

        self.target_region = "basket"
        self.input_direction = 2 # z axis

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


        stage = get_current_stage()

        prim = stage.GetPrimAtPath("/World/envs/env_0/alphabet_soup")
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(prim)
        min_corner = bbox.GetRange().GetMin()  # Vec3
        max_corner = bbox.GetRange().GetMax()  # Vec3
        self.target_object_size = torch.tensor((max_corner - min_corner), device=self.device)



        alphabet_soup_local_grasp_pose = torch.tensor([0.2, 0.3, 0.03, 1, 0, 0, 0], device=self.device)
        self.alphabet_soup_grasp_rot = alphabet_soup_local_grasp_pose[3:7].repeat((self.num_envs, 1))



        
   
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(prim)
        min_corner = bbox.GetRange().GetMin()  # Vec3
        max_corner = bbox.GetRange().GetMax()  # Vec3
        alphabet_soup_size = torch.tensor((max_corner - min_corner), device=self.device)
        if self.debug:
            print(f"alphabet_soup_size {alphabet_soup_size}")
        alphabet_soup_local_center = torch.tensor((min_corner + max_corner)/2, device=self.device)
        self.alphabet_soup_size = alphabet_soup_size.repeat((self.num_envs, 1)) # 3
        self.single_size = alphabet_soup_size
        
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
        self.alphabet_soup_inward_axis = torch.tensor([-1, 0, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.alphabet_soup_up_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
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


    def _setup_scene(self): # done
        init_states = self.data['franka'][0]["init_state"] # this is from the first scene as set up. Init states of traj should be updated in reset_idx
        robot_data = init_states['franka'] # seems that joint pos for isaaclab is always positive
        robot_joint_pos = robot_data["dof_pos"]
        print(f"robot_joint_pos:{robot_joint_pos}")
        robot_joint_pos['panda_finger_joint2'] = robot_joint_pos['panda_finger_joint1']
        print(f"robot_joint_pos:{robot_joint_pos}")
        robot_joint_pos = {k: v.item() for k, v in robot_joint_pos.items()}

        if self.debug:
            print(f"robot_joint_pos {robot_joint_pos}")
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

        keys = list(init_states.keys())
        keys.remove("franka")   # remove the one you don't want
        self.rigid_objects = {}
        for k in keys:
            cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/{k}",
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=init_states[k]["pos"],
                    rot=init_states[k]["rot"],
                ),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=f"/home/admin_01/workspace_eureka/IsaacLabEureka/libero/COMMON/stable_hope_objects/{k}/usd/{k}.usd",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        articulation_enabled=False
                    )
                ),  
            )
            object = RigidObject(cfg)
            self.rigid_objects[k] = object
            self.scene.rigid_objects[k] = object # can I directly assign the dict?
        self.target_object_name = "alphabet_soup"
        self.target_region = "basket"
        self.input_direction = 2 # z axis
        self.target_object : RigidObject = self.rigid_objects[self.target_object_name]
        if self.debug:
            print(self.rigid_objects.keys())
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
        pass


    def _apply_action(self): # done, here applying states
        # here apply all states
        t = self.common_step_counter
        joint_pos = self.robot_dof[:,t,:]
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel)
        # If I want velocity, then I have to run the controllers instead of directly setting states

        for object_name in self.object_names:
            i= self.object_index[object_name]
            object = self.rigid_objects[object_name]
            object_state = torch.zeros((self.num_envs, 13), device=self.device)
            object_state[:,0:3] = self.object_pos[:,t,i,:] + self.scene.env_origins
            object_state[:,3:7] = self.object_rot[:,t,i,:]
            object.write_root_pose_to_sim(object_state[:, :7])
            object.write_root_velocity_to_sim(object_state[:, 7:])


    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:



        truncated = self.episode_length_buf >= self.max_len
        terminated = truncated # Here for avoiding reset with single 
        return terminated, truncated
    
    def _get_rewards(self) -> torch.Tensor:

        return self._compute_rewards(self.actions
                                     ,self.cfg.action_penalty_scale)



    def _reset_idx(self, env_ids: torch.Tensor | None):
        if self.debug:
            print(f"current_time_step:{self.common_step_counter}")
        if self.common_step_counter == 0:
            print("resetting")
            super()._reset_idx(env_ids)
            t = self.common_step_counter
            joint_pos = self.robot_dof[:,t,:]
            joint_vel = torch.zeros_like(joint_pos)
            self._robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # If I want velocity, then I have to run the controllers instead of directly setting states

            for object_name in self.object_names:
                i= self.object_index[object_name]
                object = self.rigid_objects[object_name]
                object_state = torch.zeros((self.num_envs, 13), device=self.device)
                object_state[:,0:3] = self.object_pos[:,t,i,:] + self.scene.env_origins
                object_state[:,3:7] = self.object_rot[:,t,i,:]
                object.write_root_pose_to_sim(object_state[:, :7])
                object.write_root_velocity_to_sim(object_state[:, 7:])
        else:
            if self.debug:
                print("time step none zero, pass")
        
    def _get_observations(self) -> dict:
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )



        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
            ),
            dim=-1,
        )
        return {}

    # auxiliary methods


    def _compute_rewards(
        self,
        actions,
        action_penalty_scale,
    ):
        # distance from hand to the drawer
        # print(f"cream_pos{self._cream_cheese.data.root_pose_w}")
        # print(f"cream{cream_cheese_grasp_pos}") # check
        # print(franka_grasp_pos)


        # if self.debug:
        #     print(f"franka_grasp_rot{franka_grasp_rot}\ncream_cheese_grasp_rot{cream_cheese_grasp_rot}")
        

        actions_copy = torch.ones_like(actions) * self.common_step_counter
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



    

    def _get_rewards_eureka(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
