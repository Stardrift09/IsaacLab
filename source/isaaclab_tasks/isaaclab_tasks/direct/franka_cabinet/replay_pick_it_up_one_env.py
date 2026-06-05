# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physics-faithful single-env replay of LIBERO demos for the PickItUp task.

Unlike ``TestPickItUp.run_replay`` (kinematic: teleports both the arm joints and
the object every step), this replays a *single* demo in a *single* env by driving
the robot with a controller while letting the object obey physics. The grasp /
lift / place therefore has to actually emerge from the dynamics, so watching the
``terminated`` flag tells us whether the demo trajectory is physically realizable
under the chosen controller.

Two controllers, selected by the hard-coded ``REPLAY_CONTROLLER`` flag:

* ``"joint_pd"`` - drive the arm with joint position targets using the env's
  ImplicitActuator PD gains (``set_joint_position_target`` + physics step).
* ``"osc"`` - drive the arm with the IsaacLab Operational Space Controller toward
  per-frame end-effector pose targets (FK'd from the demo joints). Gains are set
  to LIBERO's robosuite ``OSC_POSE`` defaults (kp=150, damping_ratio=1), which is
  the controller LIBERO actually used to generate the demos.

NOTE on kp/kd: the demo joints came from MuJoCo OSC control, so joint-space PD
gains are not recorded anywhere. ``_set_replay_gains`` is left here as a hook for
later tuning. A non-terminating replay is informative (the gains/trajectory need
work), not necessarily a bug.
"""

from __future__ import annotations

import json
import os

import torch
from torch.utils.tensorboard import SummaryWriter

import isaaclab.sim as sim_utils
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.utils import configclass, convert_dict_to_backend
from isaaclab.utils.math import matrix_from_quat, quat_inv, quat_apply_inverse, subtract_frame_transforms

from .test_pick_it_up import TestPickItUp, TestPickItUpCfg


@configclass
class ReplayPickItUpOneEnvCfg(TestPickItUpCfg):
    # Deterministic replay: object starts on the table, no randomization.
    start_in_air = False
    randomize_init = False
    randomize_rotation = False
    arm_joint_noise = 0.0
    object_pos_noise = 0.0
    basket_pos_noise = 0.0

    # LIBERO-style video recording (agentview + wrist eye-in-hand). Off by
    # default; the recording script flips it on (and launches with enable_cameras).
    record_cameras: bool = False
    cam_w: int = 256
    cam_h: int = 256


class ReplayPickItUpOneEnv(TestPickItUp):
    cfg: ReplayPickItUpOneEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Disable debug-vis markers so they don't pollute recorded videos:
        #  - self.debug_vis gates the parent's grasp/target axis arrows (debug_vis_mine)
        #  - the robot/target-object cfgs set debug_vis=True -> per-asset frame triads
        self.debug_vis = False
        for asset in [self._robot, *self.rigid_objects.values()]:
            try:
                asset.set_debug_vis(False)
            except Exception:
                pass
        # The parent builds self.visualizer (define_markers) in __init__; its
        # prototype prims sit at the world origin (the leftover RGB triad) since we
        # never call .visualize(). Hide it.
        if getattr(self, "visualizer", None) is not None:
            try:
                self.visualizer.set_visibility(False)
            except Exception:
                pass

    # --- hard-coded replay controls (flip these to experiment) ---
    REPLAY_CONTROLLER = "osc"   # "joint_pd" | "osc"
    REPLAY_STEPS_PER_FRAME = 6       # physics steps to hold each demo frame as target
                                     # (libero 1/20 s / sim 1/120 s = 6). No interpolation;
                                     # this is the "lazy / low-Hz target update".
    OSC_KP = 150.0                   # LIBERO robosuite OSC_POSE default
    OSC_DAMPING_RATIO = 1.0          # LIBERO robosuite OSC_POSE default
    OSC_EE_BODY = "panda_hand"       # end-effector frame for OSC

    # joint_pd arm gains. None -> keep the env's _setup_scene ImplicitActuator
    # gains. Set (e.g. by the gain sweep) to override all 7 arm joints.
    PD_KP = None
    PD_KD = None

    # --- LIBERO-style recording cameras ---
    # agentview: external front view (eye in env-local coords, looking at workspace).
    AGENTVIEW_EYE = (1.05, 0.0, 0.55)
    AGENTVIEW_TARGET = (0.0, 0.0, 0.08)
    # wrist eye_in_hand: LIBERO/robosuite mount, rigidly on panda_hand. From the
    # Panda robot.xml: <camera name="eye_in_hand" pos="0.05 0 0"
    # quat="0 0.707108 0.707108 0" fovy="75"/>. That quat (w,x,y,z) makes the camera
    # look along hand +Z (approach) with up = hand +X -> OpenGL/USD camera convention,
    # so we attach with convention="opengl" and the same pos/quat.
    WRIST_OFFSET_POS = (0.05, 0.0, 0.0)
    WRIST_OFFSET_ROT = (0.0, 0.707108, 0.707108, 0.0)
    WRIST_FOVY = 75.0  # degrees (vertical)

    # ------------------------------------------------------------------
    # camera setup / capture (only when cfg.record_cameras)
    # ------------------------------------------------------------------
    def _setup_scene(self):
        super()._setup_scene()
        if getattr(self.cfg, "record_cameras", False):
            self._define_record_cameras()

    def _define_record_cameras(self) -> None:
        """Create the two LIBERO-style RGB cameras (env_0 only)."""
        import math

        spawn = sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955,
            clipping_range=(0.01, 1.0e5),
        )
        # wrist: focal_length chosen so vertical fov == LIBERO fovy (square sensor).
        ap = 20.955
        wrist_focal = ap / (2.0 * math.tan(math.radians(self.WRIST_FOVY) / 2.0))
        wrist_spawn = sim_utils.PinholeCameraCfg(
            focal_length=wrist_focal, focus_distance=400.0, horizontal_aperture=ap,
            clipping_range=(0.001, 1.0e5),
        )
        agentview_cfg = CameraCfg(
            prim_path="/World/envs/env_0/agentview_cam",
            update_period=0, height=self.cfg.cam_h, width=self.cfg.cam_w,
            data_types=["rgb"], spawn=spawn,
        )
        # wrist eye_in_hand: rigidly attached to panda_hand with LIBERO's mount.
        wrist_cfg = CameraCfg(
            prim_path="/World/envs/env_0/Robot/panda_hand/wrist_cam",
            update_period=0, height=self.cfg.cam_h, width=self.cfg.cam_w,
            data_types=["rgb"], spawn=wrist_spawn,
            offset=CameraCfg.OffsetCfg(
                pos=self.WRIST_OFFSET_POS, rot=self.WRIST_OFFSET_ROT, convention="opengl"
            ),
        )
        self._agentview_cam = Camera(cfg=agentview_cfg)
        self._wrist_cam = Camera(cfg=wrist_cfg)
        # ordered dict -> stable video/key naming (matches LIBERO obs keys)
        self._record_cams = {"agentview": self._agentview_cam, "wrist": self._wrist_cam}

    def _aim_agentview(self) -> None:
        """Point the static agentview camera at the workspace (call after play)."""
        origin = self.scene.env_origins[0]
        eye = torch.tensor([self.AGENTVIEW_EYE], device=self.device) + origin
        target = torch.tensor([self.AGENTVIEW_TARGET], device=self.device) + origin
        self._agentview_cam.set_world_poses_from_view(eye, target)

    def _capture_frame(self) -> dict:
        """Render once and return {cam_name: HxWx3 uint8 np array} for env_0."""
        self.sim.render()
        frame = {}
        for name, cam in self._record_cams.items():
            cam.update(dt=self.physics_dt)
            rgb = cam.data.output["rgb"][0][..., :3]
            frame[name] = rgb.detach().cpu().numpy().astype("uint8")
        return frame

    # ------------------------------------------------------------------
    # gain override hook (kp/kd finding deferred -- left here for later)
    # ------------------------------------------------------------------
    def _set_replay_gains(self, kp: float | None = None, kd: float | None = None,
                          joint_ids=None) -> None:
        """Override joint-space PD gains on the sim. Unused for now; kept as a
        tuning hook once we settle on joint kp/kd for joint_pd replay."""
        if joint_ids is None:
            joint_ids = self._arm_joint_ids
        n = len(joint_ids)
        if kp is not None:
            self._robot.write_joint_stiffness_to_sim(
                torch.full((self.num_envs, n), float(kp), device=self.device), joint_ids=joint_ids
            )
        if kd is not None:
            self._robot.write_joint_damping_to_sim(
                torch.full((self.num_envs, n), float(kd), device=self.device), joint_ids=joint_ids
            )

    # ------------------------------------------------------------------
    # demo helpers
    # ------------------------------------------------------------------
    def _demo_states(self, episode_idx: int):
        """Demo state dicts for this episode, sliced from the task start index."""
        return self.episodes[episode_idx]["states"][self.start_idx_in_episode:]

    def _build_demo_joint_sequence(self, episode_idx: int) -> torch.Tensor:
        """[T, num_joints] joint targets straight from the demo (no interpolation).

        Same finger-sign convention as ``TestPickItUp.run_replay`` (negate the last
        finger dof)."""
        seq = []
        for state in self._demo_states(episode_idx):
            dof_vals = [v[0] for v in state["franka"]["dof_pos"].values()]
            dof_vals[-1] = -dof_vals[-1]
            seq.append(torch.tensor(dof_vals, device=self.device, dtype=torch.float32))
        return torch.stack(seq)  # [T, num_joints]

    def _write_robot_joints(self, joint_pos: torch.Tensor) -> None:
        """Hard-set robot joint state (kinematic). joint_pos: [num_envs, num_joints]."""
        self._robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))

    def _set_objects_to_state(self, state: dict) -> None:
        """Teleport every object to the poses in a demo state dict (used only at init)."""
        zero_vel = torch.zeros(self.num_envs, 6, device=self.device)
        for obj_name in self.object_names:
            obj = self.rigid_objects[obj_name]
            pos = torch.tensor(state[obj_name]["pos"], dtype=torch.float32, device=self.device)
            rot = torch.tensor(state[obj_name]["rot"], dtype=torch.float32, device=self.device)
            pos = pos.unsqueeze(0).repeat(self.num_envs, 1) + self.scene.env_origins
            rot = rot.unsqueeze(0).repeat(self.num_envs, 1)
            obj.write_root_pose_to_sim(torch.cat([pos, rot], dim=-1))
            obj.write_root_velocity_to_sim(zero_vel)

    # ------------------------------------------------------------------
    # OSC helpers (mirror IsaacLab tutorials/05_controllers/run_osc.py)
    # ------------------------------------------------------------------
    def _osc_states(self):
        """Gather the dynamics quantities the OSC needs, in the root frame."""
        ee_idx = self._osc_ee_body_idx
        jac_idx = ee_idx - 1  # fixed base -> jacobian index offset
        jacobian_w = self._robot.root_physx_view.get_jacobians()[:, jac_idx, :, self._arm_joint_ids]
        mass_matrix = self._robot.root_physx_view.get_generalized_mass_matrices()[:, self._arm_joint_ids, :][
            :, :, self._arm_joint_ids
        ]
        gravity = self._robot.root_physx_view.get_gravity_compensation_forces()[:, self._arm_joint_ids]

        root_rot = matrix_from_quat(quat_inv(self._robot.data.root_quat_w))
        jacobian_b = jacobian_w.clone()
        jacobian_b[:, :3, :] = torch.bmm(root_rot, jacobian_b[:, :3, :])
        jacobian_b[:, 3:, :] = torch.bmm(root_rot, jacobian_b[:, 3:, :])

        root_pos_w = self._robot.data.root_pos_w
        root_quat_w = self._robot.data.root_quat_w
        ee_pos_w = self._robot.data.body_pos_w[:, ee_idx]
        ee_quat_w = self._robot.data.body_quat_w[:, ee_idx]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

        rel_vel_w = self._robot.data.body_vel_w[:, ee_idx, :] - self._robot.data.root_vel_w
        ee_lin_b = quat_apply_inverse(root_quat_w, rel_vel_w[:, 0:3])
        ee_ang_b = quat_apply_inverse(root_quat_w, rel_vel_w[:, 3:6])
        ee_vel_b = torch.cat([ee_lin_b, ee_ang_b], dim=-1)

        joint_pos = self._robot.data.joint_pos[:, self._arm_joint_ids]
        joint_vel = self._robot.data.joint_vel[:, self._arm_joint_ids]
        return jacobian_b, mass_matrix, gravity, ee_pose_b, ee_vel_b, joint_pos, joint_vel

    def _build_ee_pose_targets(self, demo_seq: torch.Tensor) -> torch.Tensor:
        """FK every demo frame -> target EE pose in root frame. [T, num_envs, 7].

        Kinematically writes demo joints and reads the EE body pose. Leaves the
        robot at the last frame; caller is expected to reset afterwards."""
        ee_idx = self._osc_ee_body_idx
        targets = []
        for t in range(demo_seq.shape[0]):
            jp = demo_seq[t].unsqueeze(0).repeat(self.num_envs, 1)
            self._write_robot_joints(jp)
            self.scene.update(dt=self.physics_dt)
            root_pos_w = self._robot.data.root_pos_w
            root_quat_w = self._robot.data.root_quat_w
            ee_pos_w = self._robot.data.body_pos_w[:, ee_idx]
            ee_quat_w = self._robot.data.body_quat_w[:, ee_idx]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
            targets.append(torch.cat([ee_pos_b, ee_quat_b], dim=-1))
        return torch.stack(targets)  # [T, num_envs, 7]

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def run_replay_one_env(self, log_dir: str, episode_idx: int = 0, render: bool = False,
                           write_report: bool = True, record_dir: str | None = None) -> dict:
        # NOTE: decorated with inference_mode so EVERY call (incl. the setup
        # _reset_idx) runs inside inference mode. Without this, replaying a second
        # episode in the same env fails: the first episode turns the data buffers
        # into inference tensors, which can't be updated in-place outside inference
        # mode. The inner `with torch.inference_mode()` below is now a harmless nest.
        print(f"[replay_one_env] log_dir={log_dir} episode={episode_idx} "
              f"controller={self.REPLAY_CONTROLLER}")
        assert self.num_envs == 1, f"run_replay_one_env expects num_envs==1, got {self.num_envs}"

        # joint index bookkeeping
        self._arm_joint_ids = self._robot.find_joints("panda_joint[1-7]")[0]
        self._finger_joint_ids = self._robot.find_joints("panda_finger_joint.*")[0]
        self._osc_ee_body_idx = self._robot.find_bodies(self.OSC_EE_BODY)[0][0]

        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._reset_idx(env_ids)

        demo_seq = self._build_demo_joint_sequence(episode_idx)  # [T, num_joints]
        T = demo_seq.shape[0]
        demo_states = self._demo_states(episode_idx)

        if self.REPLAY_CONTROLLER == "osc":
            ee_pose_targets = self._build_ee_pose_targets(demo_seq)  # [T, num_envs, 7]

        # set the whole scene to demo frame 0
        self._write_robot_joints(demo_seq[0].unsqueeze(0).repeat(self.num_envs, 1))
        self._set_objects_to_state(demo_states[0])
        self.scene.update(dt=self.physics_dt)

        writer = SummaryWriter(log_dir)
        obj_z0 = self.target_object.data.root_pos_w[0, 2].item()

        osc = None
        if self.REPLAY_CONTROLLER == "osc":
            # zero arm PD gains so OSC torque control is in charge; fingers keep PD
            zeros = torch.zeros(self.num_envs, len(self._arm_joint_ids), device=self.device)
            self._robot.write_joint_stiffness_to_sim(zeros, joint_ids=self._arm_joint_ids)
            self._robot.write_joint_damping_to_sim(zeros, joint_ids=self._arm_joint_ids)
            osc_cfg = OperationalSpaceControllerCfg(
                target_types=["pose_abs"],
                impedance_mode="fixed",
                inertial_dynamics_decoupling=True,
                gravity_compensation=True,
                motion_stiffness_task=self.OSC_KP,
                motion_damping_ratio_task=self.OSC_DAMPING_RATIO,
                nullspace_control="position",
            )
            osc = OperationalSpaceController(osc_cfg, num_envs=self.num_envs, device=self.device)
            joint_centers = torch.mean(
                self._robot.data.soft_joint_pos_limits[:, self._arm_joint_ids, :], dim=-1
            )
        elif self.PD_KP is not None or self.PD_KD is not None:
            # joint_pd with overridden arm gains (soft<->rigid sweep)
            self._set_replay_gains(kp=self.PD_KP, kd=self.PD_KD)

        # report accumulators
        terminated_step = -1
        grasped_step = -1
        max_obj_z = obj_z0
        track_err_sum = 0.0

        # recording buffers (one capture per demo frame -> 20 fps)
        recording = record_dir is not None and getattr(self.cfg, "record_cameras", False)
        cam_frames = {name: [] for name in self._record_cams} if recording else None
        states_log, actions_log = [], []
        if recording:
            self._aim_agentview()

        with torch.inference_mode():
            for t in range(T):
                target_joints = demo_seq[t].unsqueeze(0).repeat(self.num_envs, 1)

                if self.REPLAY_CONTROLLER == "osc":
                    ee_target = ee_pose_targets[t]                      # [num_envs, 7]
                    finger_target = target_joints[:, self._finger_joint_ids]
                    for _ in range(self.REPLAY_STEPS_PER_FRAME):
                        (jacobian_b, mass_matrix, gravity, ee_pose_b,
                         ee_vel_b, joint_pos, joint_vel) = self._osc_states()
                        osc.set_command(command=ee_target, current_ee_pose_b=ee_pose_b)
                        efforts = osc.compute(
                            jacobian_b=jacobian_b,
                            current_ee_pose_b=ee_pose_b,
                            current_ee_vel_b=ee_vel_b,
                            mass_matrix=mass_matrix,
                            gravity=gravity,
                            current_joint_pos=joint_pos,
                            current_joint_vel=joint_vel,
                            nullspace_joint_pos_target=joint_centers,
                        )
                        self._robot.set_joint_effort_target(efforts, joint_ids=self._arm_joint_ids)
                        self._robot.set_joint_position_target(finger_target, joint_ids=self._finger_joint_ids)
                        self.scene.write_data_to_sim()
                        self.sim.step(render=False)
                        self.scene.update(dt=self.physics_dt)
                else:  # joint_pd
                    self._robot.set_joint_position_target(target_joints)
                    for _ in range(self.REPLAY_STEPS_PER_FRAME):
                        self.scene.write_data_to_sim()
                        self.sim.step(render=False)
                        self.scene.update(dt=self.physics_dt)

                if render:
                    self.sim.render()

                # update task flags + observations (also refreshes robot_grasp_pos etc.)
                terminated, truncated = self._get_dones()
                _ = self._get_observations()

                # ---- record ----
                obj_z = self.target_object.data.root_pos_w[0, 2].item()
                max_obj_z = max(max_obj_z, obj_z)
                track_err = torch.norm(
                    self._robot.data.joint_pos[0, self._arm_joint_ids] - demo_seq[t, self._arm_joint_ids]
                ).item()
                track_err_sum += track_err
                grasped = bool(self.grasped[0].item())
                term = bool(terminated[0].item())
                if grasped and grasped_step < 0:
                    grasped_step = t
                if term and terminated_step < 0:
                    terminated_step = t

                writer.add_scalar("Replay/terminated", float(term), t)
                writer.add_scalar("Replay/grasped", float(grasped), t)
                writer.add_scalar("Replay/object_z", obj_z, t)
                writer.add_scalar("Replay/arm_track_error", track_err, t)

                # ---- recording: one camera capture per demo frame (20 fps) ----
                if recording:
                    fr = self._capture_frame()
                    for name, img in fr.items():
                        cam_frames[name].append(img)
                    states_log.append(self._robot.data.joint_pos[0].detach().cpu().numpy().copy())
                    actions_log.append(demo_seq[t].detach().cpu().numpy().copy())

            self._reset_idx(env_ids)

        # ------------------------------------------------------------------
        # report (printed AND written to file -- Isaac Sim's hard exit can eat
        # buffered stdout, so persist it)
        # ------------------------------------------------------------------
        lines = [
            "=" * 70,
            f"[replay_one_env] REPORT  (controller={self.REPLAY_CONTROLLER}, "
            f"episode={episode_idx}, frames={T}, steps/frame={self.REPLAY_STEPS_PER_FRAME})",
            f"  terminated fired : {'yes @ frame ' + str(terminated_step) if terminated_step >= 0 else 'NO'}",
            f"  grasp detected   : {'yes @ frame ' + str(grasped_step) if grasped_step >= 0 else 'NO'}",
            f"  object z         : start={obj_z0:.3f}  max={max_obj_z:.3f}  (lift={max_obj_z - obj_z0:+.3f})",
            f"  mean arm track err (rad-norm): {track_err_sum / max(T, 1):.4f}",
            "  NOTE: demo joints were generated under MuJoCo OSC control; a",
            "        non-terminating replay is informative, not necessarily a bug.",
            "=" * 70,
            "[replay_one_env] finished.",
        ]
        report = "\n".join(lines)
        if write_report:
            print(report, flush=True)
            with open(os.path.join(log_dir, "report.txt"), "w") as f:
                f.write(report + "\n")

        success = terminated_step >= 0
        stats = {
            "episode": episode_idx,
            "controller": self.REPLAY_CONTROLLER,
            "frames": T,
            "terminated_step": terminated_step,
            "grasped_step": grasped_step,
            "obj_z_start": obj_z0,
            "obj_z_max": max_obj_z,
            "lift": max_obj_z - obj_z0,
            "mean_track_err": track_err_sum / max(T, 1),
            "success": success,
        }
        if recording:
            stats.update(
                self._write_episode_recording(record_dir, episode_idx, success,
                                              cam_frames, states_log, actions_log)
            )
        return stats

    # ------------------------------------------------------------------
    # recording: encode per-episode mp4s + state/action arrays
    # ------------------------------------------------------------------
    def _gains_dict(self) -> dict:
        if self.REPLAY_CONTROLLER == "osc":
            return {"osc_kp": self.OSC_KP, "osc_damping_ratio": self.OSC_DAMPING_RATIO}
        return {"pd_kp": self.PD_KP, "pd_kd": self.PD_KD}

    def _write_episode_recording(self, record_dir, episode_idx, success,
                                 cam_frames, states_log, actions_log) -> dict:
        import imageio
        import numpy as np

        base_task = f"pick up the {self.target_object_name} and put it in the {self.target_site_name}"
        task = base_task if success else base_task + " unsuccessful"
        split = "successful" if success else "unsuccessful"
        ep_dir = os.path.join(record_dir, split, f"ep{episode_idx:02d}")
        os.makedirs(ep_dir, exist_ok=True)

        cam_names = list(cam_frames.keys())
        video_paths = {}
        for name in cam_names:
            frames = cam_frames[name]
            path = os.path.join(ep_dir, f"{name}.mp4")
            imageio.mimwrite(path, frames, fps=20, macro_block_size=1, codec="libx264")
            video_paths[name] = os.path.relpath(path, record_dir)
        # side-by-side (cameras share H) for quick eyeballing
        if len(cam_names) == 2:
            n = min(len(cam_frames[cam_names[0]]), len(cam_frames[cam_names[1]]))
            sbs = [np.concatenate([cam_frames[cam_names[0]][i], cam_frames[cam_names[1]][i]], axis=1)
                   for i in range(n)]
            sbs_path = os.path.join(ep_dir, "sidebyside.mp4")
            imageio.mimwrite(sbs_path, sbs, fps=20, macro_block_size=1, codec="libx264")
            video_paths["sidebyside"] = os.path.relpath(sbs_path, record_dir)

        states = np.asarray(states_log, dtype=np.float32)   # [T, 9]
        actions = np.asarray(actions_log, dtype=np.float32)  # [T, 9]
        timestamps = (np.arange(len(states), dtype=np.float32) / 20.0)
        np.savez(os.path.join(ep_dir, "arrays.npz"),
                 observation_state=states, action=actions, timestamp=timestamps)

        ep_meta = {
            "episode_index": episode_idx,
            "success": bool(success),
            "task": task,
            "length": int(len(states)),
            "fps": 20,
            "cameras": cam_names,
            "videos": video_paths,
            "controller": self.REPLAY_CONTROLLER,
            "gains": self._gains_dict(),
            "state_dim": int(states.shape[1]) if states.ndim == 2 else 0,
            "action_dim": int(actions.shape[1]) if actions.ndim == 2 else 0,
            "state_desc": "robot joint_pos (9): 7 arm + 2 fingers",
            "action_desc": "demo target joint_pos (9)",
        }
        with open(os.path.join(ep_dir, "episode_meta.json"), "w") as f:
            json.dump(ep_meta, f, indent=2)

        return {"task": task, "ep_dir": os.path.relpath(ep_dir, record_dir), "videos": video_paths}

    def _write_dataset_manifest(self, record_dir, results, num) -> None:
        """Write manifest.jsonl (per episode) + meta.json (dataset) for conversion."""
        base_task = f"pick up the {self.target_object_name} and put it in the {self.target_site_name}"
        n_succ = sum(bool(r.get("success")) for r in results)
        with open(os.path.join(record_dir, "manifest.jsonl"), "w") as f:
            for r in results:
                f.write(json.dumps({
                    "episode_index": r["episode"],
                    "success": bool(r.get("success")),
                    "task": r.get("task"),
                    "length": r["frames"],
                    "ep_dir": r.get("ep_dir"),
                    "videos": r.get("videos"),
                    "controller": r["controller"],
                    "gains": self._gains_dict(),
                }) + "\n")
        meta = {
            "fps": 20,
            "cameras": list(self._record_cams.keys()),
            "image_keys": [f"observation.images.{c}" for c in self._record_cams.keys()],
            "state_dim": 9, "action_dim": 9,
            "state_desc": "robot joint_pos (9): 7 arm + 2 fingers",
            "action_desc": "demo target joint_pos (9)",
            "controller": self.REPLAY_CONTROLLER,
            "gains": self._gains_dict(),
            "base_task": base_task,
            "num_episodes": num,
            "num_successful": n_succ,
            "num_unsuccessful": num - n_succ,
            "note": "Convert with scripts/convert_to_lerobot.py in a dedicated lerobot env "
                    "(do NOT install lerobot into eureka; it pulls numpy>=2 and breaks Isaac Sim).",
        }
        with open(os.path.join(record_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[replay_all] recorded {num} episodes ({n_succ} successful, {num - n_succ} unsuccessful) "
              f"-> {record_dir}", flush=True)

    # ------------------------------------------------------------------
    # batch: replay every episode (headless, no render), aggregate a summary
    # ------------------------------------------------------------------
    def run_replay_all(self, log_dir: str, num_episodes: int | None = None,
                       record_dir: str | None = None) -> list[dict]:
        """Replay demo episodes one-by-one (single env) and write an aggregate
        summary + per-episode CSV to ``log_dir``. ``num_episodes`` limits how many
        episodes are replayed (None = all). When ``record_dir`` is set (and the cfg
        has record_cameras=True), each episode also dumps two-camera videos +
        state/action arrays under ``record_dir/<successful|unsuccessful>/``, and a
        ``manifest.jsonl`` + ``meta.json`` are written for LeRobot conversion."""
        os.makedirs(log_dir, exist_ok=True)
        num = len(self.episodes) if num_episodes is None else min(num_episodes, len(self.episodes))
        recording = record_dir is not None and getattr(self.cfg, "record_cameras", False)
        if recording:
            os.makedirs(record_dir, exist_ok=True)
        print(f"[replay_all] replaying {num} episodes, controller={self.REPLAY_CONTROLLER}"
              f"{', recording cameras' if recording else ''}", flush=True)

        results = []
        for ep in range(num):
            stats = self.run_replay_one_env(log_dir, episode_idx=ep, render=False,
                                            write_report=False, record_dir=record_dir)
            results.append(stats)
            print(f"  ep {ep:>3}: term={'F' + str(stats['terminated_step']) if stats['terminated_step'] >= 0 else '-':<6} "
                  f"grasp={'F' + str(stats['grasped_step']) if stats['grasped_step'] >= 0 else '-':<6} "
                  f"lift={stats['lift']:+.3f}"
                  f"{'  ' + ('OK ' if stats.get('success') else 'FAIL') if recording else ''}", flush=True)

        if recording:
            self._write_dataset_manifest(record_dir, results, num)

        n_term = sum(r["terminated_step"] >= 0 for r in results)
        n_grasp = sum(r["grasped_step"] >= 0 for r in results)
        mean_lift = sum(r["lift"] for r in results) / max(num, 1)
        mean_err = sum(r["mean_track_err"] for r in results) / max(num, 1)

        # per-episode CSV
        csv_path = os.path.join(log_dir, "summary.csv")
        with open(csv_path, "w") as f:
            f.write("episode,terminated_step,grasped_step,obj_z_start,obj_z_max,lift,mean_track_err\n")
            for r in results:
                f.write(f"{r['episode']},{r['terminated_step']},{r['grasped_step']},"
                        f"{r['obj_z_start']:.4f},{r['obj_z_max']:.4f},{r['lift']:.4f},{r['mean_track_err']:.4f}\n")

        summary = "\n".join([
            "=" * 70,
            f"[replay_all] SUMMARY  (controller={self.REPLAY_CONTROLLER}, episodes={num}, "
            f"steps/frame={self.REPLAY_STEPS_PER_FRAME})",
            f"  terminated (task success) : {n_term}/{num}  ({100.0 * n_term / max(num, 1):.1f}%)",
            f"  grasp detected            : {n_grasp}/{num}  ({100.0 * n_grasp / max(num, 1):.1f}%)",
            f"  mean object lift (m)      : {mean_lift:+.3f}",
            f"  mean arm track err        : {mean_err:.4f}",
            f"  per-episode CSV           : {csv_path}",
            "=" * 70,
            "[replay_all] finished.",
        ])
        print(summary, flush=True)
        with open(os.path.join(log_dir, "summary.txt"), "w") as f:
            f.write(summary + "\n")
        return results
