# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg, AssetBase
from isaaclab.sim import CollisionPropertiesCfg
from isaaclab.envs import DirectMARLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import sample_uniform, quat_apply_inverse, yaw_quat, \
                                quat_from_euler_xyz, quat_mul, quat_from_angle_axis, \
                                wrap_to_pi

from .race_track_0_env_config import RaceTrack0EnvCfg

def define_markers(prim_path: str, color: tuple[float, float, float]) -> VisualizationMarkers:
    """Define markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
                "arrow": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)
class RaceTrack0Env(DirectMARLEnv):
    cfg: RaceTrack0EnvCfg


    def __init__(self, cfg: RaceTrack0EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._dof_indices = {}
        self._last_actions = {}
        self.commands = {}
        for agent_name in self.cfg.possible_agents:
            robot = self.scene.articulations[agent_name]
            self._dof_indices[agent_name], _ = robot.find_joints([".*"])
            self._last_actions[agent_name] = torch.zeros((self.num_envs, self.cfg.action_spaces[agent_name]), device=self.device)
            self.commands[agent_name] = torch.zeros((self.cfg.scene.num_envs, 3), device=self.device)
        self.heading_control_stiffness = 0.5 # large value makes the robot follow the command more rapidly
        self.action_scale = 0.5
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_scale = torch.tensor([0.25, 8.0, 0.5], device=self.device)
        self.rankings = {env_id: [] for env_id in range(self.num_envs)}
        self.min_dist_to_target = {agent: torch.full((self.num_envs,), float('inf'), device=self.device) for agent in self.cfg.possible_agents}


    def _setup_scene(self):
        # create assets
        cfg = sim_utils.UsdFileCfg(usd_path=f"../source/IsaacLab_Humanoid_Race_Environment/IsaacLab_Humanoid_Race_Environment/tasks/direct/isaaclab_humanoid_race_environment_marl/aseets/running_track0.usd")
        cfg.func("/World/ground", cfg, translation=(0.0, 0.0, 0.0))

        # Loop through all possible agents to create and add them to the scene
        for i, agent_name in enumerate(self.cfg.possible_agents, 1):
            robot_cfg = getattr(self.cfg, f"robot{i}_cfg")
            height_scanner_cfg = getattr(self.cfg, f"height_scanner{i}")
            ankle_contact_cfg = getattr(self.cfg, f"ankle_contact_forces{i}")
            torso_contact_cfg = getattr(self.cfg, f"torso_contact_forces{i}")
            self.scene.articulations[agent_name] = Articulation(robot_cfg)
            self.scene.sensors[f"height_scanner_{agent_name}"] = RayCaster(height_scanner_cfg)
            self.scene.sensors[f"ankle_contact_{agent_name}"] = ContactSensor(ankle_contact_cfg)
            self.scene.sensors[f"torso_contact_{agent_name}"] = ContactSensor(torso_contact_cfg)

        # Goal Object
        goal_cfg = RigidObjectCfg(
            prim_path="/World/Goal",
            spawn=sim_utils.CuboidCfg(
                size=(0.25, 8.0, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(100.0, 100.0, 0.0)),
        )
        self.scene.rigid_objects["goal"] = RigidObject(goal_cfg)

        # # 地板
        # self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        # self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        # self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # add markers
        # Goal marker (Cyan)
        self.goal_marker = define_markers("/Visuals/Goal", (0.0, 1.0, 1.0))
        # Command marker (Red)
        self.cmd_marker = define_markers("/Visuals/Command", (1.0, 0.0, 0.0))

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # initialize commands
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)
                             

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self.actions = actions
        
        env_0_log_strs = []
        # update z angular velocity commands for all agents
        # update z angular velocity commands for all agents
        # update z angular velocity commands for all agents
        for agent_name in self.cfg.possible_agents:
            robot = self.scene.articulations[agent_name]
            # calculate vector to target
            target_vec = self.target_pos - robot.data.root_pos_w
            # calculate desired heading
            desired_yaw = torch.atan2(target_vec[:, 1], target_vec[:, 0])
            
            current_heading = robot.data.heading_w
            heading_error = wrap_to_pi(desired_yaw - current_heading)
            
            # set forward velocity command (e.g. 1.0 m/s) if far from target
            # simple logic: always move forward if target is far enough
            # Calculate distance to goal boundary (Box Distance)
            rel_pos_abs = torch.abs(target_vec[:, :2])
            half_size = self.goal_scale[:2] / 2.0
            dist_outside = torch.maximum(rel_pos_abs - half_size, torch.tensor(0.0, device=self.device))
            dist_to_target = torch.norm(dist_outside, dim=1)
            
            # Update min distance
            self.min_dist_to_target[agent_name] = torch.min(self.min_dist_to_target[agent_name], dist_to_target)
            env_0_log_strs.append(f"{agent_name}: {self.min_dist_to_target[agent_name][0].item():.2f}")
            
            self.commands[agent_name][:, 0] = torch.where(dist_to_target > 0.5, 1.0, 0.0)
            
            self.commands[agent_name][:, 2] = torch.clamp(
                heading_error * self.heading_control_stiffness,
                min=-1.0,
                max=1.0
            )

        print(f"[INFO] Min Dist (Env 0): " + ", ".join(env_0_log_strs))

        # Visualize command arrows
        all_cmd_pos = []
        all_cmd_orient = []
        for agent_name in self.cfg.possible_agents:
            robot = self.scene.articulations[agent_name]
            # Command arrow position: 2m above robot
            cmd_pos = robot.data.root_pos_w.clone()
            cmd_pos[:, 2] += 2.0
            all_cmd_pos.append(cmd_pos)
            
            # Command arrow orientation: pointing to target
            target_vec = self.target_pos - robot.data.root_pos_w
            desired_yaw = torch.atan2(target_vec[:, 1], target_vec[:, 0])
            # Convert yaw to quaternion (assuming Z-up)
            # We need to construct a quaternion from Euler angles (roll=0, pitch=0, yaw=desired_yaw)
            # Since we don't have a direct euler_to_quat for batch, we can use existing utilities or simple math
            # Here we use quat_from_angle_axis for Z axis rotation
            zeros = torch.zeros_like(desired_yaw)
            cmd_orient = quat_from_euler_xyz(zeros, zeros, desired_yaw)
            all_cmd_orient.append(cmd_orient)
            
        self.cmd_marker.visualize(translations=torch.cat(all_cmd_pos), orientations=torch.cat(all_cmd_orient))

    def _apply_action(self) -> None:
        for agent_name in self.cfg.possible_agents:
            robot = self.scene.articulations[agent_name]
            dof_indices = self._dof_indices[agent_name]
            default_joint_pos = robot.data.default_joint_pos[:, dof_indices]
            target_joint_pos = default_joint_pos + self.action_scale * self.actions[agent_name]
            robot.set_joint_position_target(target_joint_pos, joint_ids=dof_indices)
            self._last_actions[agent_name][:] = self.actions[agent_name]

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._check_rankings()
        observations = {}
        for agent_name in self.cfg.possible_agents:
            robot = self.scene.articulations[agent_name]
            ray_caster = self.scene.sensors[f"height_scanner_{agent_name}"]
            height_scan = ray_caster.data.pos_w[:, 2].unsqueeze(1) - ray_caster.data.ray_hits_w[..., 2] - 0.5
            height_scan = torch.clip(height_scan, -1.0, 1.0)
            obs_list = [
                robot.data.root_com_lin_vel_b,
                robot.data.root_com_ang_vel_b,
                robot.data.projected_gravity_b,
                self.commands[agent_name],
                robot.data.joint_pos,
                robot.data.joint_vel,
                self._last_actions[agent_name],
                height_scan
            ]
            observations[agent_name] = torch.hstack(obs_list)
        return observations

    def _check_rankings(self):
        """Check if agents have reached the goal and update rankings."""
        for agent_name in self.cfg.possible_agents:
            robot = self.scene.articulations[agent_name]
            
            # Check distance to target (Bounding Box Check)
            # Goal is at self.target_pos
            # We check if robot is within the box defined by goal_scale
            # User requested no margin, so we check strict bounds
            margin = 0.0
            rel_pos = robot.data.root_pos_w - self.target_pos
            
            # Check X and Y bounds
            half_size = self.goal_scale / 2.0
            in_x = torch.abs(rel_pos[:, 0]) < (half_size[0] + margin)
            in_y = torch.abs(rel_pos[:, 1]) < (half_size[1] + margin)
            
            has_reached_goal = in_x & in_y
            
            # Iterate through environments to update rankings
            # Note: This loop is slow in python but fine for verification/small num_envs
            # For large scale training, this should be vectorized or moved to warp/cuda
            contact_indices = torch.nonzero(has_reached_goal).flatten()
            for env_id in contact_indices:
                env_id = env_id.item()
                if agent_name not in self.rankings[env_id]:
                    self.rankings[env_id].append(agent_name)
                    rank = len(self.rankings[env_id])
                    print(f"[INFO] Env {env_id}: {agent_name} finished {rank}!")

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        total_reward = {agent: torch.zeros((self.num_envs, 1), device=self.device) for agent in self.cfg.possible_agents}
        return total_reward

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = {}
        for agent_name in self.cfg.possible_agents:
            torso_sensor = self.scene.sensors[f"torso_contact_{agent_name}"]
            net_contact_forces = torso_sensor.data.net_forces_w_history
            # condition for termination: falling, based on torso contact
            terminated[agent_name] = torch.any(torch.max(torch.norm(net_contact_forces[:, :, [0]], dim=-1), dim=1)[0] > 1.0, dim=1)
        time_outs = {agent: time_out for agent in self.cfg.possible_agents}
        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            # Use the first robot to get all indices, assuming all robots are in all envs
            first_robot_name = self.cfg.possible_agents[0]
            env_ids = self.scene.articulations[first_robot_name]._ALL_INDICES
        super()._reset_idx(env_ids)
        
        # sample new targets for reset environments
        # For now, let's sample random points ahead on X axis, with some Y variation
        # You can customize this range based on your track size
        self.target_pos[env_ids, 0] = 32.0
        self.target_pos[env_ids, 1] = 0.0
        self.target_pos[env_ids, 2] = 0.0 # ground level

        # visualize targets
        self.goal_marker.visualize(self.target_pos)
        
        # Teleport goal object to target position
        # Orientation: Identity
        goal_pos = self.target_pos.clone()
        goal_pos[:, 2] += 0.25 # Lift half size so it sits on ground
        goal_orient = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
        self.scene.rigid_objects["goal"].write_root_pose_to_sim(torch.cat([goal_pos, goal_orient], dim=-1), env_ids)
        
        # Reset rankings for reset environments
        for env_id in env_ids:
            self.rankings[env_id.item()] = []
            
        for agent in self.cfg.possible_agents:
            self.min_dist_to_target[agent][env_ids] = float('inf')


@torch.jit.script
def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi
