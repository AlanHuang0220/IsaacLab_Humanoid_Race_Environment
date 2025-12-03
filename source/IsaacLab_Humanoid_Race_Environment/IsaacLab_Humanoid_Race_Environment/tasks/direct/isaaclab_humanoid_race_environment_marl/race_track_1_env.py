# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, AssetBase
from isaaclab.envs import DirectMARLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import sample_uniform, quat_apply_inverse, yaw_quat, \
                                quat_from_euler_xyz, quat_mul, quat_from_angle_axis, \
                                wrap_to_pi

from .race_track_1_env_config import RaceTrack1EnvCfg

def define_markers() -> VisualizationMarkers:
    """Define markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
                "forward": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
                ),
                "command": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)


class RaceTrack1Env(DirectMARLEnv):
    cfg: RaceTrack1EnvCfg

    def __init__(self, cfg: RaceTrack1EnvCfg, render_mode: str | None = None, **kwargs):
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


    def _setup_scene(self):
        # create assets
        cfg = sim_utils.UsdFileCfg(usd_path=f"D:/RL2025/IsaacLab_Humanoid_Race_Environment/source/IsaacLab_Humanoid_Race_Environment/IsaacLab_Humanoid_Race_Environment/tasks/direct/isaaclab_humanoid_race_environment_marl/aseets/ttt.usd")
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

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # initialize commands
        self.target_heading = torch.zeros(self.cfg.scene.num_envs, device=self.device)
                             

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self.actions = actions
        
        # update z angular velocity commands for all agents
        for agent_name in self.cfg.possible_agents:
            robot = self.scene.articulations[agent_name]
            current_heading = robot.data.heading_w
            heading_error = wrap_to_pi(self.target_heading - current_heading)
            self.commands[agent_name][:, 2] = torch.clamp(
                heading_error * self.heading_control_stiffness,
                min=-1.0,
                max=1.0
            )

    def _apply_action(self) -> None:
        for agent_name in self.cfg.possible_agents:
            robot = self.scene.articulations[agent_name]
            dof_indices = self._dof_indices[agent_name]
            default_joint_pos = robot.data.default_joint_pos[:, dof_indices]
            target_joint_pos = default_joint_pos + self.action_scale * self.actions[agent_name]
            robot.set_joint_position_target(target_joint_pos, joint_ids=dof_indices)
            self._last_actions[agent_name][:] = self.actions[agent_name]

    def _get_observations(self) -> dict[str, torch.Tensor]:
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

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        # TODO 先給個假資料
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


@torch.jit.script
def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi
