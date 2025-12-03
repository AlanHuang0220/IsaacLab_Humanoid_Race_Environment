# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from isaaclab_assets.robots.cart_double_pendulum import CART_DOUBLE_PENDULUM_CFG

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns

from isaaclab_assets import G1_MINIMAL_CFG 
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.terrains import TerrainImporterCfg

import isaaclab.sim as sim_utils

@configclass
class RaceTrack0EnvCfg(DirectMARLEnvCfg):
    def __post_init__(self):
        """Post-initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        self.state_space = -1

        # simulation settings
        self.sim: SimulationCfg = SimulationCfg(
            dt=0.005,
            render_interval=self.decimation,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15),
        )

        # scene settings
        self.scene: InteractiveSceneCfg = InteractiveSceneCfg(
            num_envs=4096, env_spacing=3.0, replicate_physics=True, clone_in_fabric=True
        )

        # ground plane
        self.ground_cfg = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"D:/RL2025/IsaacLab_Humanoid_Race_Environment/source/IsaacLab_Humanoid_Race_Environment/IsaacLab_Humanoid_Race_Environment/tasks/direct/isaaclab_humanoid_race_environment_marl/aseets/ttttt.usd",
                scale=(1, 1, 1),
            ),
        )

        # Number of robots
        num_robots = 5
        # Agent names
        self.possible_agents = [f"robot{i}" for i in range(1, num_robots + 1)]
        # Observation and action spaces
        self.observation_spaces = {agent: 310 for agent in self.possible_agents}
        self.action_spaces = {agent: 37 for agent in self.possible_agents}

        # Initial positions for the robots
        initial_positions = [
            (0.0, -1.0, 1.5),
            (0.0, 1.0, 1.5),
            (0.0, 0.0, 1.5),
            (0.0, -2.0, 1.5),
            (0.0, 2.0, 1.5),
        ]

        # Create robot and sensor configurations in a loop
        for i in range(1, num_robots + 1):
            robot_name = f"robot{i}"
            prim_path = f"/World/envs/env_.*/Robot{i}"

            # Robot configuration
            robot_cfg = G1_MINIMAL_CFG.replace(prim_path=prim_path).replace(
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=initial_positions[i - 1],
                    rot=(1.0, 0.0, 0.0, 0.0),
                    joint_pos={".*": 0.0},
                )
            )
            setattr(self, f"robot{i}_cfg", robot_cfg)

            # Height scanner configuration
            height_scanner_cfg = RayCasterCfg(
                prim_path=f"{prim_path}/torso_link",
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
                ray_alignment="yaw",
                pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
                debug_vis=True,
                mesh_prim_paths=["/World/ground"],
                update_period=self.decimation * self.sim.dt,
            )
            setattr(self, f"height_scanner{i}", height_scanner_cfg)

            # Ankle contact sensor configuration
            ankle_contact_cfg = ContactSensorCfg(
                prim_path=f"{prim_path}/.*_ankle_roll_link",
                history_length=3,
                track_air_time=True,
                update_period=self.sim.dt,
            )
            setattr(self, f"ankle_contact_forces{i}", ankle_contact_cfg)

            # Torso contact sensor configuration
            torso_contact_cfg = ContactSensorCfg(
                prim_path=f"{prim_path}/torso_link",
                history_length=3,
                track_air_time=True,
                update_period=self.sim.dt,
            )
            setattr(self, f"torso_contact_forces{i}", torso_contact_cfg)