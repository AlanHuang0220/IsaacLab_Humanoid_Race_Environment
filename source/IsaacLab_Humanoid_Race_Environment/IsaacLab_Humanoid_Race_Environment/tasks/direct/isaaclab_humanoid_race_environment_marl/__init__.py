# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


# gym.register(
#     id="Template-Isaaclab-Humanoid-Race-Environment-Marl-Direct-v0",
#     entry_point=f"{__name__}.isaaclab_humanoid_race_environment_marl_env:IsaaclabHumanoidRaceEnvironmentMarlEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.isaaclab_humanoid_race_environment_marl_env_cfg:IsaaclabHumanoidRaceEnvironmentMarlEnvCfg",
#         "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
#     },
# )

gym.register(
    id="Race-Track1",
    entry_point=f"{__name__}.race_track_1_env:RaceTrack1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.race_track_1_env_config:RaceTrack1EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Race-Track0",
    entry_point=f"{__name__}.race_track_0_env:RaceTrack0Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.race_track_0_env_config:RaceTrack0EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)