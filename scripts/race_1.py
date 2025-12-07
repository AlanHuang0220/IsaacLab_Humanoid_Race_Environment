# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher
import sys

parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--agent_models",
    nargs="*",
    type=str,
    help="Path to the model file for each agent. The order should match the agent names in the environment.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
import torch
import io
import os
import omni


import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import IsaacLab_Humanoid_Race_Environment.tasks

from skrl.models.torch import GaussianMixin, DeterministicMixin, Model
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.torch import wrap_env
import yaml
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from skrl.utils.runner.torch import Runner
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.envs import (
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)


agent_cfg_entry_point = "skrl_cfg_entry_point"

@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Random actions agent with Isaac Lab environment."""

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg)

    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # Create a runner for each agent
    runners = {}
    agent_names = env.possible_agents
    agent_models = args_cli.agent_models if args_cli.agent_models else []

    if len(agent_models) > len(agent_names):
        raise ValueError(
            f"Number of agent models ({len(agent_models)}) exceeds the number of agents ({len(agent_names)})."
        )
    elif len(agent_models) < len(agent_names):
        print(f"[WARNING]: Only {len(agent_models)} models provided for {len(agent_names)} agents. Remaining agents will be stationary.")

    for i in range(len(agent_models)):
        agent_name = agent_names[i]
        runner = Runner(env, experiment_cfg)
        runner.agent.load(agent_models[i])
        runner.agent.set_running_mode("eval")
        runners[agent_name] = runner

    # reset environment
    obs, _ = env.reset()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            actions = {}
            actions = {}
            for agent_name in agent_names:
                if agent_name in runners:
                    outputs = runners[agent_name].agent.act(obs[agent_name], timestep=0, timesteps=0)
                    actions[agent_name] = outputs[-1].get("mean_actions", outputs[0])
                else:
                    # Zero action for stationary agent
                    # Get action dimension from environment config
                    action_dim = env_cfg.action_spaces[agent_name]
                    actions[agent_name] = torch.zeros((env.num_envs, action_dim), device=env.device)
            # apply actions
            obs, _, _, _, _ = env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
