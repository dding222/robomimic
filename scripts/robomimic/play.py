# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play and evaluate a trained policy from robomimic.

This script loads a robomimic policy and plays it in an Isaac Lab environment.

Args:
    task: Name of the environment.
    checkpoint: Path to the robomimic policy checkpoint.
    horizon: If provided, override the step horizon of each rollout.
    num_rollouts: If provided, override the number of rollouts.
    seed: Random seed for a single evaluation (default: 101).
    seeds: Optional list of seeds; each runs num_rollouts episodes.
    norm_factor_min: If provided, minimum value of the action space normalization factor.
    norm_factor_max: If provided, maximum value of the action space normalization factor.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

#########################
# command line
#########################

parser = argparse.ArgumentParser(description="Evaluate robomimic policy for Isaac Lab environment.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Pytorch model checkpoint to load.")
parser.add_argument("--horizon", type=int, default=800, help="Step horizon of each rollout.")
parser.add_argument("--num_rollouts", type=int, default=1, help="Number of rollouts per seed.")
seed_args = parser.add_mutually_exclusive_group()
seed_args.add_argument("--seed", type=int, default=101, help="Single evaluation seed. Default: 101.")
seed_args.add_argument("--seeds", type=int, nargs="+", default=None,
                       help="Evaluate each seed and report the mean/std of their success rates.")
parser.add_argument(
    "--norm_factor_min", type=float, default=None, help="Optional: minimum value of the normalization factor."
)
parser.add_argument(
    "--norm_factor_max", type=float, default=None, help="Optional: maximum value of the normalization factor."
)
parser.add_argument("--enable_pinocchio", default=False, action="store_true", help="Enable Pinocchio.")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
evaluation_seeds = args_cli.seeds if args_cli.seeds is not None else [args_cli.seed]
if args_cli.num_rollouts < 1 or args_cli.horizon < 1:
    parser.error("--num_rollouts and --horizon must be positive.")
if any(seed < 0 or seed >= 2**32 for seed in evaluation_seeds):
    parser.error("Seeds must be between 0 and 2**32 - 1.")
if len(set(evaluation_seeds)) != len(evaluation_seeds):
    parser.error("--seeds must contain unique seeds.")

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version
    # installed by IsaacLab and not the one installed by Isaac Sim.
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import random

import gymnasium as gym
import numpy as np
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import torch

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

import isaaclab_tasks  # noqa: F401
import agibot_toy2box  # noqa: F401
from agibot_toy2box.rnn_observations import eef_to_toy
from isaaclab.sim import create_new_stage
from isaaclab_tasks.utils import parse_env_cfg


#########################
# rollout
#########################

def rollout(policy, env, success_term, horizon, device):
    """Perform a single rollout of the policy in the environment.

    Args:
        policy: The robomimicpolicy to play.
        env: The environment to play in.
        horizon: The step horizon of each rollout.
        device: The device to run the policy on.

    Returns:
        success: Whether the rollout succeeded.
        traj: The trajectory of the rollout.
    """
    policy.start_episode()
    obs_dict, _ = env.reset()
    traj = dict(actions=[], obs=[], next_obs=[])

    for i in range(horizon):
        # Prepare observations
        obs = copy.deepcopy(obs_dict["policy"])
        for ob in obs:
            obs[ob] = torch.squeeze(obs[ob])
        if "eef_to_toy" in policy.policy.obs_shapes:
            obs["eef_to_toy"] = eef_to_toy(obs)

        # Check if environment image observations
        if hasattr(env.cfg, "image_obs_list"):
            # Process image observations for robomimic inference
            for image_name in env.cfg.image_obs_list:
                if image_name in obs_dict["policy"].keys():
                    # Convert from chw uint8 to hwc normalized float
                    image = torch.squeeze(obs_dict["policy"][image_name])
                    image = image.permute(2, 0, 1).clone().float()
                    image = image / 255.0
                    image = image.clip(0.0, 1.0)
                    obs[image_name] = image

        traj["obs"].append(obs)

        # Compute actions
        actions = policy(obs)

        # Unnormalize actions
        if args_cli.norm_factor_min is not None and args_cli.norm_factor_max is not None:
            actions = (
                (actions + 1) * (args_cli.norm_factor_max - args_cli.norm_factor_min)
            ) / 2 + args_cli.norm_factor_min

        actions = torch.from_numpy(actions).to(device=device).view(1, env.action_space.shape[1])

        # Apply actions
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        obs = obs_dict["policy"]

        # Record trajectory
        traj["actions"].append(actions.tolist())
        traj["next_obs"].append(obs)

        # Check if rollout was successful
        if bool(success_term.func(env, **success_term.params)[0]):
            return True, traj
        elif terminated or truncated:
            return False, traj

    return False, traj


#########################
# seeded evaluation
#########################

def main():
    """Evaluate independent seeds and summarize success rates and episode lengths."""
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.checkpoint, device=device)
    success_rates = []
    episode_lengths = []
    total_successes = 0

    for seed_index, seed in enumerate(evaluation_seeds):
        if seed_index:
            create_new_stage()
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1,
                                use_fabric=not args_cli.disable_fabric)
        env_cfg.seed = seed
        env_cfg.observations.policy.concatenate_terms = False
        env_cfg.terminations.time_out = None
        env_cfg.recorders = None
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        try:
            env.seed(seed)
            results = []
            print(f"\n[INFO] Evaluation seed={seed}, rollouts={args_cli.num_rollouts}")
            for trial in range(args_cli.num_rollouts):
                print(f"[INFO] Starting trial {trial} (seed={seed})")
                success, traj = rollout(policy, env, success_term, args_cli.horizon, device)
                results.append(success)
                episode_lengths.append(len(traj["actions"]))
                print(f"[INFO] Trial {trial}: {success}\n")

            successes = results.count(True)
            success_rates.append(successes / len(results))
            total_successes += successes
            print(f"[Seed {seed}]")
            print(f"Successful trials: {successes}, out of {len(results)} trials")
            print(f"Success rate: {success_rates[-1]}")
            print(f"Trial Results: {results}\n")
        finally:
            env.close()
            del env

    mean_rate = float(np.mean(success_rates))
    std_rate = float(np.std(success_rates, ddof=0))
    print(f"\nEvaluation seeds: {evaluation_seeds}")
    print(f"Rollouts per seed: {args_cli.num_rollouts}")
    print(f"Total successful trials: {total_successes}, out of {len(episode_lengths)} trials")
    print(f"Success rates by seed: {dict(zip(evaluation_seeds, success_rates))}")
    print(f"Success rate mean ± std (across seeds, ddof=0): {mean_rate:.6f} ± {std_rate:.6f}")
    print(f"Success rate percent: {mean_rate * 100:.2f}% ± {std_rate * 100:.2f} percentage points")
    print(f"Episode length mean ± std (all rollouts, steps, ddof=0): "
          f"{np.mean(episode_lengths):.2f} ± {np.std(episode_lengths, ddof=0):.2f}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
